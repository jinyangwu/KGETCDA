#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jinyang Wu

import torch

import copy
import random
import numpy as np
import pandas as pd
import collections
import scipy.sparse as sp


class DataLoaderBase(object):
    """Load data from training and testing data and knowledge triples"""
    def __init__(self, args):
        self.args = args
        self.fold = args.fold
        self.dataset = args.dataset

        dataset_folder = args.data_dir + args.dataset
        self.train_file = dataset_folder + f'/fold{args.fold}' + '/pairs_train.txt'
        self.test_file = dataset_folder + f'/fold{args.fold}' + '/pairs_test.txt'
        self.kg_file = dataset_folder + '/triples_part.txt'  

        self.ap_train_data, self.train_circ_dict = self.load_ap(self.train_file)
        self.ap_test_data, self.test_circ_dict = self.load_ap(self.test_file)
        self.statistic_ap()

    def load_ap(self, filename):
        circ = []
        dis = []
        circ_dis_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split('\t')]

            if len(inter) > 1:
                circ_id, dis_ids = inter[0], inter[1:]
                dis_ids = list(set(dis_ids))

                for dis_id in dis_ids:
                    circ.append(circ_id)
                    dis.append(dis_id)
                circ_dis_dict[circ_id] = dis_ids

        circ = np.array(circ, dtype=np.int32)
        dis = np.array(dis, dtype=np.int32)
        return (circ, dis), circ_dis_dict

    def statistic_ap(self):
        self.n_circRNAs = max(max(self.ap_train_data[0]), max(self.ap_test_data[0])) + 1
        self.n_diseases = max(max(self.ap_train_data[1]), max(self.ap_test_data[1])) + 1
        self.n_ap_train = len(self.ap_train_data[0])
        self.n_ap_test = len(self.ap_test_data[0])

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()  
        return kg_data      # triples of (h, r, t)

    def sample_pos_dis_for_circ(self, circ_dict, circ_id, n_sample_pos_dis):
        pos_dis = circ_dict[circ_id]
        n_pos_dis = len(pos_dis)

        sample_pos_dis = []
        while True:
            if len(sample_pos_dis) == n_sample_pos_dis:
                break

            pos_dis_idx = np.random.randint(low=0, high=n_pos_dis, size=1)[0]
            pos_dis_id = pos_dis[pos_dis_idx]
            if pos_dis_id not in sample_pos_dis:
                sample_pos_dis.append(pos_dis_id)
        return sample_pos_dis      # returns n_sample_pos_dis positive diseases matching circRNAs 

    def sample_neg_dis_for_circ(self, circ_dict, circ_id, n_sample_neg_dis):
        pos_dis = circ_dict[circ_id]

        sample_neg_dis = []
        while True:
            if len(sample_neg_dis) == n_sample_neg_dis:
                break

            neg_dis_id = np.random.randint(low=self.n_circRNAs, high=self.n_circRNAs+self.n_diseases, size=1)[0]
            if neg_dis_id not in pos_dis and neg_dis_id not in sample_neg_dis:
                sample_neg_dis.append(neg_dis_id)
        return sample_neg_dis      # returns n_sample_pos_dis negative diseases of circRNAs

    def generate_ap_batch(self, circ_dict, batch_size): 
        exist_circ = circ_dict.keys()
        if batch_size <= len(exist_circ):  
            # Get batch_size circ_id of training_data in all entities
            batch_circ = random.sample(exist_circ, batch_size)
        else:  
            # Randomly select batch_size ids of exist_circ (ids of circRNAs of training data in all entities)
            batch_circ = [random.choice(exist_circ) for _ in range(batch_size)]

        batch_pos_dis, batch_neg_dis = [], []
        for c in batch_circ:   # 1:1 generate 1 pos disease and 1 neg disease of every circRNA in a batch
            batch_pos_dis += self.sample_pos_dis_for_circ(circ_dict, c, 1)  
            batch_neg_dis += self.sample_neg_dis_for_circ(circ_dict, c, 1)  

        batch_circ = torch.LongTensor(batch_circ)          # (batch_size,)
        batch_pos_dis = torch.LongTensor(batch_pos_dis)    # (batch_size,)
        batch_neg_dis = torch.LongTensor(batch_neg_dis)  
        return batch_circ, batch_pos_dis, batch_neg_dis

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]  # generate 1 pos sample of a h from triples in training data and its reverse triples
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]  # generate 1 neg sample of a h not in training data and its reverse triples

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]  # generate a neg-sample just by replacing its tail entity
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_h_dict, batch_size, highest_neg_idx):
        """generate a kg_batch_size of kg part
        Args:
            kg_h_dict(dict): {h1: [(t1, r1), (t2, r2), ...], h2: [], ...}
        """
        exist_heads = kg_h_dict.keys()  
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(list(exist_heads)) for _ in range(batch_size)]

        # 1:1 generate 1 positive disease and 1 negative disease of every head entity in a batch
        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_h_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail
            neg_tail = self.sample_neg_triples_for_h(kg_h_dict, h, relation[0], 1, highest_neg_idx)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)           # (kg_batch_size,)
        batch_relation = torch.LongTensor(batch_relation)   # (kg_batch_size,)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)   # (kg_batch_size,)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)   # (kg_batch_size,)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


class DataLoaderKGETCDA(DataLoaderBase):
    """Load data and constrcut all triples by preprocessing train data and triples"""
    def __init__(self, args, logger):
        super(DataLoaderKGETCDA, self).__init__(args)
        self.ap_batch_size = args.ap_batch_size      
        self.kg_batch_size = args.kg_batch_size      

        # Initial triples contains 4 relations, i.e. circ-miRNA, miRNA-disease, miRNA-lncRNA, lncRNA-disease
        kg_data = self.load_kg(self.kg_file)

        # Add reverse relations and circ-disease pairs of training data to construct kg_data of 10 relations
        self.construct_data(kg_data)
        self.print_info(logger)

        # Construct 10 adjacency matrix and each for a relation
        self.create_adjacency_r_dict()

        # Get a sparse laplacian matrix of 10 relations, respectively and then integrate them
        self.create_laplacian_dict()

    def construct_data(self, kg_data):
        """Construct kg_data of 10 relations from original data of 4 relations
        Args:
            kg_data(pd.DataFrame): triples containing 4 relations, (h,r,t)
        """
        
        # Add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = copy.deepcopy(kg_data)
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # Re-map circ id i.e. construct 10 relations
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_other_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1 - self.n_circRNAs  # miRNA, lncRNA, disease
        self.n_all_entities = self.n_circRNAs + self.n_other_entities  # circRNA, miRNA, lncRNA, disease

        self.ap_train_data = (self.ap_train_data[0].astype(np.int32), np.array(list(map(lambda d: d + self.n_circRNAs, self.ap_train_data[1]))).astype(np.int32))
        self.ap_test_data = (self.ap_test_data[0].astype(np.int32), np.array(list(map(lambda d: d + self.n_circRNAs, self.ap_test_data[1]))).astype(np.int32))

        self.train_circ_dict = {k : np.unique(v+self.n_circRNAs).astype(np.int32) for k, v in self.train_circ_dict.items()}
        self.test_circ_dict = {k : np.unique(v+self.n_circRNAs).astype(np.int32) for k, v in self.test_circ_dict.items()}

        # Add circRNA-disease to kg data
        ap2kg_train_data = pd.DataFrame(np.zeros((self.n_ap_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        ap2kg_train_data['h'], ap2kg_train_data['t'] = self.ap_train_data[0], self.ap_train_data[1]
        inverse_ap2kg_train_data = pd.DataFrame(np.ones((self.n_ap_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_ap2kg_train_data['h'], inverse_ap2kg_train_data['t'] = self.ap_train_data[1], self.ap_train_data[0]

        # Get all kg data which contains 10 relations, i.e. circ-disease, circ-miRNA, miRNA-disease, miRNA-lncRNA,
        # lncRNA-disease and their reverse relations
        self.kg_train_data = pd.concat([kg_data, ap2kg_train_data, inverse_ap2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        h_list, t_list, r_list = [], [], []
        self.train_h_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)
            self.train_h_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def convert_coo2tensor(self, coo_matrix):
        """convert to sparse tensor"""
        values = coo_matrix.data
        indices = np.vstack((coo_matrix.row, coo_matrix.col))

        indices_tensor = torch.LongTensor(indices)
        values_tensor = torch.FloatTensor(values)
        shape = coo_matrix.shape
        return torch.sparse.FloatTensor(indices_tensor, values_tensor, torch.Size(shape))

    def create_adjacency_r_dict(self):
        """Construct 10 adjacency matrix and each for a relation"""
        self.adjacency_r_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]      # get h in all triples of relation r
            cols = [e[1] for e in ht_list]      # get t in all triples of relation r
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_all_entities, self.n_all_entities))
            self.adjacency_r_dict[r] = adj

    def create_laplacian_dict(self):

        def asymmetric_norm_lap(adj):
            """Asymmetry laplacian matrix of GCN"""
            rowsum = np.array(adj.sum(axis=1))
            np.seterr(divide='ignore', invalid='ignore')  
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        norm_lap_func = asymmetric_norm_lap
        self.laplacian_r_dict = {}
        for r, adj in self.adjacency_r_dict.items():
            self.laplacian_r_dict[r] = norm_lap_func(adj)  # get laplacian matrix of every relation i.e. normalized adj matrix

        A_in = sum(self.laplacian_r_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())  # merge into a large laplacian matrix of 10 relations

    def print_info(self, logging):
        logging.info('n_circRNAs:        %d' % self.n_circRNAs)
        logging.info('n_diseases:        %d' % self.n_diseases)
        logging.info('n_all_entities:    %d' % self.n_all_entities)
        logging.info('n_relations:       %d' % self.n_relations)
        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))
        logging.info('n_ap_train:        %d' % self.n_ap_train)
        logging.info('n_ap_test:         %d' % self.n_ap_test)
        logging.info('n_kg_train:        %d' % self.n_kg_train)

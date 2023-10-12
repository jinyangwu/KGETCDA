#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Davin Wu

import os
import sys
import csv
import numpy as np
from collections import OrderedDict


def main(fold, raw_path, save_path):
    """This is the main code to preprocess CircDisease2 and there are total 5 relations
    Args:
        fold (int): fold num
        raw_path (str): the path of raw data
        save_path (str): the path to store processed data
    """
    
    # Load raw data and get mappings of all entities and triples
    triples, relations, ents2ids, rels2ids, circ_disease_pair = generate_basic_information(raw_path, save_path)
    
    # Generate train, valid and test set for every fold
    generate_fold_set(fold, save_path, triples, relations, ents2ids, rels2ids, circ_disease_pair)


def generate_basic_information(raw_path, save_path):
    """Load raw data and get mappings of all entities and triples
    Args:
        raw_path (str): the path of raw data
        save_path (str): the path to store processed data
    """
    circ_list, miR_list, lnc_list, disease_list = [], [], [], []
    circ_disease, circ_miR, miR_disease, miR_lnc, lnc_disease = [], [], [], [], []
    relations = ['circ-miRNA', 'miRNA-disease', 'miRNA-lncRNA', 'lncRNA-disease', 'circ-disease']

    
    # Load circ-miRNA pairs: relation 'circ-miRNA'
    with open(raw_path + 'circrna-mirna.txt', 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            circ_list.append(line[0])
            miR_list.append(line[1])
            circ_miR.append([line[0], relations[0], line[1]])

    # Load miRNA-disease pairs: relation 'miRNA-disease'
    with open(raw_path + 'mirna-disease.txt', 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            miR_list.append(line[0])
            disease_list.append(line[1].lower())
            miR_disease.append([line[0],  relations[1], line[1].lower()])

    # Load miRNA-lncRNA pairs: relation 'miRNA-lncRNA'
    with open(raw_path + 'lncRNA_miRNA.csv', 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            if len(line[0]) <= 1:
                continue
            miR_list.append(line[1])
            lnc_list.append(line[0])
            miR_lnc.append([line[1],  relations[2], line[0]])

    # Load lncRNA-disease pairs: relation 'lncRNA-disease'
    with open(raw_path + 'lncrna-disease.txt', 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            if len(line[0]) <= 1:
                continue
            lnc_list.append(line[0])
            disease_list.append(line[1].lower())
            lnc_disease.append([line[0],  relations[3], line[1].lower()])
    
    # Load circ-disease pairs: relation 'circ-disease'
    with open(raw_path + 'circrna-disease.txt', 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            circ_list.append(line[0])
            disease_list.append(line[1].lower())
            circ_disease.append([line[0], relations[4], line[1].lower()])
    
    circ_list, miR_list, lnc_list, disease_list = sorted(set(circ_list)), sorted(set(miR_list)), \
        sorted(set(lnc_list)), sorted(set(disease_list))
    circ_disease = np.unique(np.array(circ_disease), axis=0)
    circ_miR = np.unique(np.array(circ_miR), axis=0) 
    miR_disease = np.unique(np.array(miR_disease), axis=0) 
    miR_lnc = np.unique(np.array(miR_lnc), axis=0)
    lnc_disease = np.unique(np.array(lnc_disease), axis=0)

    entities_all = circ_list + disease_list + miR_list + lnc_list
    entities = list(set(entities_all))
    entities.sort(key = entities_all.index)
    total_entities, total_relations = len(entities), len(relations)
    ents2ids = OrderedDict(zip(entities, range(total_entities)))
    rels2ids = OrderedDict(zip(relations, range(total_relations)))
    triples = np.concatenate([circ_miR, miR_disease, miR_lnc, lnc_disease, circ_disease], axis=0)
    triples_part = np.concatenate([circ_miR, miR_disease, miR_lnc, lnc_disease], axis=0)

    # Write all entities, relations and triples with their ids into txt files 
    # and its format is {entity_name   id} for every row
    with open(save_path + 'name2id_circ.txt', 'a') as f:
        f.seek(0)	  
        f.truncate()
        for name in circ_list:
            f.write(name + '\t' + str(ents2ids[name]) + '\n')
    with open(save_path + 'name2id_miR.txt', 'a') as f:
        f.seek(0)	  
        f.truncate()
        for name in miR_list:
            f.write(name + '\t' + str(ents2ids[name]) + '\n')
    with open(save_path + 'name2id_lnc.txt', 'a') as f:
        f.seek(0)	  
        f.truncate()
        for name in lnc_list:
            f.write(name + '\t' + str(ents2ids[name]) + '\n')
    with open(save_path + 'name2id_disease.txt', 'a') as f:
        f.seek(0)	  
        f.truncate()
        for name in disease_list:
            f.write(name + '\t' + str(ents2ids[name]) + '\n')
    with open(save_path + 'entity.txt', 'a') as f:
        f.seek(0)	  
        f.truncate()
        for name in entities:
            f.write(name + '\t' + str(ents2ids[name]) + '\n')
    with open(save_path + 'relation.txt', 'a') as f:
        f.seek(0)	  
        f.truncate()
        for name in relations:
            f.write(name + '\t' + str(rels2ids[name]) + '\n')
    with open(save_path + 'triples_all.txt', 'a') as f:  
        f.seek(0)	  
        f.truncate()
        for triple in triples:
            h, r, t = triple
            f.write(str(ents2ids[h]) + '\t' + str(rels2ids[r]) + '\t' + str(ents2ids[t]) + '\n')
    with open(save_path + 'triples_part.txt', 'a') as f:  
        f.seek(0)	  
        f.truncate()
        for triple in triples_part:
            h, r, t = triple
            f.write(str(ents2ids[h]) + '\t' + str(rels2ids[r]) + '\t' + str(ents2ids[t]) + '\n')
    if os.path.exists(save_path + 'circ_list.txt'):
        os.remove(save_path + 'circ_list.txt')
    if os.path.exists(save_path + 'disease_list.txt'):
        os.remove(save_path + 'disease_list.txt')
    for i in range(len(circ_list)):
        with open(save_path + 'circ_list.txt'.format(fold), 'a') as f:
            f.write(circ_list[i] + '\t' + str(i) + '\n')
    for j in range(len(disease_list)):
        with open(save_path + 'disease_list.txt'.format(fold), 'a') as f:
            f.write(disease_list[j] + '\t' + str(j) + '\n')
    circ_disease_pair = []
    circ_disease_matrix = np.zeros([len(circ_list), len(disease_list)], dtype=np.int32)
    for [h, r, t] in circ_disease:
        circ_id, disease_id = circ_list.index(h), disease_list.index(t)
        circ_disease_pair.append([circ_id, disease_id])
        circ_disease_matrix[circ_id, disease_id] = 1
    if os.path.exists(save_path + 'circ_disease_association.txt'):
        os.remove(save_path + 'circ_disease_association.txt')
    np.savetxt(save_path + 'circ_disease_association.txt', circ_disease_matrix, fmt='%d', delimiter=',')


    print('--------------------------------------------------')
    print(f'After clearning up the CircDisease2:\n#circRNAs: {len(circ_list)} | #miRNA: {len(miR_list)} | ' + \
          f'#lncRNA: {len(lnc_list)} | #disease: {len(disease_list)}')
    print('--------------------------------------------------')
    print(f'#entities: {total_entities} | #relations: {total_relations} | #triples: {len(triples)} |\n' + \
          f'#circ-disease: {len(circ_disease)} | #circ_miR: {len(circ_miR)} | ' + \
          f'#miR_disease: {len(miR_disease)} | #miR_lnc: {len(miR_lnc)} | #lnc_disease: {len(lnc_disease)}')
    print('--------------------------------------------------')
    print(f'circ_disease_pair: {len(circ_disease_pair)}')
    return triples, relations, ents2ids, rels2ids, np.array(circ_disease_pair)


def generate_fold_set(fold, save_path, triples, relations, ents2ids, rels2ids, circ_disease_pair):
    """Generate train, valid and test set for every fold
    Args:
        fold (int): _description_
        save_path (str): the fold to store train.txt, valid.txt and test.txt
    """
    for i in range(1, fold+1):

        # Write train, valid and test triples into file, i.e. {head_name, relation, tail_name} for every row
        np.random.shuffle(circ_disease_pair)
        np.random.shuffle(triples)
        n_triples = len(triples)
        n_pairs = len(circ_disease_pair)
        
        p_train = 0.8
        save_fold_path = save_path + f'fold{i}/'  
 
        pairs_train = circ_disease_pair[:int(n_pairs*p_train), ]
        pairs_test = circ_disease_pair[int(n_pairs*p_train):, ]
        
        # Get train, valid and test txt 
        if not os.path.exists(save_fold_path):
            os.makedirs(save_fold_path)
        with open(save_fold_path + 'pairs_train.txt', 'a') as f:  
            f.seek(0)	  
            f.truncate()
            for pair in pairs_train:
                circ, disease = pair
                f.write(str(circ) + '\t' + str(disease) + '\n')
        with open(save_fold_path + 'pairs_test.txt', 'a') as f:  
            f.seek(0)	  
            f.truncate()
            for pair in pairs_test:
                circ, disease = pair
                f.write(str(circ) + '\t' + str(disease) + '\n')
        
        print('--------------------------------------------------')
        print(f'For fold {i}:')
        print(f'Pairs | #train: {len(pairs_train)} | #test: {len(pairs_test)}')
        print('--------------------------------------------------')


if __name__ == '__main__':
    os.chdir(sys.path[0])
    fold = 5    # 5-fold validation
    raw_path, save_path = '../Dataset1/', '../'   # Here change to your path
    main(fold, raw_path, save_path)

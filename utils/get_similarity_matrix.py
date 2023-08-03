import os
import sys

import copy
import h5py
import math
import numpy as np
import pandas as pd


def load_entity(path, file_name):
    with open(path + file_name, 'r') as f:
        entity_list = []
        lines = f.read().splitlines()
        for line in lines:
            entity_list.append(line.split('\t')[0])  # entity_list.append(row[0].lower())
    return entity_list


class GetGaussianSimilarity(object):
    """Get gaussian similarity matrix according to association matrix of circRNAs and diseases"""
    def __init__(self, circ_dis_matrix):
        self.mean_width = self.get_mean_width(circ_dis_matrix)
        self.circ_sim_matrix = self.get_sim_matrix(circ_dis_matrix)
        self.dis_sim_matrix = self.get_sim_matrix(circ_dis_matrix.transpose())
        
    def get_sim_matrix(self, target):
        """Get gaussian similarity matrix of circRNAs and diseases and its shape is (n_circ, n_dis)"""
        sim_matrix = np.zeros((target.shape[0], target.shape[0])).astype('float32')
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                sim_matrix[i][j] = self.compute_sim(i, j, target)

        return sim_matrix

    def compute_sim(self, i, j, association_mat):
        """Compute Gaussian Interaction Similarity(GIP) of RNAs or diseases"""
        theta = 1 / self.mean_width
        rna, disease = association_mat[i, :], association_mat[j, :]
        similarity = np.exp(-theta * np.linalg.norm(rna - disease) ** 2)

        return similarity
    
    def get_mean_width(self, association_mat):
        """Calculating the mean width in GIP"""
        sum_width = []
        for i in range(association_mat.shape[0]):
            sum_width.append(np.linalg.norm(association_mat[i, :]) ** 2)
        mean_width = np.mean(np.array(sum_width))

        return mean_width


class GetSemanticSimilarity(object):
    """Get disease semantic similarity of MeSH"""
    def __init__(self, id, disease, unique_disease, epsilon):
        self.id, self.disease, self.unique_disease = id, disease, unique_disease
        self.semantic_sim1 = self.get_sim_model1(epsilon)
        self.semantic_sim2 = self.get_sim_model2()
    
    def get_sim_model1(self, epsilon):
        # Get id-disease pairs and unique diseases name in MeSH
        ids, diseases, unique_diseases = copy.deepcopy(self.id), copy.deepcopy(self.disease), copy.deepcopy(self.unique_disease)

        # Get contributions of model1
        unique_disease_dict = self.get_contributions(ids, diseases, unique_diseases, epsilon, flag=1)

        # Compute similarity between every disease pair
        similarity = self.compute_similarity(unique_diseases, unique_disease_dict)
        
        return similarity
    
    def get_sim_model2(self):
        # Get id-disease pairs and unique diseases name in MeSH
        ids, diseases, unique_diseases = copy.deepcopy(self.id), copy.deepcopy(self.disease), copy.deepcopy(self.unique_disease)

        # Generate initial weighted contributions
        dv_weighted = {}
        all_ids = self.get_all_ids(ids)
        for key in all_ids:
            dv_weighted[key] = round(math.log((dv_weighted.get(key, 0) + 1)/len(diseases), 10)*(-1), 5)
        
        # Get final contributions of model2
        ids, diseases, unique_diseases = copy.deepcopy(self.id), copy.deepcopy(self.disease), copy.deepcopy(self.unique_disease)
        unique_disease_dict = self.get_contributions(ids, diseases, unique_diseases, epsilon, flag=2, dv_weighted=dv_weighted)
        
        # Compute similarity between every disease pair
        similarity = self.compute_similarity(unique_diseases, unique_disease_dict)
        
        return similarity
    
    def get_all_ids(self, ids):
        """Get all ids of diseases in MeSH"""
        all_ids = []
        for i in range(len(ids)):
            all_ids.append(ids[i])
            if len(ids[i]) > 3:
                ids[i] = ids[i][:-4]
                all_ids.append(ids[i])
                if len(ids[i]) > 3:
                    ids[i] = ids[i][:-4]
                    all_ids.append(ids[i])
                    if len(ids[i]) > 3:
                        ids[i] = ids[i][:-4]
                        all_ids.append(ids[i])
                        if len(ids[i]) > 3:
                            ids[i] = ids[i][:-4]
                            all_ids.append(ids[i])
                            if len(ids[i]) > 3:
                                ids[i] = ids[i][:-4]
                                all_ids.append(ids[i])
                                if len(ids[i]) > 3:
                                    ids[i] = ids[i][:-4]
                                    all_ids.append(ids[i])
                                    if len(ids[i]) > 3:
                                        ids[i] = ids[i][:-4]
                                        all_ids.append(ids[i])
                                        if len(ids[i]) > 3:
                                            ids[i] = ids[i][:-4]
                                            all_ids.append(ids[i])
        return all_ids
    
    def get_contributions(self, ids, diseases, unique_diseases, epsilon, flag, dv_weighted=None):
        
        def multiply_operation(epsilon, num, k):
            multiply_sum = 1
            for i in range(num):
                multiply_sum *= epsilon
            return round(multiply_sum, k)
        
        # Get initial contributions
        disease_dict = {i:{} for i in range(len(diseases))}
        for i in range(len(diseases)):
            if len(ids[i]) > 3:
                disease_dict[i][ids[i]] = 1 if flag==1 else dv_weighted[ids[i]] 
                ids[i] = ids[i][:-4]
                if len(ids[i]) > 3:
                    disease_dict[i][ids[i]] = multiply_operation(epsilon, 1, 5) if flag==1 else dv_weighted[ids[i]]  
                    ids[i] = ids[i][:-4]
                    if len(ids[i]) > 3:
                        disease_dict[i][ids[i]] = multiply_operation(epsilon, 2, 5) if flag==1 else dv_weighted[ids[i]]      
                        ids[i] = ids[i][:-4]
                        if len(ids[i]) > 3:
                            disease_dict[i][ids[i]] = multiply_operation(epsilon, 3, 5) if flag==1 else dv_weighted[ids[i]] 
                            ids[i] = ids[i][:-4]
                            if len(ids[i]) > 3:
                                disease_dict[i][ids[i]] = multiply_operation(epsilon, 4, 5) if flag==1 else dv_weighted[ids[i]]  
                                ids[i] = ids[i][:-4]
                                if len(ids[i]) > 3:
                                    disease_dict[i][ids[i]] = multiply_operation(epsilon, 5, 5) if flag==1 else dv_weighted[ids[i]] 
                                    ids[i] = ids[i][:-4]
                                    if len(ids[i]) > 3:
                                        disease_dict[i][ids[i]] = multiply_operation(epsilon, 6, 5) if flag==1 else dv_weighted[ids[i]] 
                                        ids[i] = ids[i][:-4]
                                        if len(ids[i]) > 3:
                                            disease_dict[i][ids[i]] = multiply_operation(epsilon, 7, 5) if flag==1 else dv_weighted[ids[i]] 
                                            ids[i] = ids[i][:-4]
                                        else:
                                            disease_dict[i][ids[i][:3]] = multiply_operation(epsilon, 7, 5) if flag==1 else dv_weighted[ids[i][:3]]
                                    else:
                                        disease_dict[i][ids[i][:3]] = multiply_operation(epsilon, 6, 5) if flag==1 else dv_weighted[ids[i][:3]]
                                else:
                                    disease_dict[i][ids[i][:3]] = multiply_operation(epsilon, 5, 5) if flag==1 else dv_weighted[ids[i][:3]]
                            else:
                                disease_dict[i][ids[i][:3]] = multiply_operation(epsilon, 4, 5) if flag==1 else dv_weighted[ids[i][:3]]
                        else:
                            disease_dict[i][ids[i][:3]] = multiply_operation(epsilon, 3, 5) if flag==1 else dv_weighted[ids[i][:3]]
                    else:
                        disease_dict[i][ids[i][:3]] = multiply_operation(epsilon, 2, 5) if flag==1 else dv_weighted[ids[i][:3]]
                else:
                    disease_dict[i][ids[i][:3]] = multiply_operation(epsilon, 1, 5) if flag==1 else dv_weighted[ids[i][:3]]
            else:
                disease_dict[i][ids[i][:3]] = 1 if flag==1 else dv_weighted[ids[i][:3]]

        # Get final contribution after removing duplicate parts
        unique_disease_dict = {}
        for i in range(len(unique_diseases)):
            unique_disease_dict[i] = {}
            for j in range(len(diseases)):
                if unique_diseases[i] == diseases[j]:
                    unique_disease_dict[i].update(disease_dict[j])

        return unique_disease_dict
    
    def compute_similarity(self, unique_diseases, unique_disease_dict):
        """Calculate similarity every disease-disease pair
        Args:
            unique_diseases (list): unique disease names in MeSH
            unique_disease_dict (dict): store the contribution of all disease-disease pairs
        """
        similarity = np.zeros([len(unique_diseases), len(unique_diseases)])
        for m in range(len(unique_diseases)):
            for n in range(len(unique_diseases)):
                denominator = sum(unique_disease_dict[m].values()) + sum(unique_disease_dict[n].values())
                numerator = 0
                for k, v in unique_disease_dict[m].items():
                    if k in unique_disease_dict[n].keys():
                        numerator += (v + unique_disease_dict[n].get(k))
                similarity[m, n] = round(numerator/denominator, 5)
        
        return similarity


class GetFunctionalSimilarity(object):
    """Get functional similarity matrix according to association matrix of circRNAs and disease semantic similarity"""
    def __init__(self, circ_dis_matrix, n_circ, dis_semantic_sim):
        self.circ_sim_matrix = self.get_sim_matrix(n_circ, circ_dis_matrix, dis_semantic_sim)
        
    def get_sim_matrix(self, n_circ, circ_dis_matrix, dis_semantic_sim):
        """Get functional similarity matrix of circRNAs and its shape is (n_circ, n_circ)"""
        sim_matrix = np.zeros((n_circ, n_circ), dtype=np.float32)
        for i in range(n_circ):
            idx = np.nonzero(circ_dis_matrix[i, :])[0]
            if idx.size == 0:
                continue
            
            for j in range(i):
                idy = np.nonzero(circ_dis_matrix[j, :])[0]
                if idy.size == 0:
                    continue
                sum1, sum2 = 0, 0
                for k1 in range(len(idx)):
                    sum1 = sum1 + max(dis_semantic_sim[idx[k1], idy])
                for k2 in range(len(idy)):
                    sum2 = sum2 + max(dis_semantic_sim[idx, idy[k2]])
                sim_matrix[i, j] = (sum1+sum2) / (len(idx)+len(idy))
                sim_matrix[j, i] = sim_matrix[i, j]
            
            for k in range(n_circ):
                sim_matrix[k, k] = 1 
            
        return sim_matrix
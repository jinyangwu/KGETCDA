#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jinyang Wu

import os
import sys
import gc

import torch
import torch.nn as nn

import h5py
import numpy as np

from utils.helper_init import set_seed
from utils.helper_metrics import calc_metrics_a_fold, sort_by_scores, get_top_k
from parsers.parser_KGETCDA import parse_args


class MLP(nn.Module):
    """Define the MLP part to make predictions"""
    def __init__(self, input_dim, layers, dropout):
        super(MLP, self).__init__()
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        linears_list = [nn.Linear(input_dim, layers[0]), self.act_fn, self.dropout]
        for i in range(len(layers[:-1])):
            linears_list.extend([nn.Linear(layers[i], layers[i+1]), self.act_fn, self.dropout])
        self.linears = nn.Sequential(*linears_list)
        self.out_layer = nn.Linear(layers[-1], 1)
        self.creterion = nn.BCELoss()  

    def cal_loss(self, y_pred, y_true, device):  # label should be float dtype for nn.BCELoss()
        return self.creterion(y_pred.reshape(-1), torch.tensor(y_true, dtype=torch.float32, device=device).reshape(-1))

    def forward(self, x):
        hidden_out = self.linears(x)
        prediction = torch.sigmoid(self.out_layer(hidden_out))

        return prediction


def load_model(model, model_path):
    '''Load model from the model_path'''
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def init_parameters(m):
    """Initialize the parameters of linear layers"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):  
        nn.init.xavier_normal_(m.weight) 


def calc_metrics_5_folds(all_tpr, all_fpr, all_recall, all_precision, all_accuracy, all_f1, top_dict, embed_dim, 
                kg_type, decoder, aggregation_type, num_heads, n_layers, tower, sim_type, result_path):
    """Calculate metrics of 5 folds"""
    mean_cross_tpr = np.mean(np.array(all_tpr), axis=0)
    mean_cross_fpr = np.mean(np.array(all_fpr), axis=0)
    mean_cross_recall = np.mean(np.array(all_recall), axis=0)
    mean_cross_precision = np.mean(np.array(all_precision), axis=0)

    mean_accuracy = np.mean(np.mean(np.array(all_accuracy), axis=1), axis=0)
    mean_recall = np.mean(np.mean(np.array(all_recall), axis=1), axis=0)
    mean_precision = np.mean(np.mean(np.array(all_precision), axis=1), axis=0)
    mean_F1 = np.mean(np.mean(np.array(all_f1), axis=1), axis=0)
    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr) 
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)
    print("In the paper, accuracy: %.4f | recall: %.4f | precision: %.4f | F1: %.4f | AUC: %.4f | AUPR: %.4f" %
          (mean_accuracy, mean_recall, mean_precision, mean_F1, roc_auc, AUPR))
    for i in list(top_dict.keys()):
        print("top" + str(i) + ": " + str(sum(top_dict[i]) / len(top_dict[i])))
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if kg_type == 'KGET':
        file_name = f'/KGETCDA_pre_rec_tpr_fpr_base.h5'
        if embed_dim != 2048:
            file_name = f'/KGETCDA_pre_rec_tpr_fpr_{embed_dim}.h5'
        if sim_type == 'gip' or sim_type == 'fun_circ':
            file_name = f'/KGETCDA_pre_rec_tpr_fpr_{sim_type}.h5'
        if decoder == 'threemult' or decoder == 'distance_based':
            file_name = f'/KGETCDA_pre_rec_tpr_fpr_{decoder}.h5'
        if aggregation_type == 'graphsage' or aggregation_type == 'bi-interaction':
            file_name = f'/KGETCDA_pre_rec_tpr_fpr_{aggregation_type}.h5'
        if tower != 2:
            file_name = f'/KGETCDA_pre_rec_tpr_fpr_tower{tower}.h5'
        if num_heads != 16:  
            file_name = f'/KGETCDA_pre_rec_tpr_fpr_heads{num_heads}.h5'
        if n_layers != 1:
            file_name = f'/KGETCDA_pre_rec_tpr_fpr_heads{num_heads}_layers_{n_layers}.h5'
    else:
        file_name = f'/{kg_type}_pre_rec_tpr_fpr_4layers_{embed_dim}.h5'
    # print(result_path+file_name)
    if os.path.exists(result_path + file_name):
        os.remove(result_path + file_name)
    with h5py.File(result_path + file_name, 'w') as hf:
        hf['tpr'] = mean_cross_tpr
        hf['fpr'] = mean_cross_fpr
        hf['recall'] = mean_cross_recall
        hf['precision'] = mean_cross_precision


def load_entity(path, file_name):
    with open(path + file_name, 'r') as f:
        entity_list = []
        lines = f.read().splitlines()
        for line in lines:
            entity_list.append(line.split('\t')[0])  
    return entity_list


def top_k_sim_circ(circ_sim_matrix, circ_index, k=10):
    """Define the top-k most similar circRNAs with circRNA-circ_index
    Args:
        circ_sim_matrix (array): store similarity of circRNAs and its shape is (n_circ, n_circ)
        circ_index (int): index of the target circRNA
        k (int, optional): threshold to be set of top-k. Defaults to 10.
    """
    top_circ_list = np.argsort(-circ_sim_matrix[circ_index,:])
    top_k_circ_list = top_circ_list[:k]
    top_k_circ_list = top_k_circ_list.tolist()
    if circ_index in top_k_circ_list:
        top_k_circ_list = top_circ_list[:k+1]
        top_k_circ_list = top_k_circ_list.tolist()
        top_k_circ_list.remove(circ_index)

    return top_k_circ_list


def find_associated_diseases(top_k_circ_list, rel_matrix, associate_dis_set):
    """Find related diseases of top-k most similar circRNAs
    Args:
        top_k_circ_list (list): store top-k most similar circRNAs
        rel_matrix (_type_): relational matrix which only store information of training data
        associate_dis_set (set): store related diseases
    """
    for circ_id in top_k_circ_list:
        for j in range(rel_matrix.shape[1]):
            if rel_matrix[circ_id, j] == 1:
                associate_dis_set.add(j)

    return associate_dis_set


def get_train_data(train_index, rel_mat, circ_sim_mat, circ_embed, dis_embed, n_dis, k=10, ratio=8):
    """Construct training data and ratio of pos:neg defaults to 1:8"""
    features_train = []
    labels_train = []    
    for [circ_index, dis_index] in train_index:
        circ_array = circ_embed[circ_index,:]
        dis_array = dis_embed[dis_index,:]
        fusion_feature = np.concatenate((circ_array, dis_array), axis=0)
        features_train.append(fusion_feature.tolist())
        labels_train.append(1)

        top_k_circ_list = top_k_sim_circ(circ_sim_mat, circ_index, k)
        associate_dis_set = set()
        associate_dis_set = find_associated_diseases(top_k_circ_list, rel_mat, associate_dis_set)
        for num in range(ratio):
            j = np.random.randint(n_dis)
            while ([circ_index, j] in train_index) or (j in associate_dis_set):
                j = np.random.randint(n_dis)

            dis_array = dis_embed[j,:]
            fusion_feature = np.concatenate((circ_array, dis_array), axis=0)
            features_train.append(fusion_feature.tolist())
            labels_train.append(0)

    return features_train, labels_train


def get_test_data(rel_matrix, circ_embed, dis_embed):
    """Construct testing data using the relational matrix"""
    features_test = []
    for row in range(rel_matrix.shape[0]):
        for col in range(rel_matrix.shape[1]):
            circ_array = circ_embed[row,:]
            dis_array = dis_embed[col,:]
            fusion_feature = np.concatenate((circ_array, dis_array), axis=0)
            features_test.append(fusion_feature.tolist())

    return features_test


def predict(model, device, rel_mat, roc_circ_dis_mat, circ_embed, dis_embed):
    """Using method of the paper to do predictions"""
    feature_test = get_test_data(rel_mat, circ_embed, dis_embed)  
    with torch.no_grad():
        predictions1 = np.array(model(torch.tensor(np.array(feature_test)[0:len(feature_test)//2], dtype=torch.float32, device=device)).cpu().detach())
        predictions2 = np.array(model(torch.tensor(np.array(feature_test)[len(feature_test)//2:], dtype=torch.float32, device=device)).cpu().detach())
        predictions = np.concatenate([predictions1, predictions2], axis=0)
    prediction_matrix = np.zeros((rel_mat.shape[0], rel_mat.shape[1]))
    predictions_index = 0
    for row in range(prediction_matrix.shape[0]):  # (n_circ, n_disease)
        for col in range(prediction_matrix.shape[1]):
            prediction_matrix[row, col] = predictions[predictions_index]
            predictions_index += 1
    print(f'prediction_matrix.shape: {prediction_matrix.shape}')
    score_matrix = prediction_matrix.copy()
    minvalue = np.min(score_matrix)
    score_matrix[np.where(roc_circ_dis_mat == 2)] = minvalue - 100
    score_matrix, roc_circ_dis_mat, prediction_matrix = score_matrix.T, roc_circ_dis_mat.T, prediction_matrix.T
    sorted_circ_dis_mat = sort_by_scores(score_matrix, roc_circ_dis_mat)

    tpr_list, fpr_list, accuracy_list, recall_list, precision_list, f1_list = calc_metrics_a_fold(sorted_circ_dis_mat)
    return tpr_list, fpr_list, recall_list, precision_list, accuracy_list, f1_list, sorted_circ_dis_mat


def test(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    result_path = f'./results/{args.dataset}/Metrics'
    mlp_input = 2 * (args.entity_dim + sum(eval(args.conv_dim_list)))
    layers = [mlp_input//args.tower, mlp_input//(args.tower*args.tower)]
    print(args.save_dir+f"/MLP_dropout_{args.mlp_dropout}/MLP_lr_{args.mlp_lr}"+\
        f"/MLP_wd_{args.mlp_wd}"+f"/MLP_epochs_{args.mlp_epochs}"+f"/sim_type_{args.sim_type}")

    # Set random seed
    set_seed(args.seed, random_lib=True, numpy_lib=True, torch_lib=True)

    # Load data
    dataset_folder = args.data_dir + args.dataset
    circ_dis_list = []
    with open(dataset_folder + '/circ_dis_association.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            circ_dis_list.append(line.split(','))
    circ_dis_mat = np.array(circ_dis_list, dtype=np.int32)

    # Start 5-fold validation
    all_tpr, all_fpr, all_recall, all_precision, all_accuracy, all_f1 = [], [], [], [], [], []
    top_dict = {10: [], 20: [], 30: [], 40: []}
    for fold in range(1, 6):
        
        # Define MLP model
        model = MLP(mlp_input, layers, args.mlp_dropout)
        model.apply(init_parameters)
        model.to(device)
        optimizer = torch.optim.Adam([{'params': model.linears[0:2].parameters(), 'weight_decay': args.mlp_wd}, \
                                      {'params': model.linears[2:].parameters()}], lr=args.mlp_lr)

        print('=========================================')
        print(f'Now testing in fold{fold}......')
        train_index, test_index = [], []
        with open(dataset_folder + f'/fold{fold}' + '/pairs_train.txt', "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                train_index.append(line.split('\t'))
        with open(dataset_folder + f'/fold{fold}' + '/pairs_test.txt', "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                test_index.append(line.split('\t'))
        train_index = np.array(train_index, dtype=np.int32).tolist()
        test_index = np.array(test_index, dtype=np.int32).tolist()
        with h5py.File(args.save_dir + f'fold{fold}/embedding/' + 'entity_relation_embeddings.h5', 'r') as hf:
            circ_embed = hf['circ_embedding'][:]
            dis_embed = hf['dis_embedding'][:]

        # Get relational matrix and process it for metrics calculation later
        new_circ_dis_mat = circ_dis_mat.copy()
        for index in test_index:
            new_circ_dis_mat[index[0], index[1]] = 0
        roc_circ_dis_mat = new_circ_dis_mat + circ_dis_mat  
        rel_mat = new_circ_dis_mat                          
        n_circ, n_dis = rel_mat.shape
        
        # Load similarity matrices
        with h5py.File(dataset_folder + f'/fold{fold}' + f"/{args.sim_type}_sim_matrix.h5", 'r') as hf:
            circ_sim_mat = hf['circ_sim_matrix'][:]
            dis_sim_matrix = hf['dis_sim_matrix'][:]

        feature_train, label_train = get_train_data(train_index, rel_mat, circ_sim_mat, circ_embed, 
                                                         dis_embed, n_dis, args.sample_k, args.sample_ratio)
        for _ in range(args.mlp_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            y_pred = model(torch.tensor(feature_train, device=device))
            loss = model.cal_loss(y_pred, label_train, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()
        tpr_list, fpr_list, recall_list, precision_list, accuracy_list, f1_list, sorted_circ_dis_mat\
            = predict(model, device, rel_mat, roc_circ_dis_mat, circ_embed, dis_embed)

        top_list = [10, 20, 30, 40]
        top_dict = get_top_k(top_dict, top_list, sorted_circ_dis_mat)

        all_tpr.append(tpr_list)
        all_fpr.append(fpr_list)
        all_recall.append(recall_list)
        all_precision.append(precision_list)
        all_accuracy.append(accuracy_list)
        all_f1.append(f1_list)

    # Calculate metrics and print information
    calc_metrics_5_folds(all_tpr, all_fpr, all_recall, all_precision, all_accuracy, all_f1, top_dict, args.entity_dim,\
        args.kg_type, args.decoder, args.aggregation_type, args.num_heads, args.n_layers, args.tower, args.sim_type, result_path)


if __name__ == '__main__':
    os.chdir(sys.path[0])
    args = parse_args()
    test(args)
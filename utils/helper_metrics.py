#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jinyang Wu


import os
import h5py
import numpy as np


def calc_metrics_a_fold(binary_hit):
    """Calculate all metrics in a fold and then return lists of accuracy, precision, recall, f1-score, tpr, fpr
    Args:
        binary_hit(array): element is binary (0 / 1), 2-dim
    """
    tpr_list, fpr_list = [], []
    accuracy_list, recall_list, precision_list, f1_list = [], [], [], []
    for i in range(binary_hit.shape[1]):
        p_matrix, n_matrix = binary_hit[:, 0:i+1], binary_hit[:, i+1:binary_hit.shape[1]+1]
        tp = np.sum(p_matrix == 1)
        fp = np.sum(p_matrix == 0)
        tn = np.sum(n_matrix == 0)
        fn = np.sum(n_matrix == 1)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tn+tp) / (tn+tp+fn+fp)
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1 = (2*tp) / (2*tp + fp + fn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)

    return tpr_list, fpr_list, accuracy_list, recall_list, precision_list, f1_list


def calc_metrics_5_folds(all_tpr, all_fpr, all_recall, all_precision, all_accuracy, all_f1, top_dict, result_path):
    """Calculate metrics of 5 folds, and get accuracy, recall, precision, f1-score, auc, aupr"""
    mean_cross_tpr = np.mean(np.array(all_tpr), axis=0)
    mean_cross_fpr = np.mean(np.array(all_fpr), axis=0)
    mean_cross_recall = np.mean(np.array(all_recall), axis=0)
    mean_cross_precision = np.mean(np.array(all_precision), axis=0)

    mean_accuracy = np.mean(np.mean(np.array(all_accuracy), axis=1), axis=0)
    mean_recall = np.mean(np.mean(np.array(all_recall), axis=1), axis=0)
    mean_precision = np.mean(np.mean(np.array(all_precision), axis=1), axis=0)
    mean_f1 = np.mean(np.mean(np.array(all_f1), axis=1), axis=0)
    auc = np.trapz(mean_cross_tpr, mean_cross_fpr)  
    aupr = np.trapz(mean_cross_precision, mean_cross_recall)
    print("In the paper, accuracy: %.4f | recall: %.4f | precision: %.4f | F1: %.4f | AUC: %.4f | AUPR: %.4f" %
          (mean_accuracy, mean_recall, mean_precision, mean_f1, auc, aupr))
    for i in list(top_dict.keys()):
        print("top" + str(i) + ": " + str(sum(top_dict[i]) / len(top_dict[i])))
    
    if os.path.exists(result_path):
        os.remove(result_path)
    with h5py.File(result_path, 'w') as hf:
        hf['tpr'] = mean_cross_tpr
        hf['fpr'] = mean_cross_fpr
        hf['recall'] = mean_cross_recall
        hf['precision'] = mean_cross_precision


def get_top_k(top_dict, top_list, sorted_circ_dis_mat):
    """Get top-k of the model predictions"""
    for num in top_list:
        p_matrix = sorted_circ_dis_mat[:, 0:num]
        top_count = np.sum(p_matrix == 1)
        top_dict[num].append(top_count)
    return top_dict


def sort_by_scores(score_matrix, interact_matrix):
    """Sort the score matrix and interaction matrix in descending order"""
    sort_index = np.argsort(-score_matrix, axis=1)
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[0]):
        score_sorted[i, :] = score_matrix[i, :][sort_index[i, :]]  
        y_sorted[i, :] = interact_matrix[i, :][sort_index[i, :]]
    return y_sorted
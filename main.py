#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jinyang Wu

import os
import gc
import sys
from time import time

import h5py
import numpy as np

import torch
import torch.optim as optim

from test import test
from models.KGETCDA import KGETCDA
from utils.helper_init import set_seed
from utils.helper_log import set_up_logger
from parsers.parser_KGETCDA import parse_args
from data_loader.loader_KGETCDA import DataLoaderKGETCDA


def train(args):
    """This is the main program of training KGETCDA"""
    
    # Set random seed
    set_seed(args.seed, random_lib=True, numpy_lib=True, torch_lib=True)

    # Set path
    args.log_path = args.save_dir + f'fold{args.fold}/log/'
    args.model_path = args.save_dir + f'fold{args.fold}/saved_model/'
    args.embedding_path = args.save_dir + f'fold{args.fold}/embedding/'
    args.result_path = args.save_dir + f'fold{args.fold}/results/'
    
    # Set up logger
    logger = set_up_logger(args.log_path, file_handler=True, stream_handler=True)
    logger.info(args)

    # GPU / CPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu>=0 else 'cpu')

    # Load data
    dataloader = DataLoaderKGETCDA(args, logger)

    # Construct model & optimizer
    model = KGETCDA(args, dataloader.n_circRNAs, dataloader.n_other_entities, dataloader.n_relations, dataloader.A_in)
    model.to(device)
    logger.info(model)
    ap_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the base model, consisting of 3 steps
    for epoch in range(1, args.n_epoch + 1):
        gc.collect()
        torch.cuda.empty_cache()
        time0 = time()
        model.train()

        # First step: train ap (attentive propagation layer)
        ap_total_loss = 0
        n_ap_batch = dataloader.n_ap_train // dataloader.ap_batch_size + 1
        for iter in range(1, n_ap_batch + 1):
            time1 = time()

            ap_batch_circ, ap_batch_pos_dis, ap_batch_neg_dis = dataloader.generate_ap_batch(dataloader.train_circ_dict, dataloader.ap_batch_size)
            ap_batch_circ = ap_batch_circ.to(device)    
            ap_batch_pos_dis = ap_batch_pos_dis.to(device)
            ap_batch_neg_dis = ap_batch_neg_dis.to(device)
            ap_batch_loss = model(ap_batch_circ, ap_batch_pos_dis, ap_batch_neg_dis, mode='train_ap')

            if np.isnan(ap_batch_loss.cpu().detach().numpy()):
                logger.info('ERROR (AP Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.
                            format(epoch, iter, n_ap_batch))
                sys.exit()

            ap_batch_loss.backward()
            ap_optimizer.step()
            ap_optimizer.zero_grad()
            ap_total_loss += ap_batch_loss.item()

            if (iter % args.ap_print_every) == 0:
                logger.info(('AP Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f}' + \
                            ' | Iter Mean Loss {:.4f}').format(epoch, iter, n_ap_batch, time()-time1, ap_batch_loss.item(), ap_total_loss/iter))


        # Second step: train kg (knowledge graph)
        kg_total_loss = 0
        n_kg_batch = dataloader.n_kg_train // dataloader.kg_batch_size + 1
        for iter in range(1, n_kg_batch + 1):
            time2 = time()

            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = dataloader.generate_kg_batch(
                dataloader.train_h_dict, dataloader.kg_batch_size, dataloader.n_all_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, args.kg_type, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logger.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.\
                            format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                logger.info(('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f}' + \
                            ' | Iter Mean Loss {:.4f}').format(epoch, iter, n_kg_batch, time() - time2, kg_batch_loss.item(), kg_total_loss/iter))


        # Third step: update the attention matrix
        time3 = time()
        h_list = dataloader.h_list.to(device)
        t_list = dataloader.t_list.to(device)
        r_list = dataloader.r_list.to(device)
        relations = list(dataloader.laplacian_r_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        logger.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time3))

        logger.info('Joint Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))


    # Save embeddings of the final model
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)
    if os.path.exists(args.embedding_path + 'entity_relation_embeddings.h5'):
        os.remove(args.embedding_path + 'entity_relation_embeddings.h5')
    all_embedding = model.calc_ap_embeddings().cpu().detach()
    with h5py.File(args.embedding_path + 'entity_relation_embeddings.h5', 'w') as hf:
        hf['circ_embedding'] = all_embedding[:dataloader.n_circRNAs, :]
        hf['dis_embedding'] = all_embedding[dataloader.n_circRNAs:(dataloader.n_diseases+dataloader.n_circRNAs), :]
        hf['rel_embedding'] = model.relation_embed.weight.cpu().detach()


if __name__ == '__main__':
    os.chdir(sys.path[0])
    args = parse_args()
    for args.fold in range(1, 6):
        print('==='*30)
        print('==='*30)
        print(f'Now, training in fold{args.fold}......')
        train(args)
    print('==='*30)
    print('==='*30)
    print('Now, testing......')
    test(args)
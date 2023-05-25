#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jinyang Wu

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KGETCDA")

    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument('--dataset', nargs='?', default='Dataset1',
                        help='Dataset to use: Dataset1, Dataset2, Dataset3')
    parser.add_argument('--fold', nargs='?', type=int, default=1,
                        help='fold to select: 1, 2, 3, 4, 5')
    parser.add_argument('--data_dir', nargs='?', default='./datasets/',
                        help='Input data path.')
    
    parser.add_argument('--kg_type', type=str, default='KGET',
                        help='Specify the type of the kg embeddings from {KGET, TransH, TransD, ConvKB, ConvE, HolE, RotatE}.')
    parser.add_argument('--aggregation_type', type=str, default='gcn',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')

    parser.add_argument('--ap_batch_size', type=int, default=80,
                        help='Attentive (embedding) Propagation layer batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=64,  
                        help='Knowledge Graph batch size.')

    parser.add_argument('--entity_dim', type=int, default=2048,   
                        help='entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=2048, 
                        help='Relation Embedding size.')
    parser.add_argument('--d_k', type=int, default=32,
                        help='Dimension of key')
    parser.add_argument('--d_v', type=int, default=50,
                        help='Dimension of value')
    parser.add_argument('--d_inner', type=int, default=512,
                        help='Dimension of inner(hidden) layer (FFN)')

    parser.add_argument('--kernels', type=int, default=64,
                        help='Number of kernels in ConvKB')
    parser.add_argument('--conv_dim_list', nargs='?', default='[512, 256, 128, 64]',   
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--decoder', type=str, default='twomult',  
                        help='decoder: twomult, threemult, distance_based in KGET.')
    parser.add_argument("--n_layers", type=int, default=1,
                        help="Number of encoder layers in KGET")
    parser.add_argument("--num_heads", type=int, default=16,  
                        help="Number of attention heads in KGET")
    parser.add_argument('--dropout_convkb', type=int, default=0.2,
                        help='Dropout of ConvKB')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--kg_dropout', nargs='?', default='[0.2, 0.3, 0.2, 0.3]',
                        help='Dropout probability for layer in KGET and Sequentially dropout before encoder, after \
                        position feedforward, softmax of scaled dot product and multi-head attention. 0: no dropout.')
    
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--ap_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating AP l2 loss.')

    parser.add_argument('--lr', type=float, default=1e-4,   
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=100, 
                        help='Number of epoch.')

    parser.add_argument('--ap_print_every', type=int, default=5,
                        help='Iter interval of printing AP loss.')
    parser.add_argument('--kg_print_every', type=int, default=10,
                        help='Iter interval of printing KG loss.')
    
    parser.add_argument('--mlp_lr', type=float, default=0.0009,   
                        help='Learning rate of mlp.')
    parser.add_argument('--mlp_epochs', type=int, default=65,     
                        help='Number of epochs of mlp.')
    parser.add_argument('--mlp_dropout', type=float, default=0.04, 
                        help='Dropout probability of mlp.')
    parser.add_argument('--mlp_wd', type=float, default=1e-7,      
                        help='Weight decay of mlp.') 
    parser.add_argument('--sample_k', type=int, default=10, 
                        help='top-k most similar circRNAs.')
    parser.add_argument('--sample_ratio', type=int, default=8, 
                        help='Ratio of negative samples to positive samples.')
    
    parser.add_argument('--tower', type=int, default=2,   
                        help='Specify the denominator of the proportion of each layer reduced to in tower structure from {2, 4, 8, 16, 32}.')
    parser.add_argument('--sim_type', type=str, default='fused',   
                        help='Specify the similarity type in negative sampling part from {fused, gip, fun_circ}.')

    args = parser.parse_args()

    save_dir = './trained_model/KGETCDA/{}/embed-dim{}_relation-dim{}_{}_{}_{}_lr{}_heads{}_layers{}_decoder{}/'\
        .format(args.dataset, args.entity_dim, args.relation_dim, args.aggregation_type, args.kg_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.num_heads, args.n_layers, args.decoder)
    args.save_dir = save_dir

    return args



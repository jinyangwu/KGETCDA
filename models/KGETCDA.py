#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jinyang Wu


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGETCDA(nn.Module):
    """The main model of KGETCDA"""
    def __init__(self, args, n_circRNAs, n_other_entities, n_relations, A_in=None):

        super(KGETCDA, self).__init__()

        self.n_circRNAs = n_circRNAs              # circRNA
        self.n_other_entities = n_other_entities  # other 3 entities, i.e. disease, miRNA, lncRNA
        self.n_relations = n_relations            # 10 relations i.e. circ-disease, circ-miRNA, miRNA-disease, miRNA-lncRNA, lncRNA-disease and their reverse relations

        # For simplicity, we use the same dimention of relations and entities
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)  
        self.mess_dropout = eval(args.mess_dropout)    
        self.kg_dropout = eval(args.kg_dropout)        
        self.n_layers = len(eval(args.conv_dim_list))  

        self.kg_l2loss_lambda = args.kg_l2loss_lambda  
        self.ap_l2loss_lambda = args.ap_l2loss_lambda  
        
        self.batch_size_ap = args.ap_batch_size
        self.batch_size_kg = args.kg_batch_size

        self.entity_embed = nn.Embedding(self.n_other_entities + self.n_circRNAs, self.entity_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        nn.init.xavier_normal_(self.entity_embed.weight)  
        nn.init.xavier_normal_(self.relation_embed.weight)
        
        # Define parameters for KG part
        if args.kg_type == 'TransH':     # parameters for TransH
            self.relation_hyper = nn.Embedding(self.n_relations, self.relation_dim) 
            nn.init.xavier_normal_(self.relation_hyper.weight)
        
        elif args.kg_type == 'TransD':   # parameters for TransD
            self.ent_transfer = nn.Embedding(self.n_other_entities + self.n_circRNAs, self.entity_dim)
            self.rel_transfer = nn.Embedding(self.n_relations, self.relation_dim)
            nn.init.xavier_normal_(self.ent_transfer.weight)
            nn.init.xavier_normal_(self.rel_transfer.weight)

        elif args.kg_type == 'ConvKB':   # parameters for ConvKB
            self.kernels = args.kernels
            self.conv1_bn = nn.BatchNorm2d(1)
            self.conv_layer = nn.Conv2d(1, args.kernels, (1, 3))
            self.conv2_bn = nn.BatchNorm2d(args.kernels)
            self.dropout = nn.Dropout(args.dropout_convkb)
            self.non_linearity = nn.ReLU()  
            self.fc_layer = nn.Linear(self.entity_dim * args.kernels, 1, bias=False)

        elif args.kg_type == 'KGET':     # parameters for KGET and initialize KGET layer part
            kg_dropout_dict = {
                'dr_enc': self.kg_dropout[0],
                'dr_pff': self.kg_dropout[1],
                'dr_sdp': self.kg_dropout[2],
                'dr_mha': self.kg_dropout[3]
            }
            self.encoder = Encoder(args.entity_dim, args.n_layers, args.num_heads, args.d_k, args.d_v, \
                args.entity_dim, args.d_inner, args.decoder, kg_dropout_dict)
            if args.decoder == 'threemult':
                self.decode_method = 'threemult'
            elif args.decoder == 'distance_based':
                self.decode_method = 'distance_based'
            elif args.decoder == 'twomult':
                self.decode_method = 'twomult'
            self.ent_bn = nn.BatchNorm1d(self.entity_dim)  
            self.rel_bn = nn.BatchNorm1d(self.entity_dim)
        
        # Parameters for relational attention mechanism
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))  
        nn.init.xavier_normal_(self.trans_M)     

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        # Initialize the first attentive matrix to be the laplacian matrix
        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_circRNAs + self.n_other_entities, self.n_circRNAs + self.n_other_entities))
        if A_in is not None:
            self.A_in.data = A_in        # a large laplacian matrix of 10 relations, shape: (n_all_entities, n_all_entities)
        self.A_in.requires_grad = False  

    def calc_ap_embeddings(self):
        ego_embed = self.entity_embed.weight  
        all_embed = [ego_embed]
        
        # Update the node embeddings after the neighbor information is aggregated by 4 gcn layers and store all layes embeddings
        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)  
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)  # (n_all_entities, concat_dim = initial_emb+layer1_emb+layer2_emb+layer3emb+layer4_emb)
        return all_embed

    def calc_ap_loss(self, circ_ids, dis_pos_ids, dis_neg_ids):
        """
        circ_ids:      (ap_batch_size)
        dis_pos_ids:   (ap_batch_size)
        dis_neg_ids:   (ap_batch_size)
        """
        all_embed = self.calc_ap_embeddings()                   # (n_all_entities, concat_dim) 
        circ_embed = all_embed[circ_ids]                        # (ap_batch_size, concat_dim)
        dis_pos_embed = all_embed[dis_pos_ids]                  # (ap_batch_size, concat_dim)
        dis_neg_embed = all_embed[dis_neg_ids]     

        pos_score = torch.sum(circ_embed*dis_pos_embed, dim=1)  # (ap_batch_size)
        neg_score = torch.sum(circ_embed*dis_neg_embed, dim=1)   

        ap_loss = F.softplus(neg_score - pos_score)  
        ap_loss = torch.mean(ap_loss)
        l2_loss = self._L2_loss_mean(circ_embed) + self._L2_loss_mean(dis_pos_embed) + self._L2_loss_mean(dis_neg_embed)
        loss = ap_loss + self.ap_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t, kg_type):
        """Calculate kg loss on a mini-batch
        Args:
            h(tensor):      (kg_batch_size)
            r(tensor):      (kg_batch_size)
            pos_t(tensor):  (kg_batch_size)
            neg_t(tensor):  (kg_batch_size)
        """
        
        def _resize(ent_tensor, axis, size):
            ent_shape = ent_tensor.size()
            osize = ent_shape[axis]
            if osize == size:
                return ent_tensor
            if (osize > size):
                return torch.narrow(ent_tensor, axis, 0, size)
            paddings = []
            for i in range(len(ent_shape)):
                if i == axis:
                    paddings = [0, size - osize] + paddings
                else:
                    paddings = [0, 0] + paddings
            print (paddings)
            return F.pad(ent_tensor, paddings = paddings, mode = "constant", value = 0)

        def _convkb_score(h, r, t, entity_dim, kernels, conv1_bn, conv_layer, conv2_bn, non_linearity, dropout, fc_layer):
            h = h.unsqueeze(1)  # bs x 1 x dim
            r = r.unsqueeze(1)
            t = t.unsqueeze(1)

            conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
            conv_input = conv_input.transpose(1, 2)
            conv_input = conv_input.unsqueeze(1)
            conv_input = conv1_bn(conv_input)
            out_conv = conv_layer(conv_input)
            out_conv = conv2_bn(out_conv)
            out_conv = non_linearity(out_conv)
            out_conv = out_conv.view(-1, entity_dim * kernels)
            input_fc = dropout(out_conv)
            score = fc_layer(input_fc).view(-1)

            return -score

        r_embed = self.relation_embed(r)           # (kg_batch_size, relation_dim)
        w_r = self.trans_M[r]                      # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.entity_embed(h)             # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embed(pos_t)     # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embed(neg_t)     

        if kg_type == 'TransE':
            pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)  
            neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)  
        
        elif kg_type == 'TransH':
            rel_norm = F.normalize(self.relation_hyper(r), p = 2, dim=-1)          
            h_embed = h_embed - torch.sum(h_embed*rel_norm, -1, keepdim=True)*rel_norm 
            pos_t_embed = pos_t_embed - torch.sum(pos_t_embed*rel_norm, -1, keepdim=True)*rel_norm
            neg_t_embed = neg_t_embed - torch.sum(neg_t_embed*rel_norm, -1, keepdim=True)*rel_norm
            pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)  
            neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)
        
        elif kg_type == 'TransD':
            transfer_r_embed = self.rel_transfer(r)           
            transfer_h_embed = self.ent_transfer(h)           
            transfer_pos_t_embed = self.ent_transfer(pos_t)
            transfer_neg_t_embed = self.ent_transfer(neg_t)
            
            h_embed = F.normalize(
                _resize(h_embed, -1, transfer_r_embed.size()[-1]) + torch.sum(h_embed*transfer_h_embed, -1, keepdim=True)*transfer_r_embed, p = 2, dim = -1
            )
            pos_t_embed = F.normalize(
                _resize(pos_t_embed, -1, transfer_r_embed.size()[-1]) + torch.sum(pos_t_embed*transfer_pos_t_embed, -1, keepdim=True)*transfer_r_embed, p = 2, dim = -1
            )
            neg_t_embed = F.normalize(
                _resize(neg_t_embed, -1, transfer_r_embed.size()[-1]) + torch.sum(neg_t_embed*transfer_neg_t_embed, -1, keepdim=True)*transfer_r_embed, p = 2, dim = -1
            )
            
            pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1) 
            neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)  

        elif kg_type == 'ConvKB':
            pos_score = _convkb_score(h_embed, r_embed, pos_t_embed, self.entity_dim, self.kernels, self.conv1_bn, \
                                      self.conv_layer, self.conv2_bn, self.non_linearity, self.dropout, self.fc_layer)
            neg_score = _convkb_score(h_embed, r_embed, neg_t_embed, self.entity_dim, self.kernels, self.conv1_bn, \
                                      self.conv_layer, self.conv2_bn, self.non_linearity, self.dropout, self.fc_layer)

        elif kg_type == 'KGET':
            h_embed = self.ent_bn(h_embed)[:, None]                # (kg_batch_size, 1, entity_dim)
            r_embed = self.rel_bn(r_embed)[:, None]                # (kg_batch_size, 1, entity_dim)
            fusion_feature = torch.cat((h_embed, r_embed), dim=1)  # (kg_batch_size, 2, entity_dim) 
            embd_edges = self.encoder(fusion_feature)              # (kg_batch_size, 2, entity_dim)
            if self.decode_method == 'threemult': 
                src_edges = embd_edges[:, 0, :] * embd_edges[:, 1, :]
                pos_score = torch.sum(src_edges * pos_t_embed, dim=1)  
                neg_score = torch.sum(src_edges * neg_t_embed, dim=1) 
            elif self.decode_method == 'twomult':
                src_edges = embd_edges[:, 1, :]                        
                pos_score = torch.sum(src_edges * pos_t_embed, dim=1)  
                neg_score = torch.sum(src_edges * neg_t_embed, dim=1) 
            elif self.decode_method == 'distance_based':
                pos_score = torch.sum(torch.pow(embd_edges[:, 0, :] + embd_edges[:, 1, :] - pos_t_embed, 2), dim=1)  
                neg_score = torch.sum(torch.pow(embd_edges[:, 0, :] + embd_edges[:, 1, :] - neg_t_embed, 2), dim=1)  
        
        kg_loss = F.softplus((neg_score - pos_score))      
        kg_loss = torch.mean(kg_loss)
        l2_loss = self._L2_loss_mean(h_embed) + self._L2_loss_mean(r_embed) + self._L2_loss_mean(pos_t_embed) + self._L2_loss_mean(neg_t_embed)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss  
        return loss

    def _L2_loss_mean(self, x):
        """Compute l2 normalization i.e. (1/2) * ||w||^2 """
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)
    
    def update_attention_batch(self, h_list, t_list, r_idx):
        """Update attention matrix for every relation
        Args:
            h_list(list): a id list of head entities appearing in all triples\
            t_list(list): a id list of tail entities appearing in all triples
            r_idx(list) : a id list of relations
        """
        r_embed = self.relation_embed.weight[r_idx]  
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_embed.weight[h_list]
        t_embed = self.entity_embed.weight[t_list]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list    # (len(h_list)), )

    def update_attention(self, h_list, t_list, r_list, relations):
        # h_list is a id list of head entity appearing in all triples of 10 relations and so are t_list and r_list
        # relations: id list of 10 relations
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)    # (len(index_list), )
            cols.append(batch_t_list)    # (len(index_list), )
            values.append(batch_v_list)  # (len(index_list), )

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def forward(self, *input, mode): 
        if mode == 'train_ap':
            return self.calc_ap_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)


class Encoder(nn.Module): 
    '''Define encoder with n encoder layers'''
    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, decoder, dropout_dict):
        super(Encoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_input = d_input     
        self.n_layers = n_layers  
        self.n_head = n_head      
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = nn.Dropout(dropout_dict['dr_enc'])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, decoder, dropout_dict)
            for _ in range(n_layers)])

    def forward(self, edges):    # edges.shape: (batch_size, 2, d_emb) and 2 means the head entity and relation
        enc_output = self.dropout(edges)  

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output


class EncoderLayer(nn.Module):
    '''The main layer of encoder: multi-head attention + FFN'''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, decoder, dropout_dict):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout_dict)
        self.pos_ffn = PositionwiseFeedForwardUseConv(d_model, d_inner, \
            dropout=dropout_dict['dr_pff'], decode_method=decoder)  

    def forward(self, enc_input):                # enc_input.shape:  (batch_size, 2, d_emb)
        enc_output = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)    # enc_output.shape: (batch_size, 2, d_emb)

        return enc_output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout_dict):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), dropout=dropout_dict['dr_sdp'])
        self.layer_norm = nn.LayerNorm(d_model)  

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout_dict['dr_mha'])

    def forward(self, q, k, v, mask=None):  # q, k, v shape: (batch_size, 2, d_emb) and 2 means head entity and relation
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # b = a.contiguous() breaks the dependency between these two variables a and b 
        # and here is to make residual=q unchanged
        q = q.transpose(1, 2).contiguous().view(-1, len_q, d_k)  
        k = k.transpose(1, 2).contiguous().view(-1, len_k, d_k)  
        v = v.transpose(1, 2).contiguous().view(-1, len_v, d_v)  

        output = self.attention(q, k, v)    
        output = output.view(sz_b, n_head, len_q, d_v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # b x lq x (h*dv)

        output = self.dropout(self.fc(output))    # transform to dimension d_emb
        output = self.layer_norm(output + residual)

        return output  # output.shape: (batch_size, length_q, d_model)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, dropout=0.2):
        super().__init__()
        self.temperature = temperature      
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):  
        # Input query, key, value are the same size: (batch_size, length_seq, d_emb) and 
        # in this paper, length_seq=2 for a sequence consisting of head entity and relation
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature          # scaled attention scores
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class PositionwiseFeedForwardUseConv(nn.Module):
    '''Implement FFN equation in the paper'''
    def __init__(self, d_in, d_hid, dropout=0.3, decode_method='twomult'):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  
        nn.init.kaiming_uniform_(self.w_1.weight, mode='fan_out', nonlinearity='relu')  
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  
        nn.init.kaiming_uniform_(self.w_2.weight, mode='fan_in', nonlinearity='relu')
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
        self.decode_method = decode_method

    def forward(self, x):   # x.shape: (batch_size, length_seq, dim_embeddings)
        residual = x  
        output = x.transpose(1, 2)  
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output) 
        output = self.layer_norm(output + residual)  # use residual shortcut and layer_norm
        return output


class Aggregator(nn.Module):
    """Aggregate information from neighbors to get the final representation of nodes"""
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)      
            nn.init.xavier_normal_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)  
            nn.init.xavier_normal_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)     
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)    
            nn.init.xavier_normal_(self.linear1.weight)
            nn.init.xavier_normal_(self.linear2.weight)

        else:
            raise NotImplementedError

    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_all_entities, in_dim)
        A_in:            (n_all_entities, n_all_entities), torch.sparse.FloatTensor
        """
        side_embeddings = torch.matmul(A_in, ego_embeddings)  # A_in is the laplacian matrix which removed its own connection

        if self.aggregator_type == 'gcn':
            embeddings = ego_embeddings + side_embeddings          
            embeddings = self.activation(self.linear(embeddings))  

        elif self.aggregator_type == 'graphsage':
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings) 
        return embeddings    # return updated embeddings after a gcn layer of information aggregation


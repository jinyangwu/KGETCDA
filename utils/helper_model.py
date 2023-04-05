#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jinyang Wu


import os
import torch


def early_stopping(metric, best_metric, current_steps, stopping_steps):
    '''Determine whether to stop'''
    is_stop = False
    if best_metric < metric:
        current_steps = 0
        best_metric = metric
    else:
        current_steps += 1
        if current_steps >= stopping_steps:
            is_stop = True
    return best_metric, current_steps, is_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    '''Save pytorch model and model_dir is the saved path'''
    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.remove(old_model_state_file)  

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)


def load_model(model, model_path):
    '''Load model from the model_path'''
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

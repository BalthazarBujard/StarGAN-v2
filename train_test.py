# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:04:05 2023
Simple script to build first draft of training loop
To be implmeented in the Trainer class

@author: balth
"""


import numpy as np
import matplotlib.pyplot as plt
from architecture.Model import Model
from dataloader.Dataloader import *
from train.Trainer import *
import torch
from munch import Munch

#%% test Trainer 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains (2 for celeba, 3 for afhq)')
parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
parser.add_argument('--eval_iter', type=int, default=50,
                        help='evaluate every ...')
parser.add_argument('--save_iter', type=int, default=2,
                        help='Save model iteration')
parser.add_argument('--lambda_ds', type=float, default=1.,
                        help='Diversification style loss coefficient')
parser.add_argument('--resume_iter', type=int, default=0,
                        help='Start iteration')
parser.add_argument('--max_iter', type=int, default=200,
                        help='Style code dimension')

parser.add_argument('--mode', type=str, default="train",
                        help='train or test mode')
parser.add_argument('--lr', type=float, default=10e-4,
                        help='Default learning rate')
parser.add_argument('--f_lr', type=float, default=10e-6,
                        help='Mapping Network learning rate')
parser.add_argument('--beta1', type=float, default=0.,
                        help='Adam optimizer first momentum')
parser.add_argument('--beta2', type=float, default=0.99,
                        help='Adam optimizer second momentum')
parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Model weights L2 regularization')
parser.add_argument('--checkpoint_dir', type=str, default="",
                        help='Directory to save models ceckpoints')
parser.add_argument('--log_iter', type=int, default=1,
                        help='log out every ...')
parser.add_argument('--fan_pretrained_fname', type=str, default="weights.pth",
                        help='FAN Pretrained')
params = parser.parse_args()

root="celeba_hq_256/train" #if local
# root = "../shared/stargan_folder/data/celeba_hq/train" #if gpu server
#Model,CopyModel = Model(params)

trainer = Trainer(params)

train_loader = get_loader(root, params.batch_size, params.img_size)
loaders = Munch(train=train_loader)
trainer.train(loaders)





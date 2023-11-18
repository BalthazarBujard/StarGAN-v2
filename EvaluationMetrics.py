# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:24:41 2023

The evaluation metrics used in StarGANv2 paper use the LPIPS and FID metrics.
Those metrics use pretrained models to measure quality and diversity in the generated images.

FID : measure the similarity of distribution from real world images and generated images
using pretrained inception model. the real and generated imagea are fed into the 
model and the feature map are used to measure distribution accross samples adn then compare those distributions 
using the frechet distance. 

@author: balth
"""

from scipy.linalg import sqrtm
import numpy as np


def fid(mu_real, mu_fake, cov_real, cov_fake):
    
    crcf,_ = sqrtm(np.dot(cov_real,cov_fake),disp=False)
    
    d = np.sum(mu_real**2 - mu_fake**2) + np.trace(cov_real+cov_fake-2*crcf)
    
    return d


def compute_mu_cov()



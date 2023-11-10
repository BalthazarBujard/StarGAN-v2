import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
        # The mean and standard-deviation are calculated per-dimension separately for each object in a mini-batch
        # Different from BN layers, here µ(x) and σ(x) are computed across spatial dimensions independently for each channel and each sample
        self.norm = nn.InstanceNorm2d(num_features,affine=False)
        # Fully connected layer
        #  The factor of 2 is because the output is split into two parts: one for scaling (gamma) and one for shifting (beta) during normalization.
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        # The forward method takes two inputs: x is the input tensor to be normalized, and s is the style tensor.
        h = self.fc(s)
        # reshape to (batch_size, num_features*2, 1, 1)
        # h = h.view(h.size(0), h.size(1), 1, 1)
        h = torch.reshape(h,(h.size(0), h.size(1), 1, 1))
        # retrieve gamma and beta from h
        # Splits the tensor h into 2 sub-tensors (gamma and beta) along dimension 1
        gamma, beta = torch.tensor_split(h, 2, dim=1) 
        # return the normalized input tensor x 
        return (1 + gamma) * self.norm(x) + beta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlk(nn.Module):
    """
    Pre-activation Residual Block
    
    This Block is a generic and customizable pre-activation rsesidual unit [https://arxiv.org/abs/1603.05027].
    
    It can function as an upsampling, downsampling, or intermediate residual block based on the
    specified instance normalization and resampling techniques.

    Parameters:
        - in_size (int): Number of input channels.
        - out_size (int): Number of output channels.
        - resampling (str,optional): Type of the block, specifying upsampling ('UP') or downsampling ('DOWN'). Defaults to None.
        - normalization (str, optional): Type of instance normalization, either 'IN' or 'AdaIN'. Defaults to None.
        - S_size (int, optional): Length of the style code used for AdaIN normalization. Defaults to None.

    Methods:
        - skip_con(x): Implements the skip connection based on the specified resampling type.
        - convBlock(x, s=None): Implements the convolutional block with optional instance normalization.
        - forward(x, s=None): Combines the skip connection and convolutional block, dividing by sqrt(2) for unit variance.
    """
    
    def __init__(self, in_size, out_size, resampling, normalization=None, S_size=None):
        super().__init__()

        # Initialize parameters
        self.in_size = in_size
        self.out_size = out_size
        self.resampling = resampling        
        self.normalization = normalization  
        self.S_size = S_size                

        # Activation function
        self.activation = nn.LeakyReLU(0.2)

        # Convolution layers
        self.conv1 = nn.Conv2d(in_size, in_size, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_size, out_size, 3, 1, 1)

        # Normalization layers
        if normalization == 'IN':
            self.norm1 = nn.InstanceNorm2d(in_size, affine=True)
            self.norm2 = nn.InstanceNorm2d(in_size, affine=True)
        elif normalization == 'AdaIN':
            self.norm1 = AdaIN(S_size, in_size) 
            self.norm2 = AdaIN(S_size, out_size)
  
    def skip_con(self, x):
        # Skip connection based on the specified resampling type
        if self.resampling == 'Down':
            x = F.avg_pool2d(x, 2)
        elif self.resampling == 'UP':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        #if self.learned_sc:   ? 
            #x = self.conv1x1(x)
        return x
    
    def convBlock(self, x, s=None):
        # Apply instance normalization or AdaIN based on the specified normalization type
        if self.normalization == "IN":
            x = self.activation(self.norm1(x))      
        elif self.normalization == "AdaIN":
            x = self.conv1(self.activation(self.norm1(x, s)))        
        else:
            x = self.conv1(self.activation(x))

        # Resampling (up/down)
        if self.resampling == 'Down':
            x = F.avg_pool2d(x, 2)

        elif self.resampling == 'UP':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.conv1(x)

        # Apply instance normalization to the second convolution
        if self.normalization:
            x = self.conv2(self.activation(self.norm2(x)))
        else:
            x = self.conv2(self.activation(x))

        return x

    def forward(self, x, s=None):
        # Return the sum of skip connection and convolution block output, divided by sqrt(2) to get unit variance
        return (self.skip_con(x) + self.convBlock(x, s)) / math.sqrt(2)
        #if self.w_hpf == 0: ?

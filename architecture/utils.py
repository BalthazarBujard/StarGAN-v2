import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
    This file contains building blocks for our network modules.
    The key modules include: Adaptive instance normalization, Residual block
"""

class AdaIN(nn.Module):
    """
        Initialize the AdainResBlk module.
        style_dim (int): The dimension of the style vector. Defaults to 64.
        num_features (int): dimension of the input feature map  
    """
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
        """
        Forward pass of the AdaIN.

        Parameters:
        x (torch.Tensor): The input tensor.
        s (torch.Tensor): The style vector.

        Returns:
        torch.Tensor: The normalized input tensor x 
        """
        # The forward method takes two inputs: x is the input tensor to be normalized, and s is the style tensor.
        h = self.fc(s)
        # reshape to (batch_size, num_features*2, 1, 1)
        # h = h.view(h.size(0), h.size(1), 1, 1)
        h = torch.reshape(h,(h.size(0), h.size(1), 1, 1))
        # retrieve gamma and beta from h
        # Splits the tensor h into 2 sub-tensors (gamma and beta) along dimension 1
        gamma, beta = torch.tensor_split(h, 2, dim=1) 
      
        return (1 + gamma) * self.norm(x) + beta
    
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
        - normalizationMethod (str, optional): Type of instance normalization, either 'IN' or 'AdaIN'. Defaults to None.
        - S_size (int, optional): Length of the style code used for AdaIN normalization. Defaults to None.

    Methods:
        - skip_con(x): Implements the skip connection based on the specified resampling type.
        - convBlock(x, s=None): Implements the convolutional block with optional instance normalization.
        - forward(x, s=None): Combines the skip connection and convolutional block, dividing by sqrt(2) for unit variance.
    """
    
    def __init__(self, in_size, out_size, resampling=None ,normalizationMethod =None, S_size=64 , wFilter
                  = 0):
        super().__init__()

        # Initialize parameters
        self.in_size = in_size
        self.out_size = out_size
        self.resampling = resampling  
        self.normalizationMethod = normalizationMethod  
        self.S_size = S_size                
        self.wFilter = wFilter
        # Activation function
        self.activation = nn.LeakyReLU(0.2)

        # Convolution layers
        if self.normalizationMethod=='AdaIN':
            self.conv1 = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.conv2 = nn.Conv2d(out_size, out_size, 3, 1, 1)
        else:
            self.conv1 = nn.Conv2d(in_size, in_size, 3, 1, 1)
            self.conv2 = nn.Conv2d(in_size, out_size, 3, 1, 1)
        
        self.conv1x1 = nn.Conv2d(in_size, out_size, 1, 1, 0, bias=False)
        
        # Normalization layers
        if self.normalizationMethod == 'IN':
            self.norm1 = nn.InstanceNorm2d(in_size, affine=True)
            self.norm2 = nn.InstanceNorm2d(in_size, affine=True)
        elif self.normalizationMethod == 'AdaIN':
            self.norm1 = AdaIN(S_size, in_size) 
            self.norm2 = AdaIN(S_size, out_size)
  
    def skip_con(self, x):
        # Skip connection based on the specified resampling type
        if self.resampling == 'UP':
            #Down/up samples the input to either the given scale_factor
            x = F.interpolate(x, scale_factor=2) 
            
        if self.in_size!=self.out_size: 
            x = self.conv1x1(x)
        
        if self.resampling == 'DOWN':
            x = F.avg_pool2d(x, kernel_size=2)
        
        return x
    
    def convBlock(self, x, s=None):
        
        # Apply instance normalization or AdaIN based on the specified normalization type
        if self.normalizationMethod  == "IN":
            x = self.norm1(x)
            x = self.activation(x)
            x = self.conv1(x)
        elif self.normalizationMethod  == "AdaIN":
           x = self.norm1(x, s)
           x = self.activation(x)     
        else:
            x = self.activation(x)
            x = self.conv1(x)

        # Resampling (up/down)
        if self.resampling == 'DOWN':
            x = F.avg_pool2d(x, 2)
        elif self.resampling == 'UP':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
    
        if self.normalizationMethod  == "AdaIN":
            x = self.conv1(x)
        # Apply instance normalization to the second convolution
        if self.normalizationMethod  == 'IN':
            x = self.norm2(x)
            x = self.activation(x)
            x = self.conv2(x)
        elif self.normalizationMethod  == 'AdaIN':
            x = self.norm2(x, s)
            x = self.activation(x)
            x = self.conv2(x)
        else:
            x = self.activation(x)
            x = self.conv2(x)
        
        return x

    def forward(self, x, s=None):
        if self.normalizationMethod  == 'AdaIN' and self.wFilter >0  :
            return self.convBlock(x, s)
        else : 
            # Return the sum of skip connection and convolution block output, divided by sqrt(2) to get unit variance
            return (self.skip_con(x) + self.convBlock(x, s)) / math.sqrt(2)

#Redundant, replaced by ResBlk

class AdainResBlk(nn.Module):
    def __init__(self, input_dim, output_dim, style_dim=64, activate=nn.LeakyReLU(0.2), upsample=False):
        """
        Initialize the AdainResBlk module.

        Parameters:
        input_dim (int): The number of input channels.
        output_dim (int): The number of output channels.
        style_dim (int, optional): The dimension of the style vector. Defaults to 64.
        activate (torch.nn.Module, optional): The activation function to use. Defaults to LeakyReLU with a negative slope of 0.2.
        upsample (bool, optional): Flag to determine if the input should be upsampled. Defaults to False.
        """
        super(AdainResBlk, self).__init__()
        self.upsample = upsample  # Whether to upsample the input
        self.activate = activate  # Activation function
        self.adjust_dim = input_dim != output_dim  # Check if input and output dimensions are different

        # Create AdaIN and convolutional layers
        self.adain1 = AdaIN(style_dim, input_dim)
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.adain2 = AdaIN(style_dim, output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)

        # Create shortcut connection
        self.shortcut = self._make_shortcut(input_dim, output_dim)

    def _make_shortcut(self, input_dim, output_dim):
        """
        Create a shortcut connection for the block.

        Parameters:
        input_dim (int): The number of input channels.
        output_dim (int): The number of output channels.

        Returns:
        torch.nn.Module: Either a convolutional layer or an identity layer, depending on whether the dimensions need adjustment.
        """
        if self.adjust_dim:
            # If dimensions differ, use a 1x1 convolution to adjust the number of channels
            return nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)
        else:
            # If dimensions are the same, use an identity layer
            return nn.Identity()

    def forward(self, x, style):
        """
        Forward pass of the AdainResBlk.

        Parameters:
        x (torch.Tensor): The input tensor.
        style (torch.Tensor): The style vector.

        Returns:
        torch.Tensor: The output tensor of the block.
        """
        identity = x  # Store the original input for the shortcut connection

        if self.upsample:
            # Upsample the input and the identity if upsample is True
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            identity = F.interpolate(identity, scale_factor=2, mode='nearest')

        # Apply the AdaIN and convolutional layers
        x = self.adain1(x, style)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.adain2(x, style)
        x = self.activate(x)
        x = self.conv2(x)

        if self.adjust_dim:
            # Adjust the identity to match the dimensions if necessary
            identity = self.shortcut(identity)

        out = x + identity  # Add the shortcut connection
        return out / math.sqrt(2)  # Normalize the output
    


class FilterKernel(nn.Module):
    def __init__(self, wFilter, device):
        super(FilterKernel, self).__init__()
        filter_kernel = torch.tensor([[-1, -1, -1],
                              [-1, 8., -1],
                              [-1, -1, -1]]) / wFilter
        self.filter = filter_kernel
        
    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


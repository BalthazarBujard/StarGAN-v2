import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.utils import *

class Generator(nn.Module):
    """
        Initialize the Generator module.

        Parameters:
        - img_size (int): Desired image size.
        - style_dim (int): Dimension of the style vector.
        - max_dim (int) : max channel dimension for down/upsampling block
        - n_layers (int) : number of down/upsampling layers (4 for afhq and 5 for celeba_hq)

    """
    def __init__(self, img_size=256, style_dim=64, max_dim=512, n_layers=4):
        super().__init__()
        self.img_size = img_size
        # Define the initial convolution layer to process RGB input
        self.from_rgb = nn.Conv2d(3, 64, 3, 1, 1) # conv 1*1
        #self.from_rgb = nn.Conv2d(3,64,1,1,0) #vrai conv 1x1 sans padding
        # Initialize lists to store encoding and decoding blocks
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        # Define the final output layer
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(64, affine=True), # Layer Norma
            nn.LeakyReLU(0.2), # Leaky relu activation Function
            nn.Conv2d(64, 3, 1, 1, 0)) # conv 1*1

        # down/up-sampling blocks
        
        """
        self.encode.append(ResBlk(64, 128, normalize=True, downsample=True))
        self.decode.insert(0, AdainResBlk(128, 64, style_dim,upsample=True))

        self.encode.append(ResBlk(128, 256, normalize=True, downsample=True))
        self.decode.insert(0, AdainResBlk(256, 128, style_dim,upsample=True))  

        self.encode.append(ResBlk(256, 512, normalize=True, downsample=True))
        self.decode.insert(0, AdainResBlk(512, 256, style_dim,upsample=True)) 

        self.encode.append(ResBlk(512, 512, normalize=True, downsample=True))
        self.decode.insert(0, AdainResBlk(512, 512, style_dim,upsample=True))  
        """
        dim_in = 64
        for _ in range(n_layers):
            dim_out = min(dim_in*2,max_dim)
            
            self.encode.append(ResBlk(dim_in, dim_out, resampling='DOWN' ,normalizationMethod ='IN', S_size=style_dim))
            self.decode.insert(0, ResBlk(dim_out, dim_in, resampling='UP' ,normalizationMethod ='AdaIN', S_size=style_dim))

            dim_in=dim_out
        
        # self.encode.append(ResBlk(64, 128, resampling='DOWN' ,normalizationMethod ='IN', S_size=style_dim))
        # self.decode.insert(0, ResBlk(128, 64, resampling='UP' ,normalizationMethod ='AdaIN', S_size=style_dim))

        # self.encode.append(ResBlk(128, 256, resampling='DOWN' ,normalizationMethod ='IN', S_size=style_dim))
        # self.decode.insert(0, ResBlk(256, 128, resampling='UP' ,normalizationMethod ='AdaIN', S_size=style_dim))

        # self.encode.append(ResBlk(256, 512, resampling='DOWN' ,normalizationMethod ='IN', S_size=style_dim))
        # self.decode.insert(0, ResBlk(512, 256, resampling='UP' ,normalizationMethod ='AdaIN', S_size=style_dim))

        # self.encode.append(ResBlk(512, 512, resampling='DOWN' ,normalizationMethod ='IN', S_size=style_dim))
        # self.decode.insert(0, ResBlk(512, 512, resampling='UP' ,normalizationMethod ='AdaIN', S_size=style_dim))

        
        # bottleneck blocks

        # Append 2 Residual Blocks to the encoding list
        self.encode.append(ResBlk(512, 512,normalizationMethod ='IN', S_size=style_dim))
        self.encode.append(ResBlk(512, 512,normalizationMethod ='IN', S_size=style_dim))
        # Append 2 Adain Residual Blocks to the decoding list (inserted from the left)
        self.decode.insert(0, ResBlk(512, 512,normalizationMethod ='AdaIN', S_size=style_dim))
        self.decode.insert(0, ResBlk(512, 512,normalizationMethod ='AdaIN', S_size=style_dim))

    def forward(self, x, s):
        """
        Forward pass of the Generator module.

        Parameters:
        - x (torch.Tensor): The input tensor.
        - s (torch.Tensor): The style vector.

        Returns:
        torch.Tensor: The output RGB image.
        """
        # Initial processing of the RGB input 
        x = self.from_rgb(x)
         # Encoding phase
        for block in self.encode:
            # Apply the current decoding block,
            x = block(x)
        # Decoding phase
        for block in self.decode:
            # Apply the current decoding block, conditioned on style 's'
            x = block(x, s)
        # Final output layer to produce the RGB image (1*1 Conv)
        return self.to_rgb(x)



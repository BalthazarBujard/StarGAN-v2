import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class Generator(nn.Module):
    """
        Initialize the Generator module.

        Parameters:
        - img_size (int): Desired image size.
        - style_dim (int): Dimension of the style vector.
        - max_conv_dim (int): Maximum convolutional dimension.
    """
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
         # Calculate the initial input dimension based on the desired image size
        dim_in = 2**14 // img_size
        self.img_size = img_size
         # Define the initial convolution layer to process RGB input
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1) # conv 1*1
        # Initialize lists to store encoding and decoding blocks
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        # Define the final output layer
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True), # Layer Norma
            nn.LeakyReLU(0.2), # Leaky relu activation Function
            nn.Conv2d(dim_in, 3, 1, 1, 0)) # conv 1*1

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        count = 0
        while count < repeat_num:
            # Determine the output dimension for the current block and clip dimensions > max_conv_dim
            dim_out = min(dim_in*2, max_conv_dim)
            # Append a Residual Block to the encoding list
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            # Append from the left Adin Residual Block to the decoding list
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim,upsample=True))  
            # Update the input dimension for the next block
            dim_in = dim_out
            count += 1

        # bottleneck blocks

        # Append 2 Residual Blocks to the encoding list
        self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
        self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
        # Append 2 Adain Residual Blocks to the decoding list (inserted from the left)
        self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim))
        self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim))

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

# JUST TO TEST !!!!!!!!!!

style_length = 64
batch_size = 1
tensor = torch.rand(batch_size, 3, 256, 256)  # (batch_size, channels, height, width)
style = torch.rand(batch_size, style_length)  # (batch_size, style_dim)
generator_model = Generator()
output = generator_model(tensor, style)

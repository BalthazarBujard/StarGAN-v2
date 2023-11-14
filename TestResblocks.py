import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ResBlk import *
from Generator import * 

# Define input dimensions
batch_size = 4
in_channels = 3
style_dim = 64
height, width = 256, 256
in_size = in_channels
out_size = 16  

# Create dummy input
input_tensor = torch.randn(batch_size, in_channels, height, width)
style_code = torch.randn(batch_size, style_dim)

# Instantiate the ResBlk class
# Replace 'resampling', 'nomalize', 'normalizationMethod', and 'S_size' with desired values
#
res_blk_yang =  AdainResBlk(in_size, out_size, style_dim,upsample=True)
res_blk_younes = ResBlk(in_size, out_size, resampling='UP', nomalize=None, normalizationMethod='AdaIN',S_size= style_dim )
# If using AdaIN, create a dummy style code and pass it to the forward method
# if res_blk.normalizationMethod == 'AdaIN':
#     S_size = 128  # Change as needed
#     
#     
# else:

output_yang = res_blk_yang(input_tensor, style_code)
print(output_yang.shape)
print(output_yang[0,0,:,:])
output_younes = res_blk_younes(input_tensor, style_code)
print(output_younes.shape)



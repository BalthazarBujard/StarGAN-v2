import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *

# Define input dimensions
batch_size = 1
style_dim = 64

height, width = 256, 256 #in
in_channels = 64
out_size = 128

# Create dummy input
input_tensor = torch.randn(batch_size, in_channels, height, width)
style_code = torch.randn(batch_size, style_dim)

#res_blk_yang =  AdainResBlk(in_channels, out_size, style_dim,upsample=True)
res_blk_younes = ResBlk(in_channels, out_size, 'DOWN')

output_younes = res_blk_younes(input_tensor, style_code)
print(output_younes.shape)



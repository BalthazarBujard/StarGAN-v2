import torch
from utils import ResBlk

# Define input dimensions
batch_size = 1
style_dim = 64

height, width = 256, 256 #in
in_channels = 64
out_size = 128

# Create dummy input
input_tensor = torch.randn(batch_size, in_channels, height, width)
style_code = torch.randn(batch_size, style_dim)

res_blk = ResBlk(in_channels, out_size, 'DOWN')

output = res_blk(input_tensor, style_code)
print(output.shape)

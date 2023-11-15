import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *

class StyleEncoder(nn.Module):
    def __init__(self, S_size, outp_branches):
        super().__init__()
        
        # Sequential layers for feature extraction
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            ResBlk(64, 128, 'DOWN'),
            ResBlk(128, 256, 'DOWN'),
            ResBlk(256, 512, 'DOWN'),
            ResBlk(512, 512, 'DOWN'),
            ResBlk(512, 512, 'DOWN'),
            ResBlk(512, 512, 'DOWN'),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        
        # Parallel branches for different output branches
        self.parallel = nn.ModuleList()
        for i in range(outp_branches):
            self.parallel += [nn.Linear(512, S_size)]

    def forward(self, x, branch):
        x = self.sequential(x)
        #x = torch.flatten(x)
        x = x.view(x.size(0), -1)
        # Generate output for different branches in parallel
        outp = torch.stack([linear(x) for linear in self.parallel], dim=1)
        s = outp[torch.arange(branch.size(0)), branch]

        return s

# Example usage of StyleEncoder
# Instantiate the StyleEncoder class
input_size = 256  # Placeholder value for input size
S_size = 64  # Placeholder value for S_size
outp_branches = 3  # Placeholder value for the number of output branches
encoder = StyleEncoder(S_size, outp_branches)

branch = 1
# Example input tensors
x = torch.randn(1, 3, input_size, input_size)  # Placeholder input tensor

# Forward pass through the StyleEncoder
output = encoder(x, branch)
print("Output shape:", output.shape)  # Print the shape of the output tensor
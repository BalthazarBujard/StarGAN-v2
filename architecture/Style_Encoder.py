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
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,512, 4, 1, 1),
            nn.LeakyReLU(0.2),
        )
        
        # Parallel branches for different output branches
        self.parallel = nn.ModuleList()
        for i in range(outp_branches):
            self.parallel += [nn.Linear(512, S_size)]

    def forward(self, x, branch):
        x = self.sequential(x)
        x = x.view(x.size(0), -1)
        # Generate output for different branches in parallel
        outp = torch.stack([linear(x) for linear in self.parallel], dim=1)
        s = outp[torch.arange(branch.size(0)).to(branch.device), branch] # to(branch.device) to avoid Runtime error

        return s


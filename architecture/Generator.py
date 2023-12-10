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
    def __init__(self, img_size=256, style_dim=64, max_dim=512 , wFilter = 1):
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
        dim_in = 64 #input dimension of encoder after 
        n_layers = int(np.log2(img_size)) - 4 #-> downscale to 16x16 the feature maps, no matter the img size -> bigger img deeper network
        if wFilter>0: n_layers+=1
        #one more layer for faces (aka if Wfilter > 0) -> 8x8
        for _ in range(n_layers):
            dim_out = min(dim_in*2,max_dim) #we double the number of filters every step (Except bottleneck) until 512
            
            self.encode.append(ResBlk(dim_in, dim_out, resampling='DOWN' ,normalizationMethod ='IN', S_size=style_dim))
            self.decode.insert(0, ResBlk(dim_out, dim_in, resampling='UP' ,normalizationMethod ='AdaIN', S_size=style_dim,wFilter=wFilter))

            dim_in=dim_out
        
        # bottleneck blocks

        # Append 2 Residual Blocks to the encoding list
        self.encode.append(ResBlk(512, 512,normalizationMethod ='IN', S_size=style_dim))
        self.encode.append(ResBlk(512, 512,normalizationMethod ='IN', S_size=style_dim))
        # Append 2 Adain Residual Blocks to the decoding list (inserted from the left)
        self.decode.insert(0, ResBlk(512, 512,normalizationMethod ='AdaIN', S_size=style_dim,wFilter=wFilter))
        self.decode.insert(0, ResBlk(512, 512,normalizationMethod ='AdaIN', S_size=style_dim,wFilter=wFilter))

        if wFilter > 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.filter = FilterKernel(wFilter, device)
        

    def forward(self, x, s,FAN_masks):
        """
        Forward pass of the Generator module.

        Parameters:
        - x (torch.Tensor): The input tensor.
        - s (torch.Tensor): The style vector.
        - fan_masks (Optional[List[torch.Tensor]]): List of FAN masks for different resolutions.
        Returns:
        torch.Tensor: The output RGB image.
        """
        # Initial processing of the RGB input 
        x = self.from_rgb(x)
        saved_feature_maps  = {}
         # Encoding phase
        for block in self.encode:
            #only apply FAN to faces and during the up/downsampling -> not for 256x256 and 16x16 constant covolutions
            if (FAN_masks is not None) and (x.size(2) in [32, 64, 128]):
                saved_feature_maps [x.size(2)] = x #save current input -> skip connections ???
            # Apply the current decoding block,
            x = block(x)
        # Decoding phase
        for block in self.decode:
            # Apply the current decoding block, conditioned on style 's'
            x = block(x, s)
            if (FAN_masks is not None) and (x.size(2) in [32, 64, 128]):
                #use landamrks heatmap during decoding phase <-> generating face
                #bring attention to key features in a face
                mask = FAN_masks[0] if x.size(2) == 32 else FAN_masks[1] #[0] contains whole landmarks and [1] only center features : eyes, nose
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear') #downscale to same size as input
                x = x + self.filter(mask * saved_feature_maps[x.size(2)])
        # Final output layer to produce the RGB image (1*1 Conv)
        return self.to_rgb(x)



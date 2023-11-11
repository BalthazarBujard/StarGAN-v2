import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Create convolutional layers
        self.conv_layers = self._make_conv_layers(input_dim, output_dim, style_dim)
        # Create shortcut connection
        self.shortcut = self._make_shortcut(input_dim, output_dim)

    def _make_conv_layers(self, input_dim, output_dim, style_dim):
        """
        Create convolutional layers for the block.

        Parameters:
        input_dim (int): The number of input channels.
        output_dim (int): The number of output channels.
        style_dim (int): The dimension of the style vector.

        Returns:
        torch.nn.Sequential: A sequence of layers comprising the convolutional part of the block.
        """
        layers = nn.Sequential(
            AdaIN(style_dim, input_dim),  # First AdaIN layer
            self.activate,  # Activation function
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),  # First convolutional layer
            AdaIN(style_dim, output_dim),  # Second AdaIN layer
            self.activate,  # Activation function
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)  # Second convolutional layer
        )
        return layers

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

        out = self.conv_layers(x, style)  # Apply the convolutional layers

        if self.adjust_dim:
            # Adjust the identity to match the dimensions if necessary
            identity = self.shortcut(identity)

        out += identity  # Add the shortcut connection
        return out / math.sqrt(2)  # Normalize the output

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
tensor_length = (3,256,256)
style_length = 64
style = torch.rand(style_length)
tensor = torch.rand(tensor_length)

generator_model = Generator()

output = generator_model(tensor,style)
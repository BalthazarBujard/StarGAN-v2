import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
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
        # The forward method takes two inputs: x is the input tensor to be normalized, and s is the style tensor.
        h = self.fc(s)
        # reshape to (batch_size, num_features*2, 1, 1)
        # h = h.view(h.size(0), h.size(1), 1, 1)
        h = torch.reshape(h,(h.size(0), h.size(1), 1, 1))
        # retrieve gamma and beta from h
        # Splits the tensor h into 2 sub-tensors (gamma and beta) along dimension 1
        gamma, beta = torch.tensor_split(h, 2, dim=1) 
        # return the normalized input tensor x 
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


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        """
        Initialize the MappingNetwork module.

        Parameters:
        latent_dim (int, optional): The dimension of the latent space. Defaults to 16.
        style_dim (int, optional): The dimension of the style vector. Defaults to 64.
        num_domains (int, optional): The number of different domains for style encoding. Defaults to 2.
        """
        super(MappingNetwork, self).__init__()

        # Shared layers are common across all domains
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Unshared layers are specific to each domain
        self.unshared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, style_dim)
            ) for _ in range(num_domains)
        ])

    def forward(self, x, y):
        """
        Forward pass of the MappingNetwork.

        Parameters:
        x (torch.Tensor): The input latent vector.
        y (torch.Tensor): The domain labels indicating which domain each input belongs to.

        Returns:
        torch.Tensor: The output style vector for each input in the corresponding domain.
        """
        x = self.shared_layers(x)  # Apply shared layers to the input

        # Process each domain that appears in y separately
        # y.unique() provides the unique domain indices present in y
        # For each unique domain index, apply the corresponding unshared layer to x
        domain_outputs = [self.unshared_layers[domain_idx](x) for domain_idx in y.unique()]

        # Concatenate the outputs for each domain according to the input domain labels
        out = torch.cat([domain_outputs[y[i]] for i in range(len(y))], dim=0)

        # Reshape the output to the desired format
        # The -1 in view function is a placeholder that gets automatically replaced with the correct number
        # to ensure the tensor is reshaped to have len(y) rows and style_dim columns.
        return out.view(len(y), -1, style_dim)



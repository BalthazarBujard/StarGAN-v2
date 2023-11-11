
import torch
import torch.nn as nn
import torch.nn.functional as F

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

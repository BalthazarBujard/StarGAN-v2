
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
        self.style_dim = style_dim
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
        x = self.shared_layers(x)  # Apply shared layers to the input

        # Process each sample with its corresponding unshared layer
        out = torch.stack([self.unshared_layers[y[i]](x[i].unsqueeze(0)) for i in range(len(y))], dim=0).squeeze(1)

        return out





# for test
mapping_network = MappingNetwork(latent_dim=16, style_dim=64, num_domains=2)

latent_vector = torch.randn(10, 16)
domain_labels = torch.randint(0, 2, (10,))
output = mapping_network(latent_vector, domain_labels)
print("Output shape:", output.shape)  # output desired [10, 64]

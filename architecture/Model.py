from Generator import *
from Mapping_Network import * 
from Style_Encoder import * 
from Discriminator import *
import munch
"""
Model Arguments :

- img_size: Image resolution (default: 256).
- num_domains: Number of domains (default: 2).
- latent_dim: Latent vector dimension (default: 16).
- hidden_dim: Hidden dimension of mapping network (default: 512).
- style_dim: Style code dimension (default: 64).
"""
def Model(params):
    """
    nn.DistributedDataParallel is used to parallelize data across multiple GPUs in a distributed fashion, 
    which can significantly speed up training times for large models or datasets.
    """
    generator = nn.DistributedDataParallel(Generator(params.img_size,params.style_dim))
    mapping_network = nn.DistributedDataParallel(MappingNetwork(params.latent_dim, params.style_dim, params.num_domains))
    style_encoder = nn.DistributedDataParallel(StyleEncoder(params.style_dim, params.num_domains))
    discriminator = nn.DistributedDataParallel(Discriminator(params.style_dim,params.num_domains))

    generator_copy = copy.deepcopy(generator)
    mapping_network_copy = copy.deepcopy(mapping_network)
    style_encoder_copy = copy.deepcopy(style_encoder)

    netwroks = munch.Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    
    netwroks_copy = munch.Munch(generator=generator_copy,
                     mapping_network=mapping_network_copy,
                     style_encoder=style_encoder_copy)
    
    return netwroks,netwroks_copy


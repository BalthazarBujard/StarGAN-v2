from .Generator import *
from .Mapping_Network import * 
from .Style_Encoder import * 
from .Discriminator import *
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from munch import Munch
from .FAN.models import FAN

"""
Model Arguments :

- img_size: Image resolution (default: 256).
- num_domains: Number of domains (default: 2 for celeba, 3 for afhq).
- latent_dim: Latent vector dimension (default: 16).
- hidden_dim: Hidden dimension of mapping network (default: 512).
- style_dim: Style code dimension (default: 64).
"""
def Model(params):
    """
    DistributedDataParallel is used to parallelize data across multiple GPUs in a distributed fashion, 
    which can significantly speed up training times for large models or datasets.
    """
    # Initialize the process group
    # dist.init_process_group(backend='cuda', rank=0, world_size=1)
    n_layers = 4 if params.wFilter>3 else 5 #5 layersa for celeba_hq -> depends on img size mainly
    generator = (Generator(params.img_size,params.style_dim, wFilter=params.wFilter))
    mapping_network = (MappingNetwork(params.latent_dim, params.style_dim, params.num_domains))
    style_encoder = (StyleEncoder(params.style_dim, params.num_domains))
    discriminator = (Discriminator(params.num_domains))

    generator_copy = copy.deepcopy(generator)
    mapping_network_copy = copy.deepcopy(mapping_network)
    style_encoder_copy = copy.deepcopy(style_encoder)
    discriminator_copy = copy.deepcopy(discriminator)

    networks = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    
    networks_copy = Munch(generator=generator_copy,
                     mapping_network=mapping_network_copy,
                     style_encoder=style_encoder_copy,
                     discriminator = discriminator_copy)

    #ADD FAN NETWORK WITH PRETRAINED WEIGHTS
    if params.wFilter>0 : #only if celeba_hq
        fan = FAN(pretrained_file=params.fan_pretrained_fname).eval().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #fan.get_heatmap = fan.module.get_heatmap
        networks.fan = fan
        networks_copy.fan = fan
    
    return networks,networks_copy


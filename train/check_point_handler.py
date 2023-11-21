import torch
import torch.nn.functional as F
import torch.nn as nn
from munch import Munch
from pathlib import Path
from architecture.Model import * 

class ModelCheckpointHandler(object):
    def __init__(self, checkpoint_format, **model_components):
        self.checkpoint_dir = Path(checkpoint_format)
        self.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        self.models_registry = model_components

    def add_models(self, **model_components):
        self.models_registry.update(model_components)

    def store_checkpoint(self, epoch_num):
        checkpoint_name = self.checkpoint_dir.with_name(self.checkpoint_dir.name.format(epoch_num))
        print(f'Checkpoint being saved at {checkpoint_name}...')
        checkpoint_data = {'epoch': epoch_num}
        for component_name, model in self.models_registry.items():
            checkpoint_data[component_name] = model.state_dict() if not self._check_parallel(model) else model.module.state_dict()
        torch.save(checkpoint_data, checkpoint_name)

    def retrieve_checkpoint(self, epoch_num):
        checkpoint_name = self.checkpoint_dir.with_name(self.checkpoint_dir.name.format(epoch_num))
        if not checkpoint_name.exists():
            raise FileNotFoundError(f'No checkpoint found at {checkpoint_name}')

        print(f'Retrieving checkpoint from {checkpoint_name}...')
        saved_data = torch.load(checkpoint_name, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
        
        for component_name, model in self.models_registry.items():
            if component_name in saved_data:
                model.load_state_dict(saved_data[component_name] if not self._check_parallel(model) else model.module.state_dict())
            else:
                print(f'Note: {component_name} not located in checkpoint.')

    def _check_parallel(self, model):
        return isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))



# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)




# # Test 
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--img_size', type=int, default=256,
#                         help='Image resolution')
# parser.add_argument('--num_domains', type=int, default=2,
#                         help='Number of domains')
# parser.add_argument('--latent_dim', type=int, default=16,
#                         help='Latent vector dimension')
# parser.add_argument('--hidden_dim', type=int, default=512,
#                         help='Hidden dimension of mapping network')
# parser.add_argument('--style_dim', type=int, default=64,
#                         help='Style code dimension')
# params = parser.parse_args()

# Network,CopyNetwork = Model(params)


# # Instantiate the model and checkpoint handler
# model = Model(params)
# checkpoint_handler = ModelCheckpointHandler('checkpoint_{0}.pth', **Network)

# # Save the initial state of the model
# checkpoint_handler.store_checkpoint(0)

# # Change the weights of the model
# with torch.no_grad():
#     Network['generator'].from_rgb.weight += 1  

# for key_network in ['generator', 'mapping_network', 'style_encoder', 'discriminator'] : 
#     print(key_network)
#     print("000000000000000")
#     print(Network[key_network])
# # Save the modified state of the model
# checkpoint_handler.store_checkpoint(1)

# # Load the initial state back into the model
# checkpoint_handler.retrieve_checkpoint(0)

# # Checking if the weights are back to their initial state
# original_state = torch.load('checkpoint_0.pth', map_location=torch.device('cpu'))['generator']

# loaded_state = Network['generator'].state_dict()

# # Compare the two states
# test_result = all(torch.equal(original_state[key], loaded_state[key]) for key in original_state)

# print(test_result)
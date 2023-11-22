# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:04:05 2023
Simple script to build first draft of training loop
To be implmeented in the Trainer class

@author: balth
"""


import numpy as np
import matplotlib.pyplot as plt
plt.plot()
plt.show()
from architecture.Model import Model
from dataloader.Dataloader import *
from train.Trainer import *
import torch
from munch import Munch

#%%
def reset_grad(optims):
    for optim in optims.values():
        optim.zero_grad()
#%% get model and input fetcher
# Initialize parameters
img_size = 256      # Image size
style_dim = 64      # Dimension of style representation
latent_dim = 16     # Dimension of latent space
num_domains = 2     # Number of domains for style transfer


params = Munch(img_size=256,num_domains=2,latent_dim=16,hidden_dim=512, style_dim=64,
               max_iter=100)

lrs = Munch(gde=10e-4,f=10e-6)
betas = [0,0.99]

#build model
# Instantiate models
generator = Generator(img_size, style_dim)  # Create generator model
discriminator = Discriminator(num_domains)  # Create discriminator model
style_encoder = StyleEncoder(style_dim, num_domains)  # Create style encoder model
mapping_network = MappingNetwork(latent_dim, style_dim, num_domains)  # Create mapping network model

device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')

nets = Munch(generator=generator.to(device), discriminator=discriminator.to(device),
             style_encoder=style_encoder.to(device), mapping_network=mapping_network.to(device))




#nets, nets_copy = Model(params) problem calling with Munch params

#optimizers
optimizers = Munch()
for key_network in ['generator', 'mapping_network', 'style_encoder', 'discriminator'] : 
    optimizers[key_network] = torch.optim.Adam(
        params=nets[key_network].parameters(),
        lr=lrs.f if key_network == 'mapping_network' else lrs.gde,
        betas=betas)


#get dataset and fetcher
root="../dataset/data/celeba_hq/train"
batch_size=1
train_loader = get_loader(root, batch_size, params.img_size)

input_fetcher = Fetcher(train_loader)

#%%

max_iter=params.max_iter
lambda_ds=1
init_lambda_ds=lambda_ds

for i in range(max_iter):
    
    inputs=next(input_fetcher)
    x_real,y_org = inputs.x,inputs.y
    x_ref1,x_ref2,y_trg = inputs.x_ref1,inputs.x_ref2,inputs.y_trg
    z1,z2 = inputs.z1,inputs.z2
    
    
    #train discriminator with latent code and reference image
    
    #latent code
    d_loss_latent, _ = loss_discriminator(nets, x_real, y_org, y_trg, z_trg=z1)
    #reset gradients
    reset_grad(optimizers)
    #backprop
    d_loss_latent.backward()
    #optimizer step
    optimizers.discriminator.step()
    
    print("discriminator latent code loss : ",d_loss_latent.cpu().detach().numpy())
    
    #ref image
    d_loss_ref, _ = loss_discriminator(nets, x_real, y_org, y_trg, x_ref=x_ref1)
    reset_grad(optimizers)
    d_loss_ref.backward()
    optimizers.discriminator.step()
    
    print("discriminator ref image loss : ",d_loss_ref.cpu().detach().numpy())
    
    #train generator
    g_loss_latent,_ = loss_generator(nets, x_real, y_org, y_trg,z_trgs=[z1,z2], lambda_ds=lambda_ds)
    reset_grad(optimizers)
    g_loss_latent.backward()
    #order doesnt matter ?
    optimizers.generator.step()
    optimizers.mapping_network.step()
    optimizers.style_encoder.step()
    
    
    print("generator latent code loss : ",g_loss_latent.cpu().detach().numpy())
    
    g_loss_ref,_ = loss_generator(nets, x_real, y_org, y_trg,x_refs=[x_ref1,x_ref2],lambda_ds=lambda_ds)
    reset_grad(optimizers)
    g_loss_ref.backward()
    #order doesnt matter ?
    optimizers.generator.step()
    optimizers.mapping_network.step()
    optimizers.style_encoder.step()
    
    print("generator ref image loss : ",g_loss_ref.cpu().detach().numpy())
    
    
    #lambda ds linear decay over max_iter
    if lambda_ds>0:
        lambda_ds-=init_lambda_ds/max_iter
    
    #visualise generator
    styles = mapping_network(z1,y_trg)
    example = nets.generator(x_ref1,styles)
    img_fake=torch.permute((example[0]+1)/2, [1,2,0]).detach().numpy()
    plt.imshow(img_fake)
    plt.show()
        
    if i>=5:
        break

    
#%% test Trainer 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains (2 for celeba, 3 for afhq)')
parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
parser.add_argument('--var', type=int, default=64,
                        help='dummy var for testing')
parser.add_argument('--eval_iter', type=int, default=50,
                        help='evaluate every ...')
parser.add_argument('--save_iter', type=int, default=50,
                        help='Save model iteration')
parser.add_argument('--lambda_ds', type=float, default=1.,
                        help='Diversification style loss coefficient')
parser.add_argument('--resume_iter', type=int, default=0,
                        help='Start iteration')
parser.add_argument('--max_iter', type=int, default=200,
                        help='Style code dimension')
parser.add_argument('--mode', type=str, default="train",
                        help='train or test mode')
parser.add_argument('--lr', type=float, default=10e-4,
                        help='Default learning rate')
parser.add_argument('--f_lr', type=float, default=10e-6,
                        help='Mapping Network learning rate')
parser.add_argument('--beta1', type=float, default=0.,
                        help='Adam optimizer first momentum')
parser.add_argument('--beta2', type=float, default=0.99,
                        help='Adam optimizer second momentum')
parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Model weights L2 regularization')
parser.add_argument('--checkpoint_dir', type=str, default="",
                        help='Directory to save models ceckpoints')
params = parser.parse_args()

root="../dataset/data/celeba_hq/train"

#Model,CopyModel = Model(params)

trainer = Trainer(params)

train_loader=train_loader = get_loader(root, params.batch_size, params.img_size)
loaders = Munch(train=train_loader)
trainer.train(loaders)





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
from Trainer import *
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

    








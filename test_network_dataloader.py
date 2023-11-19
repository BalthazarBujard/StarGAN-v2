# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:28:53 2023

@author: balth
"""

from architecture import *
from dataloader import Dataloader
import torch

root = "../dataset/data/celeba_hq/train"


train_loader=get_loader(root,batch_size=8,img_size=256)


loader_iter = iter(train_loader)
inputs = next(loader_iter)

x, y = inputs.x,inputs.y
x_ref1,x_ref2 = inputs.x_ref1,inputs.x_ref2
z1,z2,y_trg = inputs.z1, inputs.z2, inputs.y_trg


print("Input image shape :",x.shape)
print("Ref images shape :", x_ref1.shape,x_ref2.shape)
print("Latent code z shape :", z1.shape,z2.shape)
#%%

img_size=256
style_dim=64
latent_dim=16
num_domains=2 #3 for afhq

generator = Generator(img_size, style_dim)
mapping_network = MappingNetwork(latent_dim,style_dim,num_domains)
style_encoder = StyleEncoder(style_dim,num_domains)
discriminator=Discriminator(num_domains)

styles = mapping_network(z1,y_trg)

x_fake = generator(x,styles)

z=style_encoder(x_ref1,y_trg)

out=discriminator(x_fake,y_trg)


img_fake=torch.permute((x_fake[0]+1)/2, [1,2,0]).detach().numpy()
plt.imshow(img_fake)











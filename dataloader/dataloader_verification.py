# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:24:28 2023

@author: balth
"""

import matplotlib.pyplot as plt

#eviter probleme avec matplotlib et torch
plt.plot()
plt.show()

import torch
from torchvision import transforms
from Dataloader import StarDataset, get_loader

#%%
def denormalize_tensor(t):
    #convert from -1,1 to 0,1
    return (t+1)/2
#%% check dataset

root = "../../dataset/data/celeba_hq/train"

train_dataset = StarDataset(root, chunk="train")


inputs = train_dataset[0]

x, y = inputs.x,inputs.y
x_ref1,x_ref2 = inputs.x_ref1,inputs.x_ref2
z1,z2,y_trg = inputs.z1, inputs.z2, inputs.y_trg

crop = transforms.RandomResizedCrop(256, scale=(0.8,1), ratio=(0.9,1.1))

img=x

img_c = crop(img)

plt.imshow((torch.permute(img,[1,2,0])+1)/2)
plt.title(f"image of domain : {y}")
plt.show()

plt.imshow((torch.permute(img_c,[1,2,0])+1)/2)
plt.title(f"image of domain : {y}")
plt.show()

print(z1.shape)
print(z2.shape)
print(y)

#%% check dataloader


train_loader=get_loader(root,batch_size=8,img_size=256)


loader_iter = iter(train_loader)
inputs = next(loader_iter)

x, y = inputs.x,inputs.y
x_ref1,x_ref2 = inputs.x_ref1,inputs.x_ref2
z1,z2,y_trg = inputs.z1, inputs.z2, inputs.y_trg


print("Input image shape :",x.shape)
print("Ref images shape :", x_ref1.shape,x_ref2.shape)
print("Latent code z shape :", z1.shape,z2.shape)


#%%viz
img = torch.permute(denormalize_tensor(x[0]), [1,2,0])
ref1 = torch.permute(denormalize_tensor(x_ref1[0]),[1,2,0])
ref2 = torch.permute(denormalize_tensor(x_ref2[0]),[1,2,0])
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(ref1)
plt.subplot(133)
plt.imshow(ref2)
plt.show()

#%% test with network
from architecture import Generator

img_size=256
style_dim=64
latent_dim=16

generator=Generator(img_size, style_dim)









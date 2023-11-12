# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:24:28 2023

@author: balth
"""

import matplotlib.pyplot as plt

#eviter probleme avec matplotlib et torch
# plt.plot()
# plt.show()

import torch
from torchvision import transforms
from Dataloader import StarDataset, get_loader
#%% check dataset

root = "../dataset/data/celeba_hq/train"

train_dataset = StarDataset(root)


img, label, z1, z2, y = train_dataset[0]

crop = transforms.RandomResizedCrop(256, scale=(0.8,1), ratio=(0.9,1.1))

img_c = crop(img)

plt.imshow(torch.permute(img,[1,2,0]))
plt.title(f"image of domain : {label}")
plt.show()

plt.imshow(torch.permute(img_c,[1,2,0]))
plt.title(f"image of domain : {label}")
plt.show()

print(z1.shape)
print(z2.shape)
print(y)

#%% check dataloader


train_loader=get_loader(root,batch_size=8,img_size=256)


loader_iter = iter(train_loader)
imgs, labels, z1s, z2s, ys = next(loader_iter)









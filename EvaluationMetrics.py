# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:24:41 2023

The evaluation metrics used in StarGANv2 paper use the LPIPS and FID metrics.
Those metrics use pretrained models to measure quality and diversity in the generated images.

FID : measure the similarity of distribution from real world images and generated images
using pretrained inception model. the real and generated imagea are fed into the 
model and the feature map are used to measure distribution accross samples adn then compare those distributions 
using the frechet distance. 

@author: balth
"""

from scipy.linalg import sqrtm
import numpy as np

from torchvision import models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def fid(mu_real, mu_fake, cov_real, cov_fake):
    
    crcf,_ = sqrtm(np.dot(cov_real,cov_fake),disp=False)
    
    d = np.sum(mu_real**2 - mu_fake**2) + np.trace(cov_real+cov_fake-2*crcf)
    
    return d


def compute_mu_cov() : 
    pass


def frechet_distance(mu1, cov1, mu2, cov2):
    """
    Calculate the Fréchet distance between two multivariate Gaussian distributions.

    Parameters:
    mu1, mu2 : array-like
        Mean vectors of the two distributions.
    cov1, cov2 : array-like
        Covariance matrices of the two distributions.

    Returns:
    float
        The Fréchet distance.
    """
    # Compute the square root of the product of the covariance matrices
    cc, _ = sqrtm(np.dot(cov1, cov2), disp=False)

    # Ensure the result is a real number (to avoid complex numbers in sqrt)
    cc = np.real(cc)

    # Calculate the squared Euclidean distance between the means
    mean_diff = np.sum((mu1 - mu2)**2)

    # Compute the trace of the sum of the covariance matrices, adjusted for their similarity
    trace_term = np.trace(cov1 + cov2 - 2 * cc)

    # Combine the two components
    dist = mean_diff + trace_term

    return dist

class IncepV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights="Inception_V3_Weights.DEFAULT")
        self.layers = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.layers(x)
        return x.view(x.size(0), -1)


def get_eval_loader():
    pass
from tqdm import tqdm
def calculateFID(paths, img_size=256, batch_size=50):
    path_real, path_fake = paths

    print('Real Path %s \t path_fake %s...' % (path_real, path_fake))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = IncepV3().eval().to(device)
    #loader_real = get_eval_loader(path_real, img_size, batch_size)
    #loader_fake = get_eval_loader(path_fake, img_size, batch_size) 

    mu, cov = {"real": None ,"fake" : None}, {"real" :None , "fake" : None }
    loaders = {"real":get_loader(path_real, img_size, batch_size),"fake" :get_loader(path_fake, img_size, batch_size)}
    for key in loaders:
        actvs = []
        print(loaders[key])
        for x in tqdm(loaders[key],total=len(loaders[key])) : 
            print(inception(x.to(device)))
            actvs.append(inception(x.to(device))) 
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu[key]= np.mean(actvs, axis=0)
        cov[key] = np.cov(actvs, rowvar=False)
    fid_value = frechet_distance(mu["real"], cov["real"], mu["fake"], cov["fake"])
    return fid_value


# Create an instance of the IncepV3 class
model = IncepV3()

# Create a random tensor to simulate an input image
# Inception v3 expects a 3x299x299 input tensor
input_tensor = torch.randn(1, 3, 299, 299)

# Test the model with the input tensor
output1 = model(input_tensor)
print(output1[:10])





import matplotlib.pyplot as plt

#eviter probleme avec matplotlib et torch
#plt.plot()
#plt.show()

import torch
from torchvision import transforms
from dataloader.Dataloader import StarDataset, get_loader

#%%
def denormalize_tensor(t):
    #convert from -1,1 to 0,1
    return (t+1)/2
#%% check dataset

root = "train"
fakeroot = "fake"
train_dataset = StarDataset(root, chunk="train")


#inputs = train_dataset[0]
"""
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
"""
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
"""
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(ref1)
plt.subplot(133)
plt.imshow(ref2)
plt.show()
"""
#%% test with network
#from architecture import Generator

#img_size=256
#style_dim=64
#latent_dim=16

#generator=Generator(img_size, style_dim)

i= 0
for x in (train_dataset) : 
    print(type(x))
    i+=1
    print(i)
print(calculateFID([root,root], img_size=256, batch_size=8))
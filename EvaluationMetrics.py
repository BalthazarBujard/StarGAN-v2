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
    
    return np.real(d)


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
    loaders = {"real":get_loader(path_real, img_size, batch_size, chunk="eval"),"fake" :get_loader(path_fake, img_size, batch_size, chunk="eval")}
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

def claclulateFID_fromLoaders(loaders):
    #real_loader, fake_loader=loaders
    device = torch.device("cpu")#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = IncepV3().eval().to(device)
    
    #loaders = {"real":real_loader,"fake" : fake_loader}
    mu, cov = {"real": None ,"fake" : None}, {"real" :None , "fake" : None }
    for loader, key in zip(loaders, ["real", "fake"]):
        actv=[]
        for imgs, _ in loader:
            # print(imgs.shape)
            actv.append(inception(imgs.to(device)))
        actv = torch.cat(actv, dim=0).cpu().detach().numpy()
        mu[key]=np.mean(actv,axis=0)
        cov[key]=np.cov(actv,rowvar=False)
        # print(cov)
    d=fid(mu["real"], mu["fake"], cov["real"], cov["fake"])
    return d
    
#%%    
    
# Create an instance of the IncepV3 class
model = IncepV3()

# Create a random tensor to simulate an input image
# Inception v3 expects a 3x299x299 input tensor
input_tensor = torch.randn(1, 3, 299, 299)

# Test the model with the input tensor
output1 = model(input_tensor)
print(output1[:10])



#%%

import matplotlib.pyplot as plt

#eviter probleme avec matplotlib et torch
plt.plot()
plt.show()

import torch
from torchvision import transforms
from dataloader.Dataloader import get_loader, Fetcher

#%%
def denormalize_tensor(t):
    #convert from -1,1 to 0,1
    return (t+1)/2
#%% check dataset
import os
root = "../dataset/data/celeba_hq"
train_root=os.path.join(root, "train")
val_root=os.path.join(root, "val")

train_loader = get_loader(train_root, 8, 256, chunk="test") #chunk = test to not apply transform
train_fetcher = Fetcher(train_loader)

test_loader = get_loader(val_root, 8, 256, chunk="test")
test_fetcher = Fetcher(test_loader)


# i=1
# while i<len(train_loader):
#     train_inputs = next(train_fetcher) #trop long a aller jusqu'au bout
#     #test_inputs=next(test_fetcher)
#     print("batch:",i,"/",len(train_loader))
#     i+=1
i=1
for inputs in test_loader:
    print(i)
    i+=1




#%% test with network
from architecture.Generator import Generator

#img_size=256
#style_dim=64
#latent_dim=16

#generator=Generator(img_size, style_dim)
use_gpu = True if torch.cuda.is_available() else False
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                        'PGAN', model_name='celebAHQ-256',
                        pretrained=True, useGPU=use_gpu)
#%% generate fake images and save them
from torchvision.utils import save_image

num_images = 32
noise, _ = model.buildNoiseData(num_images)
with torch.no_grad():
    generated_images = model.test(noise)

# let's plot these images using torchvision and matplotlib
# import matplotlib.pyplot as plt
# import torchvision
# grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
# plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
# plt.show()
i=1
for img in generated_images:
    save_image(img,fp=f"../fake_imgs/fake/fake_img{i}.jpg")
    i+=1
#%%
from torchvision.datasets import ImageFolder

fake_path="../fake_imgs"
real_path="../real_imgs"

size=256
transform = transforms.Compose([
    transforms.Resize([size, size]),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

fake_dataset=ImageFolder(fake_path,transform=transform)
fake_loader=torch.utils.data.DataLoader(fake_dataset, batch_size=8)

real_dataset=ImageFolder(real_path, transform = transform)
real_loader=torch.utils.data.DataLoader(real_dataset, batch_size=8)

# for imgs, _ in fake_loader:
#     print(imgs.shape)

#%%

fid=claclulateFID_fromLoaders([real_loader, fake_loader])











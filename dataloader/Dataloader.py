# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:19:57 2023

script for creating the different datasets useful for the StarGANv2 


@author: balth
"""

import os
from PIL import Image
import numpy as np
from munch import Munch
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
#ste seed for reproductibility 
#torch.manual_seed(123)



class StarDataset(Dataset):
    def __init__(self, root, size=256, latent_dim = 16, transform=None, chunk="train"):
                
        self.img_paths, self.labels = self.create_dataset(root, chunk)
        
        self.size=size
        self.latent_dim = latent_dim
        self.chunk=chunk
        
        if transform is not None:
            self.transform = transform
        
        else :
            #default transform should : resize and normalize images [-1,1]
            self.transform = transforms.Compose([
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        
        
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        #function that returns an image and its corresponding domain (label)
        #should also return a pair of latent codes z1 and z2 of latent_dim dimension
        #and a pair of reference images -> during training, either the latent codes or references are used
        #to compute the style (from style encoder or mapping network)
        #and its target domain (also chosen at random)
        
        img_path, label = self.img_paths[index], self.labels[index]
        
        img = Image.open(img_path).convert("RGB") #convert to RGB for model input
        
        
        
        #get 2 ref images and 2 latent codes for training
        if self.chunk == "train":
            img_ref1, img_ref2 = np.random.choice(self.img_paths,size=2,replace=False)
            img_ref1 = Image.open(img_ref1).convert("RGB")
            img_ref2 = Image.open(img_ref2).convert("RGB")
            
            #generate 2 random latent codes from normal distribution and a random domain
            z1 = torch.randn(self.latent_dim)
            z2 = torch.randn(self.latent_dim)
            y_trg = torch.randint(0, len(self.domains), ())
            
            #should at least have resize and normalize (and toTensor obviously)
            if self.transform is not None:
                img = self.transform(img)
                img_ref1 = self.transform(img_ref1)
                img_ref2 = self.transform(img_ref2)
            
            inputs = Munch(x = img, y=label, z1 = z1,
                           z2=z2, x_ref1 = img_ref1, x_ref2 = img_ref2,
                           y_trg = y_trg)
        
        
      
        elif self.chunk=="test":
            if self.transform is not None:
                img = self.transform(img)
            
            inputs = Munch(x=img, y = label)
        
        else :
            raise Exception("Invalid chunk")
        
                   
        
        return inputs
    
    def create_dataset(self, root, chunk):
        #extract imgs_paths and domains (labels) from the root directory
        #containing all imgs and domain folders
        
        domains = os.listdir(root)
        
        self.domains = range(len(domains))
        
        labels = [] #labels list
        img_paths=[] #img paths list
        
        #for every domain folder
        for i,domain in enumerate(domains):
            label = i #ith folder corresponds to ith label/domain
            path = os.path.join(root,domain)
            #for every image in a domain folder
            for fname in os.listdir(path):
                labels.append(label)
                img_paths.append(os.path.join(path,fname))
        
        assert len(img_paths)==len(labels), f"{len(img_paths)} imgs paths != {len(labels)} labels"
        
        return img_paths, labels
                




#function to create a weighted sampler for a dataset given the labels list
#protected function to be called only in this file
def _balanced_sampler(labels):
    
    counts = np.bincount(labels) #returns a list of counts per unique value in labels
    
    #weights are the inverse of the count(distribution)
    #if there are more examples of a domain (label) their probability of being sampled are lower
    weights = 1/counts 
    
    #for every sample (label) we assign a weight inverse of its distribution
    weight_per_sample = weights[labels]
    
    return WeightedRandomSampler(weight_per_sample, len(weight_per_sample))


#function to create a dataset, fromthat dataset instantiate a dataloader and return it
def get_loader(root, batch_size, img_size, chunk = "train"):
    """

    Parameters
    ----------
    root : str
        root directory of the dataset 
    batch_size : int
        size of the dataloader batch
    img_size : int
        size of the output images
    chunk : str, optional
        train, val or evaluation set. The default is "train".

    Returns
    -------
    loader : DataLoader

    """
    
    if chunk == "train":
        #data augmentation : randomCrop, horizontalFlip and random rotation (small +-10°)
        #randomResizedCrop -> scale : portion of the image to make the crop (80-100%)
        # -> ratio : aspect ratio boundaries of the crop (0.9-1.1) : mimics light deformation of face
        #small rotation and random horizintal flip
        #resize usefull?
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.8,1), ratio=(0.9,1.1),antialias=True),
            transforms.Resize([img_size,img_size],antialias=True),
            #transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
            ])
    
    elif chunk == "test" : transform = None #with transform as none we apply default transforms
    
    elif chunk == "eval":
        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        
        dataset = ImageFolder(root,transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        return loader
    
    else :
        raise Exception(f"Invalid chunk : {chunk}. Valid chunks are train, test or eval")
    
    #create dataset
    dataset = StarDataset(root, size = img_size, transform = transform, chunk=chunk)
    
    #create sampler
    sampler = _balanced_sampler(dataset.labels)
    
    #create dtaloader
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    return loader
    
    
#class to fetch inputs, mainly to handle end of dataloader
#dataloader already handles cases of train and test so no need to do it here
class Fetcher:
    def __init__(self,loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _fetch_inputs(self):
        #method to fetch next set of inputs
        try:
            #try to fectch next inputs
            inputs = next(self.iter_loader)
        except (AttributeError, StopIteration):
            #if self.iter_loader not already instantiated or end of loader
            self.iter_loader = iter(self.loader)
            inputs = next(self.iter_loader)
        
        return inputs
    
    def __next__(self):
        inputs = self._fetch_inputs()
        
        #pass inputs to cuda
        return Munch({key : item.to(self.device) for key, item in inputs.items()})
      
            
            







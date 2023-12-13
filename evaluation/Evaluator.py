import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from munch import Munch
import os
import json
import shutil #igh level directory management
import lpips
from .EvaluationMetrics import * 
from torchvision.utils import save_image
from dataloader.Dataloader import get_loader, Fetcher

lpips_metric = lpips.LPIPS(net="alex").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) #LPIPS module with alexNet pretrained model


#ADD TQMD TO SHOW PROGRESION (it is kinda slow...)
#CLEAN CODE (sub methods etc)
class Evaluator:
    """
    Evlautor object used to compute metrics for StarGANv2
    LPIPS : computes lpips for every task (src2trg domain) and stores every output to params.save_dir to be used for FID
    FID : to be added
    """
    def __init__(self, eval_params, nets):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = nets
        self.img_size = eval_params.img_size
        self.val_dir = eval_params.val_dir
        self.save_dir = eval_params.save_dir
        self.train_dir = eval_params.train_dir
        self.val_batch_size = eval_params.batch_size
        self.metrics=Munch(lpips=[], fid=[])

        self.params = eval_params

    @torch.no_grad()
    def evaluate_lpips(self, mode):
        assert mode in ["reference", "latent"], "evaluation must be either latent or reference guided"
        
        #for every domain, evaluate lpips from src domain to trg domain for every pair of domains
        domains = [d for d in os.listdir(self.val_dir) if ".ipynb" not in d] #handle bad folder from ipynb notebooks...
        params = self.params
        #dict conatining the mean values for every task (task is src_domain2trg_domain generation/synthesis)
        #also contains the mean value for a given mode (latent or ref)
        lpips_dict={} 
        
        #trg domains
        for idx, domain in enumerate(domains):
            src_domains = [src_domain for src_domain in domains if src_domain != domain]

            if mode == "reference":
                #build reference dataset
                ref_path = os.path.join(self.val_dir,domain)
                ref_loader = get_loader(root = ref_path, batch_size=params.batch_size, img_size=params.img_size, chunk="eval")
                ref_fetcher = Fetcher(ref_loader, chunk="eval")
                
            #for every pair of trg/src domains
            lpips_values=[] #list of mean lpips values for every pair of src and trg domain
            f_inc = 0 #file name is given by its task and iter number
            for src_domain in src_domains:

                print(f"{src_domain} to {domain} generation")
                task = f"{src_domain}2{domain}"
                folder = os.path.join(self.save_dir,mode)
                path_fake = os.path.join(folder, task)
                #create directory to store task specific outputs (used for FID)
                #delete pre-existing folder
                shutil.rmtree(path_fake, ignore_errors=True) #removes previous folder and its content
                #create folder
                os.makedirs(path_fake) #makedirs creates the intermediate folders (mkdir dpesnt)
                
                src_path = os.path.join(self.val_dir,src_domain)
                src_loader = get_loader(root = src_path, batch_size=params.batch_size, img_size=params.img_size, chunk="eval")
                src_fetcher = Fetcher(src_loader, chunk="eval")
                
                outputs=[] #list of outputs -> used to compute lpips for every pair of images -> want high mean lpips
                for i in range(len(src_loader)):
                    x = next(src_fetcher) #source image
                    n = len(x) #or eval_params.batch_size
                    #trg domain to translate src images
                    y_trg = torch.tensor([idx]*n).to(self.device)
                    #get face landmarks heatmap
                    masks = self.nets.fan.get_heatmap(x) if params.wFilter>0 else None

                    #for every input form source domain, generate 10 (batched) outputs using either the reference guided or latent guided synthesis
                    for j in range(params.num_outputs):
                        if mode=="reference":
                            x_ref = next(ref_fetcher)
                            s_trg = self.nets.style_encoder(x_ref, y_trg)
                        
                        else : #latent
                            z_trg = torch.randn(n, params.latent_dim).to(self.device)
                            s_trg = self.nets.mapping_network(z_trg,y_trg)


                        #generate ouptuts
                        x_fake = self.nets.generator(x, s_trg, masks)
                        outputs.append(x_fake)

                        #save outputs for FID metric
                        for img in x_fake:
                            fname=os.path.join(path_fake,f"{task}_{f_inc}.png")
                            img=denormalize(img)
                            save_image(img.cpu(),fname)
                            f_inc+=1
                    
                            
                #compute lpips value for every pair of of generated outputs for that input
                lpips_all = []
                for i in range(len(outputs)):
                    for j in range(i+1,len(outputs)):
                        lpips_ = lpips_metric(outputs[i],outputs[j])
                        lpips_all.append(lpips_)
                        #print(lpips_.shape)
                lpips_mean = torch.mean(torch.stack(lpips_all,dim=0)) #compute mean lpips value for this task
                lpips_dict[f"{src_domain}2{domain}"]=lpips_mean.cpu().detach().numpy() #add value to lpips dictionnary
                    
                #delete loaders
                del src_loader
                del src_fetcher
                if mode=="reference":
                    del ref_loader
                    del ref_fetcher
                    
        #compute mean lpips for every task
        mean = 0
        for key, item in lpips_dict.items():
            mean+= item
        lpips_dict[f"LPIPS_{mode}/mean"]=mean/len(lpips_dict) #mean lpips across all tasks

        return lpips_dict
                

    def evaluate_fid(self, mode,step) : 
        #for every domain, evaluate lpips from src domain to trg domain for every pair of domains
        domains = [d for d in os.listdir(self.val_dir) if ".ipynb" not in d] #handle bad folder from ipynb notebooks...
        params = self.params
        #dict conatining the mean values for every task (task is src_domain2trg_domain generation/synthesis)
        #also contains the mean value for a given mode (latent or ref)
        
        fid_values = {}
        #trg domains
        for domain in domains :
            src_domains = [src_domain for src_domain in domains if src_domain != domain]
            for src_domain in src_domains:
                print(f"{src_domain} to {domain} generation")
                task = f"{src_domain}2{domain}"
                path_fake = os.path.join(self.save_dir, task)
                path_real = os.path.join(self.train_dir, domain)
                fid_value = calculateFID(
                    paths=[path_real, path_fake],
                    img_size=self.img_size,
                    batch_size=self.val_batch_size)
                fid_values[f'FID_{mode}/{task}'] = fid_value
        fid_mean = 0 
        for _ , item in fid_values.items():
            fid_mean+= item / len(fid_values)
        fid_values[f"FID_{mode}/mean"] = fid_mean

        filename = os.path.join(self.save_dir, f'FID_{step}_{mode}.json')
        with open(filename, "w") as outfile:
            json.dump(fid_values, outfile)


def denormalize(x):
    #returns a tensor in [0,1] range no matter the input (generator output approx -1,1)
    out = (x-x.min())/(x.max()-x.min())
    return out

















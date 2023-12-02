import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from munch import Munch
import os
from train.check_point_handler import *  
from train.loss import loss_discriminator, loss_generator#, loss_discriminator
#from train.loss_cheat import loss_generator #,loss_generator #for debugging
from architecture.Model import *
from dataloader.Dataloader import Fetcher
import time
import datetime
import sys
from IPython.display import clear_output #for display
from torchvision.utils import make_grid #for plot

def moving_average(model, model_copy, beta=0.999):
    for param, param_test in zip(model.parameters(), model_copy.parameters()):
        # Does a linear interpolation of two tensors start (given by input) and end based on a scalar or tensor weight and returns the resulting out tensor.
        param_test.data = torch.lerp(param.data, param_test.data, beta) 


#proceedes to He inititlaization of all modules
def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class Trainer(nn.Module) : 
    def __init__(self, params):
        #what is in params? -> see train_test.py
        super().__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.networks, self.networks_copy = Model(params)
        self.optimizers = Munch()

        for key_network in ['generator', 'mapping_network', 'style_encoder', 'discriminator'] : 
            setattr(self, key_network, self.networks[key_network])
            setattr(self, key_network + '_copy', self.networks_copy[key_network])

        if params.mode == 'train':
            for key_network in ['generator', 'mapping_network', 'style_encoder', 'discriminator'] : 
                self.optimizers[key_network] = torch.optim.Adam(
                    params=self.networks[key_network].parameters(),
                    lr=params.f_lr if key_network == 'mapping_network' else params.lr,
                    betas=[params.beta1, params.beta2],
                    weight_decay=params.weight_decay)
                
            self.checkpoints = [
                ModelCheckpointHandler(os.path.join(params.checkpoint_dir, '{:06d}_networs.cpt'),  **self.networks),
                ModelCheckpointHandler(os.path.join(params.checkpoint_dir, '{:06d}__networs_copy.cpt'), **self.networks_copy),
                ModelCheckpointHandler(os.path.join(params.checkpoint_dir, '{:06d}_optims.cpt'), **self.optimizers)]

        else : 
            self.checkpoints = [ModelCheckpointHandler(os.path.join(params.checkpoint_dir, '{:06d}__networs_copy.cpt'), **self.netwroks_copy)]

        
        self.to(self.device)
        
        #init weights of main module (aka not copy nor FAN)
        for name, network in self.named_children():
            if ('copy' not in name) and ('fan' not in name):
               print('Initializing %s...' % name)
               network.apply(he_init)
        
    def _reset_grad(self):
        for optim in self.optimizers.values():
            optim.zero_grad()

    def _save_checkpoint(self, step):
        for cpt in self.checkpoints:
            cpt.store_checkpoint(step)

    def _load_checkpoint(self, step):
        for cpt in self.checkpoints:
            cpt.retrieve_checkpoint(step)

    def train(self,loaders):
        params=self.params
        nets=self.networks
        nets_copy=self.networks_copy
        optims=self.optimizers
        #loaders shouldhave a train, val and test/eval loader
        
        #get input fetcher
        train_loader = loaders.train 
        
        input_fetcher = Fetcher(train_loader)
        
        #get val and test/eval loader
            #not implemented yet
        
        if params.resume_iter>0:
            self._load_checkpoint(step=params.resume_iter)
        
        #retreive starting lr rate
        l_ds_init = params.lambda_ds
        
        print("Start training...")
        t0 = time.time()
        #add epochs viz
        losses = Munch(g_latent=[],g_ref=[],d_latent=[],d_ref=[])

        #number of epochs
        max_iter = params.epochs*len(train_loader)
        for epoch in range(params.epochs):
            #for every batch in dataloader
            for i in range(len(train_loader)): #ATTENTION POUR LES CONDITIONS DE SAVE, EVALUATE ET PLOT SI BOUCLE PAR EPOCH
        #for i in range(params.resume_iter,params.max_iter):
            
                inputs = next(input_fetcher)
                
                x_org,y_org = inputs.x, inputs.y
                z1, z2 = inputs.z1, inputs.z2
                x_ref1, x_ref2 = inputs.x_ref1, inputs.x_ref2
                y_trg = inputs.y_trg

                #landmark mask -> to be used if celeba_hq data; used in Generator ! (to be implemented)
                #masks = nets.fan.get_heatmap(x_org)
                
                #Train discriminator
                #with latent code
                d_loss, d_loss_latent = loss_discriminator(nets, x_org, y_org,
                                                           y_trg, z_trg=z1)
                
                #add loss to plot
                losses.d_latent.append(d_loss.cpu().detach().numpy())
                
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()
                
                #with reference image
                d_loss, d_loss_ref = loss_discriminator(nets, x_org, y_org,
                                                            y_trg, x_ref=x_ref1)
                losses.d_ref.append(d_loss.cpu().detach().numpy())
    
                
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()
                
                # Train generator
                g_loss, g_loss_latent = loss_generator(nets, x_org, y_org, y_trg,
                                                        z_trgs=[z1,z2],
                                                        lambda_ds=params.lambda_ds)
                
                losses.g_latent.append(g_loss.cpu().detach().numpy())
                
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                optims.mapping_network.step()
                optims.style_encoder.step()
                
                g_loss, g_loss_ref = loss_generator(nets, x_org, y_org, y_trg,
                                                        x_refs=[x_ref1,x_ref2],
                                                        lambda_ds=params.lambda_ds)
                losses.g_ref.append(g_loss.cpu().detach().numpy())
    
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                optims.mapping_network.step()
                optims.style_encoder.step()
                
                #moving average
                #apply moving average to network copy used for eval/test
                #only applied to generator modules (generator, mapp, encoder)
                moving_average(nets.generator, nets_copy.generator)
                moving_average(nets.mapping_network, nets_copy.mapping_network)
                moving_average(nets.style_encoder, nets_copy.style_encoder)
                
                #update lambda ds with linear decay
                if params.lambda_ds>0:
                    params.lambda_ds -= l_ds_init/max_iter#params.max_iter
                    
                #log output
                if (i+1)%params.log_iter==0:
                    t=time.time()-t0
                    t=str(datetime.timedelta(seconds=t))#[:-7]
                    
                    log = f"Time elapsed : {t}\nEpoch : {epoch}/{params.epochs}, Batch {i+1}/{len(train_loader)}\n"#{params.max_iter}"
                    print(log)
                    
                    #losses
                    all_losses = dict()
                    for loss, prefix in zip([d_loss_latent, d_loss_ref, g_loss_latent, g_loss_ref],
                                            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                        for key, value in loss.items():
                            #all_losses['D/latent_adv,....]
                            all_losses[prefix + key] = value
                    all_losses['G/lambda_ds'] = params.lambda_ds
                    log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
    
                    clear_output(wait=True)
                    
                    
                    #plot losses
                    plt.plot(losses.g_latent,label="Generator latent loss")
                    plt.plot(losses.g_ref,label="Generator ref loss")
                    plt.plot(losses.d_latent, label="Discriminator latent loss")
                    plt.plot(losses.d_ref, label="Discriminator ref loss")
                    #plt.ylim(-5,10)
                    plt.legend()
                    plt.show()
    
                    print(log, end="\r")
    
    
                #show example during training on val dataset 
                
                #save model
                if (i+1)%params.save_iter==0:
                    self._save_checkpoint(step=i+1)
                
                #evaluation metrics
                if (i+1)%params.eval_iter==0:
                    #not implemented yet
                    #use in-training generator to generate a set of images to compute metrics FID (difference of distribution from real/fake imgs set)
                    #and LPIPS (measure perceived quality of generated images)
                    #use val folder to generate images
                    pass
            
                
            #show example at the end of every epoch
            generator = self.networks.generator
            mn = self.networks.mapping_network
            
            inputs = next(input_fetcher)
            x_org,y_org = inputs.x, inputs.y
            z1, z2 = inputs.z1, inputs.z2
            x_ref1, x_ref2 = inputs.x_ref1, inputs.x_ref2
            y_trg = inputs.y_trg
            
            style = mn(z1,y_trg)
            input_img=x_org
            
            x_fake=generator(input_img,style)
            
            x_n = [(x-x.min())/(x.max()-x.min()) for x in x_fake]
            
            grid = make_grid(x_n)
            
            imgs=torch.permute(grid, [1,2,0]).cpu().detach().numpy()
                        
            plt.figure(figsize=(10,5))
            plt.imshow(imgs)
            plt.show()
            
            
        


    @torch.no_grad()
    def sample(self, loaders):
        pass
    

    
    
    @torch.no_grad()
    def evaluate(self):
        params = self.params
        networks_copy = self.networks_copy
        resume_iter = params.resume_iter
        self._load_checkpoint(params.resume_iter)
        calculate_metrics(networks_copy, params, step=resume_iter, mode='latent') #calculate metrics from latent vector
        calculate_metrics(networks_copy, params, step=resume_iter, mode='reference') #claculate metrics from refernece image


def calculate_metrics (networks_copy, params, step, mode):

    val_folder = params.val_folder

    domains = os.listdir(val_folder)

    print(f"there are {len(domains)} domains")

    #generate images 
    for trg_domain in domains:
        #the source domain has to be different form the target domain
        src_domain = [domain for domain in domains if domain!=trg_domain]
        #we want to generate images from the source domain using the target domain as input
    
    raise NotImplementedError("calculate_metrics NOT IMPLEMENTED YET !")




# -----------------------------------------------------------test loss---------------------------------------------------

# sys.path.append('architecture')
# from architecture.Generator import *
# from architecture.Mapping_Network import *
# from architecture.Style_Encoder import *
# from architecture.Discriminator import *
# from architecture.Model import *
# from torch.autograd import Variable


# # Initialize parameters
# img_size = 256      # Image size
# style_dim = 64      # Dimension of style representation
# latent_dim = 16     # Dimension of latent space
# num_domains = 2     # Number of domains for style transfer

# # Instantiate models
# generator = Generator(img_size, style_dim)  # Create generator model
# discriminator = Discriminator(num_domains)  # Create discriminator model
# style_encoder = StyleEncoder(style_dim, num_domains)  # Create style encoder model
# mapping_network = MappingNetwork(latent_dim, style_dim, num_domains)  # Create mapping network model

# # Create test data
# real_img = torch.randn(1, 3, img_size, img_size)  # Real image tensor
# latent_code = torch.randn(1, latent_dim)         # Latent code tensor
# latent_code2 = torch.randn(1, latent_dim)        # Another latent code tensor
# domain_idx = torch.randint(0, num_domains, (1,)) # Domain index tensor

# # Create a collection of networks
# nets = Munch(generator=generator, discriminator=discriminator,
#              style_encoder=style_encoder, mapping_network=mapping_network)

# branch_idx = torch.tensor([0])  # Branch index for style encoding
# """
# # Calculate generator loss
# g_total_loss, g_loss_munch = loss_generator(
#     nets, real_img, domain_idx, domain_idx,
#     z_trgs=(latent_code, latent_code2), branch_idx=branch_idx
# )
# print(f"Generator Total Loss: {g_total_loss.item()}")
# print(f"Generator Loss Components: Adv: {g_loss_munch.adv}, Sty: {g_loss_munch.sty}, DS: {g_loss_munch.ds}, Cyc: {g_loss_munch.cyc}")

# # Calculate discriminator loss
# d_total_loss, d_loss_munch = loss_discriminator(
#     nets, x_real=real_img, y_org=domain_idx, y_trg=domain_idx,
#     z_trg=latent_code, x_ref=None, lambda_reg=1.0
# )
# print(f"Discriminator Total Loss: {d_total_loss.item()}")
# print(f"Discriminator Loss Components: Real: {d_loss_munch.real}, Fake: {d_loss_munch.fake}, Reg: {d_loss_munch.reg}")

# """

# -----------------------------------------------------------test moving average ---------------------------------------------------
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
# parser.add_argument('--var', type=int, default=64,
#                         help='dummy var for testing')
# parser.add_argument('--eval_iter', type=int, default=50,
#                         help='evaluate every ...')
# parser.add_argument('--save_iter', type=int, default=50,
#                         help='Save model iteration')
# parser.add_argument('--lambda_ds', type=float, default=1.,
#                         help='Diversification style loss coefficient')
# parser.add_argument('--resume_iter', type=int, default=0,
#                         help='Start iteration')
# parser.add_argument('--max_iter', type=int, default=200,
#                         help='Style code dimension')
# params = parser.parse_args()

# #Model,CopyModel = Model(params)

# trainer = Trainer(params)

#moving_average(Model.generator, CopyModel.generator, beta=0.999)
#print(Model.keys())
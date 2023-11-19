import torch
import torch.nn.functional as F
import torch.nn as nn
from munch import Munch
import os
from check_point_handler import *  
from architecture.Model import *
from dataloader.Dataloader import Fetcher
import sys

#  Computes adversarial loss for discriminator.
def adversarial_loss(discriminator, real_img, fake_img, real_label, fake_label, y_org=None, y_trg=None):
    real_loss = torch.tensor(0.0)
    fake_loss = torch.tensor(0.0)
    if real_img is not None and y_org is not None:
        real_loss = F.binary_cross_entropy_with_logits(discriminator(real_img, y_org), real_label)
    if fake_img is not None and y_trg is not None:
        fake_loss = F.binary_cross_entropy_with_logits(discriminator(fake_img, y_trg), fake_label)
    return real_loss + fake_loss



# Calculates style reconstruction loss for style encoder.
def style_reconstruction_loss(style_encoder, generated_img, target_style, branch):
    # Predicts style from the generated image using the specified branch (domain from y_trg)
    predicted_style = style_encoder(generated_img, branch)
    # Calculates the mean absolute error between predicted and target styles
    return torch.mean(torch.abs(predicted_style - target_style))


# Computes style diversification loss for generator.
def style_diversification_loss(generator, input_img, style1, style2):
    # Generates two images from the same input image but with different styles
    generated_img1 = generator(input_img, style1)
    generated_img2 = generator(input_img, style2)
    # Calculates the mean absolute error between the two generated images
    return torch.mean(torch.abs(generated_img1 - generated_img2))

# Measures cycle consistency loss for generator.
def cycle_consistency_loss(generator, input_img, reconstructed_img):
    # Calculates the mean absolute error between the input and reconstructed images
    return torch.mean(torch.abs(reconstructed_img - input_img))


# Generator Loss Function
def loss_generator(nets, x_real, y_org, y_trg, z_trgs=None, x_refs=None, lambda_sty=1.0, lambda_ds=1.0, lambda_cyc=1.0):
    """
    Calculates the total loss for the generator.

    Args:
    nets (Munch): Contains generator, discriminator, style_encoder, and mapping_network.
    x_real (Tensor): Real images tensor.
    y_org (Tensor): Original domain indices.
    y_trg (Tensor): Target domain indices.
    z_trgs (Tuple[Tensor, Tensor], optional): Pair of latent codes for target styles.
    x_refs (Tuple[Tensor, Tensor], optional): Pair of reference images for target styles.
    branch_idx (int): Index of the style branch for encoding.
    lambda_sty, lambda_ds, lambda_cyc (float): Weights for style, diversification, and cycle losses.

    Returns:
    Tuple[Tensor, Munch]: Total generator loss and loss components as a Munch object.
    """
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
        s_trg = nets.mapping_network(z_trg, y_trg)
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    elif x_refs is not None:
        x_ref, x_ref2 = x_refs
        s_trg = nets.style_encoder(x_ref, y_trg)
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    else:
        raise ValueError("Either z_trgs or x_refs must be provided.")

    # Generate fake images
    x_fake = nets.generator(x_real, s_trg)
    x_fake2 = nets.generator(x_real, s_trg2)

    # Compute losses
    out_fake = nets.discriminator(x_fake, y_trg)
    #TO DO : dont create out_fake outside as its not used, for ones_like, use the batch size of x_real/fake
    loss_adv = adversarial_loss(nets.discriminator, x_real, x_fake, torch.ones_like(out_fake), torch.zeros_like(out_fake), y_org, y_trg)
    loss_sty = style_reconstruction_loss(nets.style_encoder, x_fake, s_trg, y_trg)
    loss_ds = style_diversification_loss(nets.generator, x_real, s_trg, s_trg2)

    # Compute cycle consistency loss (use x_real for reconstruction)
    x_rec = nets.generator(x_fake, nets.style_encoder(x_real, y_org))

    loss_cyc = cycle_consistency_loss(nets.generator, x_real, x_rec)

    # Combined Loss
    total_loss = loss_adv + lambda_sty * loss_sty - lambda_ds * loss_ds + lambda_cyc * loss_cyc
    return total_loss, Munch(adv=loss_adv.item(), sty=loss_sty.item(), ds=loss_ds.item(), cyc=loss_cyc.item())


# Discriminator Loss Function
def loss_discriminator(nets, x_real, y_org, y_trg, z_trg=None, x_ref=None, lambda_reg=1.0):
    """
    Calculates the total loss for the discriminator.

    Args:
    nets (Munch): Contains generator, discriminator, style_encoder, and mapping_network.
    x_real (Tensor): Real images tensor.
    y_org (Tensor): Original domain indices.
    y_trg (Tensor): Target domain indices.
    z_trg (Tensor, optional): Latent code for target style.
    x_ref (Tensor, optional): Reference image for target style.
    lambda_reg (float): Weight for regularization loss.

    Returns:
    Tuple[Tensor, Munch]: Total discriminator loss and loss components as a Munch object.
    """
    # Compute real loss
    x_real.requires_grad_()
    out_real = nets.discriminator(x_real, y_org)
    loss_real = adversarial_loss(nets.discriminator, x_real, None, torch.ones_like(out_real), None, y_org)

    # Compute fake loss
    if z_trg is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    elif x_ref is not None:
        s_trg = nets.style_encoder(x_ref, y_trg)
    else:
        raise ValueError("Either z_trg or x_ref must be provided.")
    x_fake = nets.generator(x_real, s_trg)
    out_fake = nets.discriminator(x_fake, y_trg)
    loss_fake = adversarial_loss(nets.discriminator, None, x_fake, None, torch.zeros_like(out_fake), y_trg)



    # Regularization term
    loss_reg = r1_reg(out_real, x_real)

    # Combined Loss
    total_loss = loss_real + loss_fake + lambda_reg * loss_reg
    return total_loss, Munch(real=loss_real.item(), fake=loss_fake, reg=loss_reg)


def r1_reg(out_real, x_real):
    """
    Computes the R1 regularization penalty.

    Args:
    out_real (Tensor): Output of the discriminator for real images.
    x_real (Tensor): Real images tensor.

    Returns:
    Tensor: R1 regularization penalty.
    """
    grad_real = torch.autograd.grad(outputs=out_real.sum(), inputs=x_real, create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    return grad_penalty



def moving_average(model, model_copy, beta=0.999):
    for param, param_test in zip(model.parameters(), model_copy.parameters()):
        # Does a linear interpolation of two tensors start (given by input) and end based on a scalar or tensor weight and returns the resulting out tensor.
        param_test.data = torch.lerp(param.data, param_test.data, beta) 


class Trainer : 
    def __init__(self, params):
        #what is in params?
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.networks, self.networks_copy = Model(params)
        self.optimizers = Munch()

        for key_network in ['generator', 'mapping_network', 'style_encoder', 'discriminator'] : 
            setattr(self, key_network, self.netwroks[key_network])
            setattr(self, key_network + '_ema', self.netwroks_copy[key_network])

        if params.mode == 'train':
            for key_network in ['generator', 'mapping_network', 'style_encoder', 'discriminator'] : 
                self.optimizers[key_network] = torch.optim.Adam(
                    params=self.networks[key_network].parameters(),
                    lr=params.f_lr if key_network == 'mapping_network' else params.lr,
                    betas=[params.beta1, params.beta2],
                    weight_decay=params.weight_decay)
                
            self.checkpoints = [
                ModelCheckpointHandler(os.path.join(params.checkpoint_dir, '{:06d}_networs.cpt'), data_parallel=True, **self.netwroks),
                ModelCheckpointHandler(os.path.join(params.checkpoint_dir, '{:06d}__networs_copy.cpt'), data_parallel=True, **self.netwroks_copy),
                ModelCheckpointHandler(os.path.join(params.checkpoint_dir, '{:06d}_optims.cpt'), **self.optimizers)]

        else : 
            self.checkpoints = [ModelCheckpointHandler(os.path.join(params.checkpoint_dir, '{:06d}__networs_copy.cpt'), data_parallel=True, **self.netwroks_copy)]

        self.to(self.device)
    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self, step):
        for cpt in self.checkpoints:
            cpt.save(step)

    def _load_checkpoint(self, step):
        for cpt in self.checkpoints:
            cpt.load(step)
        
    def train(self,loaders):
        #loaders shouldhave a train, val and eval loader
        
        #get input fetcher
        fetcher = Fetcher(loaders.train)
        
        #get val and eval loader
            #not implemented yet
            
        


    @torch.no_grad()
    def sample(self, loaders):
        pass
    

    
    
    @torch.no_grad()
    def evaluate(self):
        params = self.params
        netwroks_copy = self.netwroks_copy
        resume_iter = params.resume_iter
        self._load_checkpoint(params.resume_iter)
        calculate_metrics(netwroks_copy, params, step=resume_iter, mode='latent')
        calculate_metrics(netwroks_copy, params, step=resume_iter, mode='reference')


def calculate_metrics (netwroks_copy, params, step, mode):
    raise NotImplementedError("calculate_metrics NOT IMPLEMENTED YET !")




# -----------------------------------------------------------test loss---------------------------------------------------

sys.path.append('architecture')
from architecture.Generator import *
from architecture.Mapping_Network import *
from architecture.Style_Encoder import *
from architecture.Discriminator import *
from architecture.Model import *
from torch.autograd import Variable


# Initialize parameters
img_size = 256      # Image size
style_dim = 64      # Dimension of style representation
latent_dim = 16     # Dimension of latent space
num_domains = 2     # Number of domains for style transfer

# Instantiate models
generator = Generator(img_size, style_dim)  # Create generator model
discriminator = Discriminator(num_domains)  # Create discriminator model
style_encoder = StyleEncoder(style_dim, num_domains)  # Create style encoder model
mapping_network = MappingNetwork(latent_dim, style_dim, num_domains)  # Create mapping network model

# Create test data
real_img = torch.randn(1, 3, img_size, img_size)  # Real image tensor
latent_code = torch.randn(1, latent_dim)         # Latent code tensor
latent_code2 = torch.randn(1, latent_dim)        # Another latent code tensor
domain_idx = torch.randint(0, num_domains, (1,)) # Domain index tensor

# Create a collection of networks
nets = Munch(generator=generator, discriminator=discriminator,
             style_encoder=style_encoder, mapping_network=mapping_network)

branch_idx = torch.tensor([0])  # Branch index for style encoding
"""
# Calculate generator loss
g_total_loss, g_loss_munch = loss_generator(
    nets, real_img, domain_idx, domain_idx,
    z_trgs=(latent_code, latent_code2), branch_idx=branch_idx
)
print(f"Generator Total Loss: {g_total_loss.item()}")
print(f"Generator Loss Components: Adv: {g_loss_munch.adv}, Sty: {g_loss_munch.sty}, DS: {g_loss_munch.ds}, Cyc: {g_loss_munch.cyc}")

# Calculate discriminator loss
d_total_loss, d_loss_munch = loss_discriminator(
    nets, x_real=real_img, y_org=domain_idx, y_trg=domain_idx,
    z_trg=latent_code, x_ref=None, lambda_reg=1.0
)
print(f"Discriminator Total Loss: {d_total_loss.item()}")
print(f"Discriminator Loss Components: Real: {d_loss_munch.real}, Fake: {d_loss_munch.fake}, Reg: {d_loss_munch.reg}")

"""

# -----------------------------------------------------------test moving average ---------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
params = parser.parse_args()

Model,CopyModel = Model(params)

#moving_average(Model.generator, CopyModel.generator, beta=0.999)
print(Model.keys())
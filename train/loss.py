import torch
import torch.nn.functional as F
import torch.nn as nn
from munch import Munch


def adversarial_loss(logits, target):
    assert target in [1, 0]
    targets = target * torch.ones_like(logits)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def style_loss(predicted_style, target_style):
    # Calculate the style loss as the mean absolute error between the predicted and target styles
    return torch.mean(torch.abs(predicted_style - target_style))

def diversity_loss(fake_img1, fake_img2):
    # Calculate the diversity loss as the mean absolute error between two generated images
    return torch.mean(torch.abs(fake_img1 - fake_img2))

def cycle_loss(reconstructed_img, real_img):
    # Calculate the cycle-consistency loss as the mean absolute error between the input and reconstructed images
    return torch.mean(torch.abs(reconstructed_img - real_img))


def loss_generator(nets, x_real, y_org, y_trg, z_trgs=None, x_refs=None, lambda_sty=1.0, lambda_ds=1.0, lambda_cyc=1.0):
    """
    Calculates the total loss for the generator.

    Parameters:
    nets (Munch): Contains generator, discriminator, style_encoder, and mapping_network.
    x_real (Tensor): Real images tensor.
    y_org (Tensor): Original domain labels.
    y_trg (Tensor): Target domain labels.
    z_trgs (Tuple[Tensor, Tensor], optional): Pair of latent codes for target styles.
    x_refs (Tuple[Tensor, Tensor], optional): Pair of reference images for target styles.
    lambda_sty, lambda_ds, lambda_cyc (float): Weights for style, diversification, and cycle losses.

    Returns:
    Tensor: Total generator loss.
    Munch: A dictionary-like object containing individual loss components.
    """
    # Ensure that either latent codes or reference images are provided for the style transfer
    assert (z_trgs is None) != (x_refs is None), "Either z_trgs or x_refs must be provided, but not both or neither."

    # Compute style codes using either the mapping network or style encoder
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trgs[0], y_trg)
        s_trg2 = nets.mapping_network(z_trgs[1], y_trg)
    else:
        s_trg = nets.style_encoder(x_refs[0], y_trg)
        s_trg2 = nets.style_encoder(x_refs[1], y_trg)

    # Encode original image style for cycle consistency
    s_org = nets.style_encoder(x_real, y_org)
    
    # Generate fake images in the target style
    x_fake = nets.generator(x_real, s_trg)
    # Generate another fake image with a different style for diversity loss
    x_fake2 = nets.generator(x_real, s_trg2).detach()  # Detach to avoid gradients affecting the second image
    # Reconstruct the original image from the fake one for cycle consistency
    x_rec = nets.generator(x_fake, s_org)
    
    # Get discriminator's judgement of the fake image for adversarial loss
    out = nets.discriminator(x_fake, y_trg)
    
    # Calculate the adversarial loss (generator wants discriminator to judge fake as real)
    loss_adv = adversarial_loss(out, 1)
    # Calculate the style reconstruction loss (how well does the generated image retain the target style?)
    loss_sty = style_loss(nets.style_encoder(x_fake, y_trg), s_trg)
    # Calculate the diversity loss (are the two generated images in different styles distinguishable?)
    loss_ds = diversity_loss(x_fake, x_fake2)
    # Calculate the cycle consistency loss (can we reconstruct the original image from the fake one?)
    loss_cyc = cycle_loss(x_rec, x_real)

    # Combine all the loss components into the total loss
    total_loss = loss_adv + lambda_sty * loss_sty - lambda_ds * loss_ds + lambda_cyc * loss_cyc
    
    # Return the total loss along with individual loss components for logging or further analysis
    return total_loss, Munch(adv=loss_adv.item(), sty=loss_sty.item(), ds=loss_ds.item(), cyc=loss_cyc.item())


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
    assert (z_trg is None) != (x_ref is None)
    # Compute real loss
    x_real.requires_grad_() 
    out_real = nets.discriminator(x_real, y_org)
    # Regularization term
    loss_reg = r1_reg(out_real, x_real)
    loss_real =  adversarial_loss(out_real, 1)
    # Compute fake loss
    with torch.no_grad(): #explain why no grad ? F and E trained with generator
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        elif x_ref is not None:
            s_trg = nets.style_encoder(x_ref, y_trg)
        else:
            raise ValueError("Either z_trg or x_ref must be provided.")
        x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adversarial_loss(out, 0)
    
    # Combined Loss
    total_loss = loss_real + loss_fake + lambda_reg * loss_reg
    return total_loss, Munch(real=loss_real.item(), fake=loss_fake, reg=loss_reg)



# -----------------------------------------------------------test loss---------------------------------------------------
# import sys
# sys.path.append('architecture')
# from Generator import *
# from Mapping_Network import *
# from Style_Encoder import *
# from Discriminator import *
# # from Model import *
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

# # Calculate generator loss
# g_total_loss, g_loss_munch = loss_generator(
#     nets, real_img, domain_idx, domain_idx,
#     z_trgs=(latent_code, latent_code2)
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

import torch
import torch.nn.functional as F
import torch.nn as nn
from munch import Munch


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
    # Predicts style from the generated image using the specified branch
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
def loss_generator(nets, x_real, y_org, y_trg, z_trgs=None, x_refs=None, branch_idx=0, lambda_sty=1.0, lambda_ds=1.0, lambda_cyc=1.0):
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
        s_trg = nets.style_encoder(x_ref, y_trg, branch_idx)
        s_trg2 = nets.style_encoder(x_ref2, y_trg, branch_idx)
    else:
        raise ValueError("Either z_trgs or x_refs must be provided.")

    # Generate fake images
    x_fake = nets.generator(x_real, s_trg)
    x_fake2 = nets.generator(x_real, s_trg2)

    # Compute losses
    out_fake = nets.discriminator(x_fake, y_trg)
    loss_adv = adversarial_loss(nets.discriminator, x_real, x_fake, torch.ones_like(out_fake), torch.zeros_like(out_fake), y_org, y_trg)
    loss_sty = style_reconstruction_loss(nets.style_encoder, x_fake, s_trg, branch_idx)
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



# -----------------------------------------------------------test loss---------------------------------------------------

from Generator import *
from Mapping_Network import *
from Style_Encoder import *
from Discriminator import *
from Model import *
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

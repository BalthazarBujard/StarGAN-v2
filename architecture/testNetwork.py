from Generator import * 
from Mapping_Network import * 
from Style_Encoder import * 
from Discriminator import * 
from Model import * 
from torch.autograd import Variable

def test_generator(showArchitecure = False):
    # Set the parameters for the generator
    img_size = 256  # Image size
    style_dim = 64  # Style vector dimension


    # Create a Generator instance
    generator = Generator(img_size, style_dim)

    # Generate a random "image" tensor and a random "style" tensor
    # The image tensor should have the shape [batch_size, channels, height, width]
    # For simplicity, let's use a batch size of 1, and 3 channels (RGB)
    random_image = torch.randn(1, 3, img_size, img_size)
    random_style = torch.randn(1, style_dim)

    # Generate the output image using the generator
    with torch.no_grad():  # No need to compute gradients for testing
        output_image = generator(random_image, random_style)

    # Print the shape of the output image
    print(f"Output image shape: {output_image.shape}")
    if showArchitecure : print(generator)

def test_mapping_network(showArchitecure = False):
    # Set the parameters for the mapping network
    latent_dim = 16  # Dimension of the latent vector
    style_dim = 64   # Dimension of the style vector
    num_domains = 2  # Number of different domains

    # Create a MappingNetwork instance
    mapping_network = MappingNetwork(latent_dim, style_dim, num_domains)

    # Generate a batch of random latent vectors and domain indices
    # Let's use a batch size of 4 for this example
    batch_size = 4
    random_latent_vectors = torch.randn(batch_size, latent_dim)
    # Randomly select a domain for each sample in the batch
    domain_indices = torch.randint(low=0, high=num_domains, size=(batch_size,))

    # Generate the style vectors using the mapping network
    with torch.no_grad():  # No need to compute gradients for testing
        style_vectors = mapping_network(random_latent_vectors, domain_indices)

    # Print the shape of the style vectors to verify the output
    print(f"Style vectors shape: {style_vectors.shape}")
    if showArchitecure : print(mapping_network)

def test_style_encoder(showArchitecure):
    # Parameters
    batch_size = 1   # Number of images in the batch
    image_size = 256  # Size of the image (assuming square images)
    S_size = 64     # Size of the style vector
    outp_branches = 2 # Number of branches

    # Instantiate the encoder
    encoder = StyleEncoder(S_size,outp_branches)

    # Create a dummy input (batch of images)
    dummy_input = Variable(torch.randn(batch_size, 3, image_size, image_size))

    # Choose a branch for each item in the batch (randomly for testing)
    branches = torch.randint(0, outp_branches, (batch_size,))

    # Get the output from the encoder
    output = encoder(dummy_input, branches)

    print("Output shape: ", output.shape)
    print("Output: ", output)
    if showArchitecure : print(encoder)

def test_discriminator(showArchitecure):
    # Parameters
    batch_size = 4   # Number of images in the batch
    image_size = 256  # Size of the image (assuming square images)
    outp_branches = 2 # Number of branches

    # Instantiate the discriminator
    discriminator = Discriminator(outp_branches)

    # Create a dummy input (batch of images)
    dummy_input = Variable(torch.randn(batch_size, 3, image_size, image_size))

    # Choose a branch for each item in the batch (randomly for testing)
    branches = torch.randint(0, outp_branches, (batch_size,))

    # Get the output from the discriminator
    output = discriminator(dummy_input, branches)

    print("Output shape: ", output.shape)
    print("Output: ", output)
    if showArchitecure : print(discriminator)


# Run the test function

showArchitecure = False
test_mapping_network(showArchitecure)
test_generator(showArchitecure)
test_style_encoder(showArchitecure)
test_discriminator(showArchitecure)

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

Network,CopyNetwork = Model(params)
print(type(Network))
print(Network)

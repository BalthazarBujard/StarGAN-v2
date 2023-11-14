from Generator import * 
from Mapping_Network import * 

def test_generator():
    # Set the parameters for the generator
    img_size = 256  # Image size
    style_dim = 64  # Style vector dimension
    max_conv_dim = 512

    # Create a Generator instance
    generator = Generator(img_size, style_dim, max_conv_dim)

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

def test_mapping_network():
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



test_mapping_network()
test_generator()
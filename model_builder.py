import torch
from torch import nn

def get_generator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Creates a block for the generator's neural network.
    
    Args:
        input_dim (int): Dimension of the input vector.
        output_dim (int): Dimension of the output vector.
    
    Returns:
        nn.Sequential: A generator block with Linear, BatchNorm, and ReLU.
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

def get_discriminator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Creates a block for the discriminator's neural network.
    
    Args:
        input_dim (int): Dimension of the input vector.
        output_dim (int): Dimension of the output vector.
    
    Returns:
        nn.Sequential: A discriminator block with Linear and LeakyReLU.
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Generator(nn.Module):
    """
    Generator class for GAN.
    
    Args:
        z_dim (int): Dimension of the noise vector.
        im_dim (int): Dimension of the output image (default 784 for MNIST).
        hidden_dim (int): Inner dimension of hidden layers.
    """
    def __init__(self, z_dim: int = 10, im_dim: int = 784, hidden_dim: int = 128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            noise (torch.Tensor): Noise tensor of shape (n_samples, z_dim).
        
        Returns:
            torch.Tensor: Generated images of shape (n_samples, im_dim).
        """
        return self.gen(noise)

class Discriminator(nn.Module):
    """
    Discriminator class for GAN.
    
    Args:
        im_dim (int): Dimension of the input image (default 784 for MNIST).
        hidden_dim (int): Inner dimension of hidden layers.
    """
    def __init__(self, im_dim: int = 784, hidden_dim: int = 128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            image (torch.Tensor): Image tensor of shape (n_samples, im_dim).
        
        Returns:
            torch.Tensor: Probability scores of shape (n_samples, 1).
        """
        return self.disc(image)

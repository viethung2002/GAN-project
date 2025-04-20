import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

def get_noise(n_samples: int, z_dim: int, device: str = 'cpu') -> torch.Tensor:
    """
    Creates noise vectors from a normal distribution.
    
    Args:
        n_samples (int): Number of samples to generate.
        z_dim (int): Dimension of the noise vector.
        device (str): Device to place the tensor ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: Noise tensor of shape (n_samples, z_dim).
    """
    return torch.randn(n_samples, z_dim, device=device)

def show_tensor_images(image_tensor: torch.Tensor, num_images: int = 25, size: tuple = (1, 28, 28)):
    """
    Visualizes a tensor of images in a grid.
    
    Args:
        image_tensor (torch.Tensor): Tensor of images (expected in [0, 1]).
        num_images (int): Number of images to display.
        size (tuple): Size of each image (channels, height, width).
    """
    # Ensure image_tensor is in [0, 1]
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

def save_models(generator: torch.nn.Module, discriminator: torch.nn.Module, save_dir: str, epoch: int):
    """
    Saves the state dictionaries of the generator and discriminator.
    
    Args:
        generator (torch.nn.Module): Generator model.
        discriminator (torch.nn.Module): Discriminator model.
        save_dir (str): Directory to save the models.
        epoch (int): Current epoch number.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(save_dir, f'generator_epoch_{epoch}.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, f'discriminator_epoch_{epoch}.pth'))

def load_checkpoint(generator: torch.nn.Module, discriminator: torch.nn.Module, checkpoint_dir: str, epoch: int, device: str):
    """
    Loads the state dictionaries of the generator and discriminator from checkpoint files.
    
    Args:
        generator (torch.nn.Module): Generator model.
        discriminator (torch.nn.Module): Discriminator model.
        checkpoint_dir (str): Directory containing checkpoint files.
        epoch (int): Epoch number of the checkpoint to load.
        device (str): Device to load the tensors ('cpu' or 'cuda').
    
    Raises:
        FileNotFoundError: If checkpoint files do not exist.
    """
    gen_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth')
    disc_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth')
    
    if not os.path.exists(gen_path) or not os.path.exists(disc_path):
        raise FileNotFoundError(f"Checkpoint files not found: {gen_path}, {disc_path}")
    
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    discriminator.load_state_dict(torch.load(disc_path, map_location=device))
    print(f"Loaded checkpoint from epoch {epoch}")

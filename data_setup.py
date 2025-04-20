import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a DataLoader for the MNIST dataset.
    
    Args:
        data_dir (str): Directory to store/download MNIST data.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
    
    Returns:
        DataLoader: DataLoader for MNIST training data.
    """
    # Define transforms: Convert to tensor only
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

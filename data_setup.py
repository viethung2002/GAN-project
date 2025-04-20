import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    model_type: str = "gan",
    num_workers: int = 0
) -> DataLoader:
    """
    Tạo DataLoader cho tập dữ liệu MNIST, điều chỉnh chuẩn hóa theo loại mô hình.

    Args:
        data_dir (str): Thư mục lưu/tải dữ liệu MNIST.
        batch_size (int): Kích thước lô.
        model_type (str): Loại mô hình ('gan' hoặc 'dcgan').
        num_workers (int): Số tiến trình phụ để tải dữ liệu.

    Returns:
        DataLoader: DataLoader cho dữ liệu huấn luyện MNIST.
    """
    # Biến đổi: Chuyển thành tensor và chuẩn hóa theo model_type
    if model_type.lower() == "dcgan":
        # DCGAN: Chuẩn hóa về [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        # GAN: Không chuẩn hóa, giữ [0, 1]
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # Tải tập dữ liệu MNIST
    mnist_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )

    # Tạo DataLoader
    dataloader = DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

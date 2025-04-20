import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

def get_noise(n_samples: int, z_dim: int, device: str = 'cpu') -> torch.Tensor:
    """
    Tạo các vector nhiễu ngẫu nhiên từ phân phối chuẩn.

    Args:
        n_samples (int): Số mẫu cần tạo.
        z_dim (int): Kích thước của vector nhiễu.
        device (str): Thiết bị để đặt tensor ('cpu' hoặc 'cuda').

    Returns:
        torch.Tensor: Tensor nhiễu có kích thước (n_samples, z_dim).
    """
    return torch.randn(n_samples, z_dim, device=device)

def show_tensor_images(image_tensor: torch.Tensor, num_images: int = 25, size: tuple = (1, 28, 28), model_type: str = "gan"):
    """
    Hiển thị lưới hình ảnh từ tensor, điều chỉnh chuẩn hóa theo loại mô hình.

    Args:
        image_tensor (torch.Tensor): Tensor chứa hình ảnh (GAN: [0, 1], DCGAN: [-1, 1]).
        num_images (int): Số hình ảnh cần hiển thị.
        size (tuple): Kích thước mỗi hình ảnh (kênh, chiều cao, chiều rộng).
        model_type (str): Loại mô hình ('gan' hoặc 'dcgan').
    """
    # Chuẩn hóa về [0, 1] cho hiển thị
    if model_type.lower() == "dcgan":
        # DCGAN: Từ [-1, 1] về [0, 1]
        image_tensor = (image_tensor + 1) / 2
    # GAN: Đã ở [0, 1], không cần chuẩn hóa
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

def save_models(generator: torch.nn.Module, discriminator: torch.nn.Module, save_dir: str, epoch: int):
    """
    Lưu trạng thái của Generator và Discriminator.

    Args:
        generator (torch.nn.Module): Mô hình Generator.
        discriminator (torch.nn.Module): Mô hình Discriminator.
        save_dir (str): Thư mục lưu trữ mô hình.
        epoch (int): Số epoch hiện tại.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(save_dir, f'generator_epoch_{epoch}.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, f'discriminator_epoch_{epoch}.pth'))

def load_checkpoint(generator: torch.nn.Module, discriminator: torch.nn.Module, checkpoint_dir: str, epoch: int, device: str):
    """
    Tải trạng thái của Generator và Discriminator từ tệp checkpoint.

    Args:
        generator (torch.nn.Module): Mô hình Generator.
        discriminator (torch.nn.Module): Mô hình Discriminator.
        checkpoint_dir (str): Thư mục chứa tệp checkpoint.
        epoch (int): Số epoch của checkpoint cần tải.
        device (str): Thiết bị để tải tensor ('cpu' hoặc 'cuda').

    Raises:
        FileNotFoundError: Nếu tệp checkpoint không tồn tại.
    """
    gen_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth')
    disc_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth')

    if not os.path.exists(gen_path) or not os.path.exists(disc_path):
        raise FileNotFoundError(f"Checkpoint files not found: {gen_path}, {disc_path}")

    generator.load_state_dict(torch.load(gen_path, map_location=device))
    discriminator.load_state_dict(torch.load(disc_path, map_location=device))
    print(f"Loaded checkpoint from epoch {epoch}")

def get_generator_loss(
    disc_fake: torch.Tensor,
    criterion: torch.nn.Module,
    real_labels: torch.Tensor,
    model_type: str = "gan"
) -> torch.Tensor:
    """
    Tính mất mát của Generator.

    Generator muốn Discriminator đánh giá hình ảnh giả là thật (D(fake) ≈ 1).

    Args:
        disc_fake (torch.Tensor): Đầu ra của Discriminator cho hình ảnh giả (logits hoặc xác suất).
        criterion (torch.nn.Module): Hàm mất mát (BCELoss hoặc BCEWithLogitsLoss).
        real_labels (torch.Tensor): Nhãn thật (0.9 hoặc 1.0).
        model_type (str): Loại mô hình ('gan' hoặc 'dcgan').

    Returns:
        torch.Tensor: Giá trị mất mát của Generator.
    """
    return criterion(disc_fake, real_labels)

def get_discriminator_loss(
    disc_real: torch.Tensor,
    disc_fake: torch.Tensor,
    criterion: torch.nn.Module,
    real_labels: torch.Tensor,
    fake_labels: torch.Tensor,
    model_type: str = "gan"
) -> torch.Tensor:
    """
    Tính mất mát của Discriminator.

    Discriminator muốn phân biệt hình ảnh thật (D(real) ≈ 1) và giả (D(fake) ≈ 0).

    Args:
        disc_real (torch.Tensor): Đầu ra của Discriminator cho hình ảnh thật (logits hoặc xác suất).
        disc_fake (torch.Tensor): Đầu ra của Discriminator cho hình ảnh giả (logits hoặc xác suất).
        criterion (torch.nn.Module): Hàm mất mát (BCELoss hoặc BCEWithLogitsLoss).
        real_labels (torch.Tensor): Nhãn thật (0.9 hoặc 1.0).
        fake_labels (torch.Tensor): Nhãn giả (0.0).
        model_type (str): Loại mô hình ('gan' hoặc 'dcgan').

    Returns:
        torch.Tensor: Giá trị mất mát của Discriminator (trung bình của thật và giả).
    """
    disc_real_loss = criterion(disc_real, real_labels)
    disc_fake_loss = criterion(disc_fake, fake_labels)
    return (disc_real_loss + disc_fake_loss) / 2

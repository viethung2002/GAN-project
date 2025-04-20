import torch
from torch import nn

def get_gan_generator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Tạo khối cho Generator của standard GAN (fully connected).

    Args:
        input_dim (int): Kích thước đầu vào.
        output_dim (int): Kích thước đầu ra.

    Returns:
        nn.Sequential: Khối với Linear, BatchNorm, ReLU.
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

def get_gan_discriminator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Tạo khối cho Discriminator của standard GAN (fully connected).

    Args:
        input_dim (int): Kích thước đầu vào.
        output_dim (int): Kích thước đầu ra.

    Returns:
        nn.Sequential: Khối với Linear, LeakyReLU.
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

class GANGenerator(nn.Module):
    """
    Generator cho standard GAN (fully connected).

    Args:
        z_dim (int): Kích thước vector nhiễu.
        im_dim (int): Kích thước hình ảnh (mặc định 784 cho MNIST).
        hidden_dim (int): Kích thước ẩn.
    """
    def __init__(self, z_dim: int = 10, im_dim: int = 784, hidden_dim: int = 128):
        super(GANGenerator, self).__init__()
        self.gen = nn.Sequential(
            get_gan_generator_block(z_dim, hidden_dim),
            get_gan_generator_block(hidden_dim, hidden_dim * 2),
            get_gan_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_gan_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
        # Khởi tạo trọng số
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Khởi tạo trọng số (chuẩn hóa, mean=0, std=0.02).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Chạy forward pass của GAN Generator.

        Args:
            noise (torch.Tensor): Tensor nhiễu (n_samples, z_dim).

        Returns:
            torch.Tensor: Hình ảnh tạo ra (n_samples, im_dim).
        """
        return self.gen(noise)

class GANDiscriminator(nn.Module):
    """
    Discriminator cho standard GAN (fully connected).

    Args:
        im_dim (int): Kích thước hình ảnh (mặc định 784 cho MNIST).
        hidden_dim (int): Kích thước ẩn.
    """
    def __init__(self, im_dim: int = 784, hidden_dim: int = 128):
        super(GANDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            get_gan_discriminator_block(im_dim, hidden_dim * 4),
            get_gan_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_gan_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Khởi tạo trọng số
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Khởi tạo trọng số (chuẩn hóa, mean=0, std=0.02).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Chạy forward pass của GAN Discriminator.

        Args:
            image (torch.Tensor): Tensor hình ảnh (n_samples, im_dim).

        Returns:
            torch.Tensor: Điểm xác suất (n_samples, 1).
        """
        return self.disc(image)

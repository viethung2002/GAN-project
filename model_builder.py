import torch
from torch import nn
from abc import ABC, abstractmethod

class BaseGenerator(nn.Module, ABC):
    """
    Lớp cơ sở trừu tượng cho Generator.
    """
    @abstractmethod
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        pass

class BaseDiscriminator(nn.Module, ABC):
    """
    Lớp cơ sở trừu tượng cho Discriminator.
    """
    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pass

# GAN (Fully Connected)
def get_gan_generator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Tạo khối cho Generator của GAN (fully connected).

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
    Tạo khối cho Discriminator của GAN (fully connected).

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

class GANGenerator(BaseGenerator):
    """
    Generator cho GAN (fully connected).

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

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Chạy forward pass của GAN Generator.

        Args:
            noise (torch.Tensor): Tensor nhiễu (n_samples, z_dim).

        Returns:
            torch.Tensor: Hình ảnh tạo ra (n_samples, im_dim).
        """
        return self.gen(noise)

class GANDiscriminator(BaseDiscriminator):
    """
    Discriminator cho GAN (fully connected).

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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Chạy forward pass của GAN Discriminator.

        Args:
            image (torch.Tensor): Tensor hình ảnh (n_samples, im_dim).

        Returns:
            torch.Tensor: Điểm xác suất (n_samples, 1).
        """
        return self.disc(image)

# DCGAN (Convolutional)
def get_dcgan_generator_block(input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 2, final_layer: bool = False) -> nn.Sequential:
    """
    Tạo khối cho Generator của DCGAN (convolutional).

    Args:
        input_channels (int): Số kênh đầu vào.
        output_channels (int): Số kênh đầu ra.
        kernel_size (int): Kích thước bộ lọc.
        stride (int): Bước tích chập.
        final_layer (bool): True nếu là lớp cuối.

    Returns:
        nn.Sequential: Khối với ConvTranspose2d, BatchNorm2d, ReLU/Tanh.
    """
    if not final_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.Tanh()
        )

def get_dcgan_discriminator_block(input_channels: int, output_channels: int, kernel_size: int = 4, stride: int = 2, final_layer: bool = False) -> nn.Sequential:
    """
    Tạo khối cho Discriminator của DCGAN (convolutional).

    Args:
        input_channels (int): Số kênh đầu vào.
        output_channels (int): Số kênh đầu ra.
        kernel_size (int): Kích thước bộ lọc.
        stride (int): Bước tích chập.
        final_layer (bool): True nếu là lớp cuối.

    Returns:
        nn.Sequential: Khối với Conv2d, BatchNorm2d, LeakyReLU.
    """
    if not final_layer:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride)
        )

class DCGANGenerator(BaseGenerator):
    """
    Generator cho DCGAN (convolutional).

    Args:
        z_dim (int): Kích thước vector nhiễu.
        im_chan (int): Số kênh hình ảnh (mặc định 1 cho MNIST).
        hidden_dim (int): Kích thước ẩn.
    """
    def __init__(self, z_dim: int = 10, im_chan: int = 1, hidden_dim: int = 64):
        super(DCGANGenerator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            get_dcgan_generator_block(z_dim, hidden_dim * 4),
            get_dcgan_generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            get_dcgan_generator_block(hidden_dim * 2, hidden_dim),
            get_dcgan_generator_block(hidden_dim, im_chan, kernel_size=4, final_layer=True)
        )

    def unsqueeze_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Reshape vector nhiễu thành dạng phù hợp với tích chập.

        Args:
            noise (torch.Tensor): Tensor nhiễu (n_samples, z_dim).

        Returns:
            torch.Tensor: Tensor nhiễu (n_samples, z_dim, 1, 1).
        """
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Chạy forward pass của DCGAN Generator.

        Args:
            noise (torch.Tensor): Tensor nhiễu (n_samples, z_dim).

        Returns:
            torch.Tensor: Hình ảnh tạo ra (n_samples, im_chan, 28, 28).
        """
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

class DCGANDiscriminator(BaseDiscriminator):
    """
    Discriminator cho DCGAN (convolutional).

    Args:
        im_chan (int): Số kênh hình ảnh (mặc định 1 cho MNIST).
        hidden_dim (int): Kích thước ẩn.
    """
    def __init__(self, im_chan: int = 1, hidden_dim: int = 16):
        super(DCGANDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            get_dcgan_discriminator_block(im_chan, hidden_dim),
            get_dcgan_discriminator_block(hidden_dim, hidden_dim * 2),
            get_dcgan_discriminator_block(hidden_dim * 2, 1, final_layer=True)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Chạy forward pass của DCGAN Discriminator.

        Args:
            image (torch.Tensor): Tensor hình ảnh (n_samples, im_chan, 28, 28).

        Returns:
            torch.Tensor: Điểm xác suất (n_samples, 1).
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

def get_model(model_type: str, z_dim: int = 10, im_chan: int = 1, im_dim: int = 784):
    """
    Tạo Generator và Discriminator dựa trên loại mô hình.

    Args:
        model_type (str): Loại mô hình ('gan' hoặc 'dcgan').
        z_dim (int): Kích thước vector nhiễu.
        im_chan (int): Số kênh hình ảnh (cho DCGAN).
        im_dim (int): Kích thước hình ảnh phẳng (cho GAN).

    Returns:
        tuple: (Generator, Discriminator).
    """
    if model_type.lower() == "dcgan":
        return DCGANGenerator(z_dim=z_dim, im_chan=im_chan), DCGANDiscriminator(im_chan=im_chan)
    elif model_type.lower() == "gan":
        return GANGenerator(z_dim=z_dim, im_dim=im_dim), GANDiscriminator(im_dim=im_dim)
    else:
        raise ValueError(f"Loại mô hình không hợp lệ: {model_type}. Chọn 'gan' hoặc 'dcgan'.")

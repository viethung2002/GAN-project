import torch
from torch import nn

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
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=0),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=0),
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
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=0),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=0)
        )

class DCGANGenerator(nn.Module):
    """
    Generator cho DCGAN (convolutional).

    Args:
        z_dim (int): Kích thước vector nhiễu.
        im_chan (int): Số kênh hình ảnh (mặc định 1 cho MNIST).
        hidden_dim (int): Kích thước ẩn.
    """
    def __init__(self, z_dim: int = 64, im_chan: int = 1, hidden_dim: int = 64):
        super(DCGANGenerator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            get_dcgan_generator_block(z_dim, hidden_dim * 4, kernel_size=3, stride=2),
            get_dcgan_generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            get_dcgan_generator_block(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2),
            get_dcgan_generator_block(hidden_dim, im_chan, kernel_size=4, stride=2, final_layer=True)
        )
        # Khởi tạo trọng số
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Khởi tạo trọng số theo khuyến nghị của DCGAN (chuẩn hóa, mean=0, std=0.02).
        """
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

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

class DCGANDiscriminator(nn.Module):
    """
    Discriminator cho DCGAN (convolutional).

    Args:
        im_chan (int): Số kênh hình ảnh (mặc định 1 cho MNIST).
        hidden_dim (int): Kích thước ẩn.
    """
    def __init__(self, im_chan: int = 1, hidden_dim: int = 16):
        super(DCGANDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            get_dcgan_discriminator_block(im_chan, hidden_dim, kernel_size=4, stride=2),
            get_dcgan_discriminator_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2),
            get_dcgan_discriminator_block(hidden_dim * 2, 1, kernel_size=4, stride=2, final_layer=True)
        )
        # Khởi tạo trọng số
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Khởi tạo trọng số theo khuyến nghị của DCGAN (chuẩn hóa, mean=0, std=0.02).
        """
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

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

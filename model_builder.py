from gan_model import GANGenerator, GANDiscriminator
from dcgan_model import DCGANGenerator, DCGANDiscriminator

def get_model(model_type: str, z_dim: int = 64, im_chan: int = 1, im_dim: int = 784):
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

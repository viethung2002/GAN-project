import torch
from torch import nn
from tqdm.auto import tqdm
from utils import get_noise, show_tensor_images, save_models, get_generator_loss, get_discriminator_loss

def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    z_dim: int,
    n_epochs: int,
    lr: float,
    device: str,
    model_type: str = "gan",
    display_step: int = 500,
    save_dir: str = "models"
) -> tuple[list, list]:
    """
    Huấn luyện mô hình GAN hoặc DCGAN.

    Args:
        generator (nn.Module): Mô hình Generator.
        discriminator (nn.Module): Mô hình Discriminator.
        dataloader (DataLoader): DataLoader cho dữ liệu huấn luyện.
        z_dim (int): Kích thước vector nhiễu.
        n_epochs (int): Số epoch huấn luyện.
        lr (float): Tỷ lệ học.
        device (str): Thiết bị huấn luyện ('cpu' hoặc 'cuda').
        model_type (str): Loại mô hình ('gan' hoặc 'dcgan').
        display_step (int): Khoảng cách bước để hiển thị hình ảnh.
        save_dir (str): Thư mục lưu mô hình.

    Returns:
        tuple: Danh sách mất mát của Generator và Discriminator.
    """
    criterion = nn.BCELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    generator.to(device)
    discriminator.to(device)

    gen_losses = []
    disc_losses = []
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            cur_batch_size = len(real)
            real = real.to(device)

            # Điều chỉnh hình ảnh theo model_type
            if model_type.lower() == "gan":
                real = real.view(cur_batch_size, -1)  # Phẳng cho GAN
            # DCGAN: Giữ dạng (batch_size, 1, 28, 28)

            # In kích thước tensor để gỡ lỗi
            if cur_step == 0:
                print(f"Real shape: {real.shape}, Batch size: {cur_batch_size}")

            # Huấn luyện Discriminator
            disc_optimizer.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = generator(fake_noise)

            # Nhiễu nhãn
            real_labels = torch.ones(cur_batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(cur_batch_size, 1, device=device)

            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(real)

            disc_loss = get_discriminator_loss(
                disc_real_pred, disc_fake_pred, criterion, real_labels, fake_labels
            )
            disc_loss.backward(retain_graph=True)
            disc_optimizer.step()

            # Theo dõi mất mát Discriminator
            mean_discriminator_loss += disc_loss.item() / display_step

            # Huấn luyện Generator
            gen_optimizer.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_2 = generator(fake_noise_2)
            disc_fake_pred = discriminator(fake_2)

            gen_loss = get_generator_loss(disc_fake_pred, criterion, real_labels)
            gen_loss.backward()
            gen_optimizer.step()

            # Theo dõi mất mát Generator
            mean_generator_loss += gen_loss.item() / display_step
            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            # Hiển thị hình ảnh
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss:.4f}, Discriminator loss: {mean_discriminator_loss:.4f}")
                fake_display = fake.view(-1, 1, 28, 28) if model_type.lower() == "gan" else fake
                real_display = real.view(-1, 1, 28, 28) if model_type.lower() == "gan" else real
                show_tensor_images(fake_display, model_type=model_type)
                show_tensor_images(real_display, model_type=model_type)
                mean_generator_loss = 0
                mean_discriminator_loss = 0

            cur_step += 1

        # Lưu mô hình mỗi 50 epoch
        if (epoch + 1) % 50 == 0:
            save_models(generator, discriminator, save_dir, epoch + 1)

    return gen_losses, disc_losses

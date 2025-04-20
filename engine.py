import torch
from torch import nn
from tqdm.auto import tqdm
from utils import get_noise, show_tensor_images, save_models

def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    z_dim: int,
    n_epochs: int,
    lr: float,
    device: str,
    display_step: int = 500,
    save_dir: str = "models"
) -> tuple[list, list]:
    """
    Trains the GAN model.
    
    Args:
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        dataloader (DataLoader): DataLoader for training data.
        z_dim (int): Dimension of the noise vector.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate for optimizers.
        device (str): Device to train on ('cpu' or 'cuda').
        display_step (int): Interval to display generated images.
        save_dir (str): Directory to save models.
    
    Returns:
        tuple: Lists of generator and discriminator losses.
    """
    criterion = nn.BCELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    generator.to(device)
    discriminator.to(device)
    
    gen_losses = []
    disc_losses = []
    cur_step = 0
    
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            batch_size = real.size(0)
            real = real.view(batch_size, -1).to(device)
            
            # Print tensor shapes for debugging
            if cur_step == 0:
                print(f"Real shape: {real.shape}, Batch size: {batch_size}")
            
            # Train Discriminator
            disc_optimizer.zero_grad()
            noise = get_noise(batch_size, z_dim, device)
            fake = generator(noise)
            
            # Label smoothing
            real_labels = torch.ones(batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            disc_real = discriminator(real)
            disc_fake = discriminator(fake.detach())
            
            disc_real_loss = criterion(disc_real, real_labels)
            disc_fake_loss = criterion(disc_fake, fake_labels)
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train Generator
            gen_optimizer.zero_grad()
            disc_fake = discriminator(fake)
            gen_loss = criterion(disc_fake, real_labels)  # Generator wants fake to be classified as real
            gen_loss.backward()
            gen_optimizer.step()
            
            # Track losses
            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())
            
            # Visualize
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Step {cur_step}: Gen loss: {gen_loss.item():.4f}, Disc loss: {disc_loss.item():.4f}")
                fake_reshaped = fake.view(-1, 1, 28, 28)
                real_reshaped = real.view(-1, 1, 28, 28)
                show_tensor_images(fake_reshaped)
                show_tensor_images(real_reshaped)
            
            cur_step += 1
        
        # Save models every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_models(generator, discriminator, save_dir, epoch + 1)
    
    return gen_losses, disc_losses

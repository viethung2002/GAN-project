import torch
import argparse
from data_setup import create_dataloaders
from model_builder import Generator, Discriminator
from engine import train_gan
from utils import load_checkpoint

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--z_dim", type=int, default=10, help="Noise vector dimension")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for MNIST data")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Directory containing checkpoint files")
    parser.add_argument("--checkpoint_epoch", type=int, default=None, help="Epoch number of checkpoint to load")
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup data
    dataloader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Setup models
    generator = Generator(z_dim=args.z_dim)
    discriminator = Discriminator()
    
    # Load checkpoint if provided
    if args.checkpoint_path and args.checkpoint_epoch:
        try:
            load_checkpoint(generator, discriminator, args.checkpoint_path, args.checkpoint_epoch, device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    
    # Train
    try:
        gen_losses, disc_losses = train_gan(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            z_dim=args.z_dim,
            n_epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_dir=args.save_dir
        )
        print("Training completed!")
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()

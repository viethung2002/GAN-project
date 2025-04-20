import torch
import argparse
from data_setup import create_dataloaders
from model_builder import get_model
from engine import train_gan
from utils import load_checkpoint

def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Huấn luyện GAN hoặc DCGAN trên MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Kích thước lô")
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=0.0002, help="Tỷ lệ học")
    parser.add_argument("--z_dim", type=int, default=10, help="Kích thước vector nhiễu")
    parser.add_argument("--model_type", type=str, default="gan", help="Loại mô hình: 'gan' hoặc 'dcgan'")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục dữ liệu MNIST")
    parser.add_argument("--save_dir", type=str, default="models", help="Thư mục lưu mô hình")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Thư mục chứa checkpoint")
    parser.add_argument("--checkpoint_epoch", type=int, default=None, help="Epoch của checkpoint để tải")
    args = parser.parse_args()

    # Thiết lập thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # Thiết lập dữ liệu
    dataloader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        model_type=args.model_type
    )

    # Thiết lập mô hình
    generator, discriminator = get_model(
        model_type=args.model_type,
        z_dim=args.z_dim,
        im_chan=1,
        im_dim=784
    )

    # Tải checkpoint nếu có
    if args.checkpoint_path and args.checkpoint_epoch:
        try:
            load_checkpoint(generator, discriminator, args.checkpoint_path, args.checkpoint_epoch, device)
        except Exception as e:
            print(f"Lỗi khi tải checkpoint: {e}")
            return

    # Huấn luyện
    try:
        gen_losses, disc_losses = train_gan(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            z_dim=args.z_dim,
            n_epochs=args.epochs,
            lr=args.lr,
            device=device,
            model_type=args.model_type,
            save_dir=args.save_dir
        )
        print("Hoàn tất huấn luyện!")
    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện: {e}")
        raise

if __name__ == "__main__":
    main()

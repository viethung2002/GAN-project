# GAN Project: Generating Handwritten Digits with PyTorch

This project implements a **Generative Adversarial Network (GAN)** to generate handwritten digit images similar to the MNIST dataset. It is based on the `C1W1_Your_First_GAN.ipynb` from a machine learning course, restructured into a modular Python project for better organization and scalability.

The GAN consists of a **Generator** (creates fake images from random noise) and a **Discriminator** (distinguishes real MNIST images from fake ones). The code is written in **PyTorch** and supports training on both CPU and GPU, with features like checkpoint saving and resuming from pre-trained models.

## Features
- Modular structure with separate modules for data loading, model building, training, and utilities.
- Trains a GAN to generate 28x28 grayscale digit images.
- Saves model checkpoints every 50 epochs.
- Supports resuming training from pre-trained checkpoints.
- Visualizes generated and real images during training.

## Project Structure
```
gan_project/
├── data/
│   └── MNIST/              # Downloaded automatically by torchvision
├── models/                 # Stores saved model checkpoints
├── data_setup.py           # Loads and prepares MNIST dataset
├── model_builder.py        # Defines Generator and Discriminator models
├── utils.py                # Utility functions (noise generation, visualization, checkpoint handling)
├── engine.py               # Training logic for the GAN
├── train.py                # Main script to run training
└── README.md               # Project documentation
```

## Requirements
- Python 3.8+
- PyTorch (with torchvision)
- tqdm (for progress bars)
- matplotlib (for visualization)

## Installation
1. **Clone or set up the project**:
   - If you haven't set up the directory structure, run the following script to create it:
     ```bash
     python create_project_structure.py
     ```
     (See [previous response](#) for the script if needed.)

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision tqdm matplotlib
   ```
   For GPU support, install the CUDA version of PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### 1. Train from Scratch
Run the following command to train the GAN from scratch for 100 epochs, saving checkpoints every 50 epochs:
```bash
python train.py --batch_size 128 --epochs 100 --lr 0.0002 --z_dim 10 --save_dir models
```

- **Arguments**:
  - `--batch_size`: Number of images per batch (default: 128).
  - `--epochs`: Number of training epochs (default: 50).
  - `--lr`: Learning rate for optimizers (default: 0.0002).
  - `--z_dim`: Dimension of noise vector (default: 10).
  - `--save_dir`: Directory to save checkpoints (default: models).
- **Output**:
  - Checkpoints saved at `models/generator_epoch_50.pth` and `models/discriminator_epoch_50.pth` (and every 50 epochs).
  - Generated and real images displayed every 500 steps.
  - Training progress and losses printed to the console.

### 2. Resume Training from Checkpoint
To continue training from a pre-trained model (e.g., checkpoint at epoch 50):
```bash
python train.py --batch_size 128 --epochs 50 --lr 0.0002 --z_dim 10 --save_dir models --checkpoint_path models --checkpoint_epoch 50
```

- **Additional Arguments**:
  - `--checkpoint_path`: Directory containing checkpoint files (e.g., `models/`).
  - `--checkpoint_epoch`: Epoch number of the checkpoint to load (e.g., `50`).
- **Output**:
  - Loads `models/generator_epoch_50.pth` and `models/discriminator_epoch_50.pth`.
  - Trains for an additional 50 epochs, saving new checkpoints at epoch 100.

### 3. Check Generated Images
During training, generated images are displayed every 500 steps. To manually check the quality of a pre-trained Generator:
1. Modify `train.py` to include a test section (example):
   ```python
   from utils import get_noise, show_tensor_images
   generator.eval()
   with torch.no_grad():
       noise = get_noise(25, args.z_dim, device)
       fake = generator(noise).view(-1, 1, 28, 28)
       show_tensor_images(fake)
   ```
2. Run `train.py` with the checkpoint loaded to visualize generated images.

## Notes
- **Training Stability**:
  - If the Generator loss increases rapidly and Discriminator loss drops to near 0, try:
    - Increasing `--lr` (e.g., `0.0003`).
    - Increasing `--z_dim` (e.g., `64`).
    - Adding more epochs (e.g., `--epochs 200`).
  - The code includes label smoothing (0.9 for real labels) to improve stability.
- **Checkpoint Compatibility**:
  - Ensure the checkpoint files match the model architecture in `model_builder.py`.
  - Checkpoints are saved in `models/` with names like `generator_epoch_50.pth`.
- **GPU Support**:
  - The code automatically uses CUDA if available. Verify GPU support:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```
  - If `False`, reinstall PyTorch with CUDA support.
- **Common Errors**:
  - **Checkpoint not found**: Verify `--checkpoint_path` and `--checkpoint_epoch` match existing files.
  - **Tensor size mismatch**: Check console output for tensor shapes (printed at the first step).
  - **Visualization issues**: Ensure `matplotlib` is installed and images are in `[0, 1]` range.

## Contributing
Feel free to open issues or submit pull requests to improve the project. Suggestions for enhancing training stability, adding new features (e.g., loss plotting), or optimizing the code are welcome.

## Acknowledgments
- Based on the `C1W1_Your_First_GAN.ipynb` from a machine learning course.
- Inspired by the modular structure from [Learn PyTorch: Going Modular](https://www.learnpytorch.io/05_pytorch_going_modular/).
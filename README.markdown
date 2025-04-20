# GAN/DCGAN Project: Generating Handwritten Digits with PyTorch

This project implements a flexible **Generative Adversarial Network (GAN)** framework that supports both **standard GAN** (fully connected layers) and **Deep Convolutional GAN (DCGAN)** (convolutional layers) to generate handwritten digit images similar to the MNIST dataset. It is based on the `C1W1_Your_First_GAN.ipynb` and `C1_W2_Assignment.ipynb` from a machine learning course, restructured into a modular Python project for better organization, scalability, and flexibility.

The framework allows users to choose between **GAN** (from `C1W1_Your_First_GAN.ipynb`) and **DCGAN** (from `C1_W2_Assignment.ipynb`) via a command-line argument (`--model_type`). The code is written in **PyTorch** and supports training on both CPU and GPU, with features like checkpoint saving, resuming from pre-trained models, and separate loss functions.

## Features
- Modular structure with separate modules for data loading, model building, training, and utilities.
- Supports both **GAN** (fully connected) and **DCGAN** (convolutional) architectures, selectable via `--model_type`.
- Trains models to generate 28x28 grayscale digit images.
- Saves model checkpoints every 50 epochs.
- Supports resuming training from pre-trained checkpoints.
- Visualizes generated and real images during training.
- Separate loss functions (`get_generator_loss`, `get_discriminator_loss`) for modularity.
- Label smoothing (real labels set to 0.9) for training stability.

## Project Structure
```
gan_project/
├── data/
│   └── MNIST/              # Downloaded automatically by torchvision
├── models/                 # Stores saved model checkpoints
├── data_setup.py           # Loads and prepares MNIST dataset, adjusts normalization based on model type
├── model_builder.py        # Defines GAN and DCGAN Generator/Discriminator models
├── utils.py                # Utility functions (noise generation, visualization, checkpoint handling, loss functions)
├── engine.py               # Training logic for GAN/DCGAN
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
Run the following command to train a GAN or DCGAN from scratch for 100 epochs, saving checkpoints every 50 epochs:

- **For standard GAN**:
  ```bash
  python train.py --batch_size 128 --epochs 100 --lr 0.0002 --z_dim 10 --model_type gan --save_dir models
  ```

- **For DCGAN**:
  ```bash
  python train.py --batch_size 128 --epochs 100 --lr 0.0002 --z_dim 10 --model_type dcgan --save_dir models
  ```

- **Arguments**:
  - `--batch_size`: Number of images per batch (default: 128).
  - `--epochs`: Number of training epochs (default: 50).
  - `--lr`: Learning rate for optimizers (default: 0.0002).
  - `--z_dim`: Dimension of noise vector (default: 10).
  - `--model_type`: Model type, either `gan` or `dcgan` (default: gan).
  - `--save_dir`: Directory to save checkpoints (default: models).
- **Output**:
  - Checkpoints saved at `models/generator_epoch_50.pth` and `models/discriminator_epoch_50.pth` (and every 50 epochs).
  - Generated and real images displayed every 500 steps.
  - Training progress and average losses printed to the console.

### 2. Resume Training from Checkpoint
To continue training from a pre-trained model (e.g., checkpoint at epoch 50):

- **For GAN**:
  ```bash
  python train.py --batch_size 128 --epochs 50 --lr 0.0002 --z_dim 10 --model_type gan --save_dir models --checkpoint_path models --checkpoint_epoch 50
  ```

- **For DCGAN**:
  ```bash
  python train.py --batch_size 128 --epochs 50 --lr 0.0002 --z_dim 10 --model_type dcgan --save_dir models --checkpoint_path models --checkpoint_epoch 50
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
       fake = generator(noise)
       if args.model_type.lower() == "gan":
           fake = fake.view(-1, 1, 28, 28)
       show_tensor_images(fake, model_type=args.model_type)
   ```
2. Run `train.py` with the checkpoint loaded to visualize generated images.

## Model Architectures
- **GAN (Fully Connected)**:
  - **Generator**:
    - Uses fully connected layers (`Linear`) with 5 layers.
    - Hidden layers: `Linear` + `BatchNorm1d` + `ReLU`.
    - Output layer: `Linear` + `Sigmoid` (produces images in `[0, 1]`).
    - Input: Noise vector of size `z_dim` (default: 10).
    - Output: Flattened images of size `(784)` (reshaped to `(1, 28, 28)` for display).
  - **Discriminator**:
    - Uses fully connected layers (`Linear`) with 4 layers.
    - Hidden layers: `Linear` + `LeakyReLU` (slope 0.2).
    - Output layer: `Linear` + `Sigmoid`.
    - Input: Flattened images of size `(784)`.
    - Output: Probability scores of size `(batch_size, 1)`.
- **DCGAN (Convolutional)**:
  - **Generator**:
    - Uses transposed convolutions (`ConvTranspose2d`) with 4 layers.
    - Hidden layers: `ConvTranspose2d` + `BatchNorm2d` + `ReLU`.
    - Output layer: `ConvTranspose2d` + `Tanh` (produces images in `[-1, 1]`).
    - Input: Noise vector of size `z_dim` (default: 10).
    - Output: Grayscale images of size `(1, 28, 28)`.
  - **Discriminator**:
    - Uses convolutions (`Conv2d`) with 3 layers.
    - Hidden layers: `Conv2d` + `BatchNorm2d` + `LeakyReLU` (slope 0.2).
    - Output layer: `Conv2d` (no activation, produces logits).
    - Input: Images of size `(1, 28, 28)`.
    - Output: Probability scores of size `(batch_size, 1)`.

## Loss Functions
The project includes separate loss functions for better modularity:
- **`get_generator_loss`**: Calculates the Generator's loss, encouraging the Discriminator to classify fake images as real (`D(fake) ≈ 1`).
- **`get_discriminator_loss`**: Calculates the Discriminator's loss, aiming to correctly classify real images (`D(real) ≈ 1`) and fake images (`D(fake) ≈ 0`).
- Both use `BCELoss` (Binary Cross-Entropy) with label smoothing (real labels set to 0.9) for training stability.

These functions are defined in `utils.py` and used in `engine.py` during training.

## Notes
- **Training Stability**:
  - If the Generator loss increases rapidly and Discriminator loss drops to near 0, try:
    - Increasing `--lr` (e.g., `0.0003`).
    - Increasing `--z_dim` (e.g., `64`).
    - Adding more epochs (e.g., `--epochs 200`).
  - The code includes label smoothing (0.9 for real labels) to improve stability, particularly for DCGAN.
- **Checkpoint Compatibility**:
  - Ensure the checkpoint files match the model architecture (`GANGenerator`/`GANDiscriminator` or `DCGANGenerator`/`DCGANDiscriminator`).
  - Checkpoints are saved in `models/` with names like `generator_epoch_50.pth`.
- **Model Selection**:
  - Use `--model_type gan` for the standard GAN (faster for small datasets but less expressive).
  - Use `--model_type dcgan` for DCGAN (better for image generation but computationally heavier).
- **GPU Support**:
  - The code automatically uses CUDA if available. Verify GPU support:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```
  - If `False`, reinstall PyTorch with CUDA support.
- **Common Errors**:
  - **Checkpoint not found**: Verify `--checkpoint_path` and `--checkpoint_epoch` match existing files.
  - **Tensor size mismatch**: Check console output for tensor shapes (printed at the first step).
  - **Visualization issues**: Ensure `matplotlib` is installed and images are correctly normalized (`[0, 1]` for GAN, `[-1, 1]` for DCGAN).

## Contributing
Feel free to open issues or submit pull requests to improve the project. Suggestions for enhancing training stability, adding new features (e.g., loss plotting, new GAN variants), or optimizing the code are welcome.

## Acknowledgments
- Based on the `C1W1_Your_First_GAN.ipynb` and `C1_W2_Assignment.ipynb` from a machine learning course.
- Inspired by the modular structure from [Learn PyTorch: Going Modular](https://www.learnpytorch.io/05_pytorch_going_modular/).
- DCGAN architecture from Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2016).

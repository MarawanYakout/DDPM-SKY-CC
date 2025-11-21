# DDPM-SKY-CC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/MarawanYakout/DDPM-SKY-CC)

> **⚠️ Development Status:** This project is currently in **alpha testing**. Features and APIs may change without notice.

A context-free DDPM (Denoising Diffusion Probabilistic Model) training pipeline for wind-speed imagery using pre-generated noise for fully deterministic and reproducible experiments.

---

## Overview

DDPM-SKY-CC provides a complete training pipeline for diffusion models with a focus on reproducibility. Unlike traditional implementations that generate noise on-the-fly, this system uses **pre-generated deterministic noise** stored on disk, enabling:

- Fully reproducible training runs
- Simplified debugging and experimentation
- Separation of data preparation, noise generation, and training phases

---

## Key Features

- **Deterministic Training**: Pre-generated noise ensures identical results across runs
- **Unconditional Generation**: Simplified model without label conditioning
- **Modular Pipeline**: Clear separation between data prep, noise generation, and training
- **Configuration-Driven**: YAML config files for easy experiment management
- **WandB Integration**: Optional experiment tracking and logging

---

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Sufficient disk space for pre-generated noise (~200+ GB for full datasets)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/MarawanYakout/DDPM-SKY-CC.git
cd DDPM-SKY-CC
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare Your Dataset

Convert raw wind-speed images into a consolidated NumPy array:

```bash
python scripts/prepare_data.py \
  --folder_path ./Data/training_data \
  --sample_fraction 1.0 \
  --image_size 16 \
  --output_images wind_3D16X16.npy
```

**Parameters:**
- `folder_path`: Directory containing raw training images
- `sample_fraction`: Fraction of dataset to use (e.g., 0.25 for quick tests)
- `image_size`: Target image dimensions (e.g., 16 for 16×16 patches)
- `output_images`: Path for output `.npy` file

### 2. Pre-Generate Noise

Create deterministic noise tensors for all images and timesteps:

```bash
python scripts/pregenerate_noise.py \
  --images_np wind_3D16X16.npy \
  --timesteps 500 \
  --height 16 \
  --channels 3 \
  --save_dir ./pregenerated_noise \
  --images_per_file 1000
```

**Parameters:**
- `images_np`: Path to prepared dataset
- `timesteps`: Number of diffusion steps (typically 500)
- `height`: Image height (must match dataset)
- `channels`: Number of channels (3 for RGB)
- `save_dir`: Output directory for noise chunks
- `images_per_file`: Images per chunk file (for memory management)

**Note:** This step can take several hours and requires significant disk space.

### 3. Train the Model

Start training using either a config file or CLI arguments:

#### Using Config File (Recommended):

```bash
python scripts/train_cli.py \
  --config ./config/train.yaml \
  --pregenerated_noise_dir ./pregenerated_noise
```

#### Using CLI Arguments:

```bash
python scripts/train_cli.py \
  --data_np wind_3D16X16.npy \
  --pregenerated_noise_dir ./pregenerated_noise \
  --timesteps 500 \
  --epochs 250 \
  --batch_size 32 \
  --lr 0.0001 \
  --save_dir ./weights \
  --save_every 4
```

---

## Configuration

Use a YAML config file in `config/`:

- There are ones for 25% training testing 
- Full 100% testing for major computers

```yaml
dataset:
  npy_images: wind_3D16X16.npy

model:
  height: 16
  n_feat: 64

diffusion:
  timesteps: 500
  beta1: 0.0001
  beta2: 0.02

train:
  epochs: 250
  batch_size: 32
  lr: 0.0001
  save_dir: weights/
  save_every: 4

wandb:
  enabled: true
  project: DDPM-Wind
  group: unconditional
  name: run-v1
```

---

## Usage Examples

### Training with 25% of Data (Quick Test)

```bash
# Prepare subset
python scripts/prepare_data.py \
  --folder_path ./Data/training_data \
  --sample_fraction 0.25 \
  --image_size 16 \
  --output_images wind_3D16X16_25pct.npy

# Generate noise
python scripts/pregenerate_noise.py \
  --images_np wind_3D16X16_25pct.npy \
  --timesteps 500 \
  --height 16 \
  --channels 3 \
  --save_dir ./noise_25pct

# Train
python scripts/train_cli.py \
  --data_np wind_3D16X16_25pct.npy \
  --pregenerated_noise_dir ./noise_25pct \
  --epochs 50 \
  --batch_size 32
```

---

## Project Components

### Core Scripts

- **`scripts/prepare_data.py`** - Dataset preparation and preprocessing
- **`scripts/pregenerate_noise.py`** - Pre-generate deterministic noise tensors
- **`scripts/train_cli.py`** - Main training entry point

### Source Modules

- **`src_refactored/datasets.py`** - Custom dataset loader for images and pre-generated noise
- **`src_refactored/trainer.py`** - DDPM training loop implementation
- **`src_refactored/context_unet.py`** - U-Net model architecture
- **`src_refactored/diffusion.py`** - Diffusion schedule utilities

---

## Development Status

This project is actively under development and currently in **alpha testing phase**:

- ✅ Core training pipeline functional
- ✅ Pre-generated noise system working
- ✅ Basic configuration system implemented
- 🚧 API may change in future releases
- 🚧 Additional features and improvements planned
- ⚠️ Not recommended for production use yet

---

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit with clear messages (`git commit -m 'Add new feature'`)
5. Push to your branch (`git push origin feature/your-feature`)
6. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Original DDPM paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Forked from [tannisthamaiti/DiffusionModels_DDPM_DDIM](https://github.com/tannisthamaiti/DiffusionModels_DDPM_DDIM)

---

## Contact

- **Author**: Marawan Yakout
- **GitHub**: [@MarawanYakout](https://github.com/MarawanYakout)
- **Email**: mmyay1@student.london.ac.uk
- **LinkedIn**: [linkedin.com/in/marawanyakout](https://www.linkedin.com/in/marawanyakout)
- **YouTube**: [youtube.com/@centrified](https://www.youtube.com/@centrified)

---

**Note**: This is an experimental research project under active development. Feedback and suggestions are greatly appreciated!

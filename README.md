# DiffusionModels_DDPM_DDIM

Context‑conditioned diffusion models (DDPM) with a UNet backbone for small images, a clean training/sampling pipeline, and a reproducible data workflow. The project supports class conditioning (e.g., 5 wind‑speed bins), linear beta schedules, and Colab‑friendly CLIs.

This repository trains a denoising diffusion probabilistic model (DDPM) on small square RGB images (default 16×16) with optional class/context conditioning. The model predicts Gaussian noise at random timesteps; sampling runs the reverse process to synthesize images conditioned on a class label.

## Features

- Context‑aware UNet with residual blocks and GroupNorm

- Linear beta schedule with cached cumulative alphas

- Noise‑prediction loss and DDPM reverse sampler

- AugMix preprocessing and NumPy dataset wrappers

- Clean CLIs for data preparation, training, and sampling

- Works on CPU/GPU; optimized for Colab

## Resources:

- lucidrains denoising-diffusion-pytorch: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch

- Video intro to diffusion: https://www.youtube.com/watch?v=a4Yfz2FxXiY

# File tree:

DiffusionModels_DDPM_DDIM/
├─ config/ # config placeholders (future YAMLs)
├─ data/
│ └─ create_data.py # original preprocessing (reference) File -> create_data.py may need later
├─ experiments/ # experiment outputs (gitignored recommended)
├─ notebooks/ # exploratory notebooks
├─ scripts/
│ ├─ train_cli.py # CLI: train
│ └─ sample_cli.py # CLI: sample
├─ src_refactored/ # core library
│ ├─ augmentation.py # AugMix ops
│ ├─ context_unet.py # context‑conditioned UNet (time + class)
│ ├─ data.py # end‑to‑end dataset preparation
│ ├─ datasets.py # NumPy Dataset wrapper + transforms
│ ├─ diffusion.py # schedules, q(x_t|x0), loss
│ ├─ model.py # UNet building blocks
│ ├─ sampler.py # DDPM reverse sampler
│ ├─ seg_unet.py # optional ResNet UNet (auxiliary)
│ ├─ trainer.py # training loop + checkpointing
│ ├─ utils.py # augmentation utilities, label encoding
│ └─ vis.py # visualization
├─ tests/
│ ├─ smoke_test.py # minimal import/forward checks
│ └─ test_data_migration.py # AugMix + label encoder tests
├─ training/ # original scripts/notebooks (legacy reference)
│ ├─ Sampling_storm
│ ├─ sampling.py
│ └─ train.py
└─ requirements-colab.txt

# Checking code compiles correctly use the follwoing (Should Return No Error)

python -m py_compile src_refactored/context_unet.py ─╯
python -m py_compile src_refactored/diffusion.py
python -m py_compile src_refactored/sampler.py
python -m py_compile src_refactored/trainer.py
python -m py_compile scripts/train_cli.py
python -m py_compile scripts/sample_cli.py

# To delete Pycache files locally after your checkup (Optional):

## Run from repo root

'''sh
find . -type d -name "**pycache**" -exec rm -rf {} +
find . -name "\*.pyc" -delete
'''

Last Adsjusted 2 Nov 2025 by Marawan Yakout


# Research Goals So Far:

- Train the diffusion model so it can generate realistic images
- Put the context of the windspeed and timestamp.

- MDPI, IEEE
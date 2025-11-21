# Pre-generated Noise

This folder stores the pre-generated Gaussian noise used for DDPM-SKY-CC training.

## Contents

- Chunked noise files:
  - `noise_chunk_0000.npy`
  - `noise_chunk_0001.npy`
  - `noise_chunk_XXXX.npy`
- Metadata file:
  - `metadata.npy` with information such as:
    - Number of images
    - Number of timesteps
    - Image dimensions
    - Images per chunk

## Notes

- Noise files are created once by running `scripts/pregenerate_noise.py`.
- During training, the dataset and trainer load noise from this folder instead of generating it on-the-fly.
- This directory can be large and should be placed on fast storage if possible.

# Checkpoints

This folder stores model checkpoints saved during DDPM-SKY-CC training.

## Contents

- Model state dictionaries saved at configured intervals (for example every N epochs).
- Filenames typically include the epoch number, such as:
  - `model_epoch_0.pth`
  - `model_epoch_4.pth`
  - `model_epoch_100.pth`

## Notes

- Checkpoints can be used to resume training or run inference from a specific training stage.
- Files in this directory are generated automatically by the training script and may grow over time.

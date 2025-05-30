import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# --- AugMix Utils ---
def int_parameter(level, maxval):
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    return float(level) * maxval / 10.

def sample_level(n):
    return np.random.uniform(low=0.1, high=n)

def autocontrast(img):
    return ImageOps.autocontrast(img)

def rotate(img, level):
    return img.rotate(int_parameter(level, 30))

def shear_x(img, level):
    return img.transform(img.size, Image.AFFINE, (1, float_parameter(level, 0.3), 0, 0, 1, 0))

def shear_y(img, level):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, float_parameter(level, 0.3), 1, 0))

augmentations = [
    lambda x: x,
    autocontrast,
    lambda x: rotate(x, sample_level(3)),
    lambda x: shear_x(x, sample_level(3)),
    lambda x: shear_y(x, sample_level(3)),
]

def augmix(image, severity=3, width=3, depth=-1, alpha=1.):
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = torch.zeros_like(T.ToTensor()(image))
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = random.choice(augmentations)
            image_aug = op(image_aug)
        mix += ws[i] * T.ToTensor()(image_aug)

    mixed = (1 - m) * T.ToTensor()(image) + m * mix
    return mixed

# --- Batch AugMix and Save ---

# --- AugMix Parameters ---
batch_size = 1000
num_aug_per_image = 40

# Create output folder for augmented images
#image_folder = os.path.join(folder_path, "train")
augmented_folder = os.path.join("/kaggle/working", "augmented")
os.makedirs(augmented_folder, exist_ok=True)


# --- Build list of existing image paths from labels ---
image_paths = [
    os.path.join(image_folder, fname + ".jpg")
    for fname in labels['Image ID']
    if os.path.exists(os.path.join(image_folder, fname + ".jpg"))
]

print(f"Total original images: {len(image_paths)}")
print(f"Total expected AugMix images: {len(image_paths) * num_aug_per_image}")


# --- Augment in batches ---
for batch_start in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[batch_start: batch_start + batch_size]
    for img_path in batch_paths:
        try:
            img = Image.open(img_path).convert("RGB").resize((256, 256))
            base_filename = os.path.basename(img_path)
            for j in range(num_aug_per_image):
                aug = augmix(img)
                aug_filename = f"{base_filename.replace('.jpg','')}_augmix_{j}.jpg"
                aug_path = os.path.join(augmented_folder, aug_filename)
                ToPILImage()(aug).save(aug_path, "JPEG", quality=85)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("\nAll augmented images saved to:", augmented_folder)


# --- Batch AugMix and Save ---

# --- AugMix Parameters ---
batch_size = 1000
num_aug_per_image = 40

# Create output folder for augmented images
#image_folder = os.path.join(folder_path, "train")
augmented_folder = os.path.join("/kaggle/working", "augmented")
os.makedirs(augmented_folder, exist_ok=True)


# --- Build list of existing image paths from labels ---
image_paths = [
    os.path.join(image_folder, fname + ".jpg")
    for fname in labels['Image ID']
    if os.path.exists(os.path.join(image_folder, fname + ".jpg"))
]

print(f"Total original images: {len(image_paths)}")
print(f"Total expected AugMix images: {len(image_paths) * num_aug_per_image}")


# --- Augment in batches ---
for batch_start in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[batch_start: batch_start + batch_size]
    for img_path in batch_paths:
        try:
            img = Image.open(img_path).convert("RGB").resize((256, 256))
            base_filename = os.path.basename(img_path)
            for j in range(num_aug_per_image):
                aug = augmix(img)
                aug_filename = f"{base_filename.replace('.jpg','')}_augmix_{j}.jpg"
                aug_path = os.path.join(augmented_folder, aug_filename)
                ToPILImage()(aug).save(aug_path, "JPEG", quality=85)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("\nAll augmented images saved to:", augmented_folder)
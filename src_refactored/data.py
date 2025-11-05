"""
Dataset loading and preprocessing for wind speed classification.

In detail the following folder contains the following:
    - Dataset Metadata Loading 
    - Wind Speed Dictionary Creation 
    - AugMix Batch Processing 
    - Image Processing with Rotation 
    - Main Pipeline Function
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from src_refactored.utils import channel3, crop_center, labels_process
from src_refactored.augmentation import augmix


# ===========================
# Dataset Metadata Loading 
# ===========================

def load_metadata(folder_path="."):
    """
    Load training labels and metadata CSVs.
    
    Args:
        folder_path: Path to folder containing CSV files
    
    Returns:
        labels: DataFrame with image IDs and wind speeds
        meta DataFrame with additional metadata
    """
    train_file_path = os.path.join(folder_path, "train_labels.csv")
    train_file_path_meta = os.path.join(folder_path, "train_metadata.csv")
    labels = pd.read_csv(train_file_path)
    metadata = pd.read_csv(train_file_path_meta)
    return labels, metadata

# ==================================
# Wind Speed Dictionary Creation 
# =================================


def create_wind_speed_dict(folder_path, labels_df):
    """
    Create mapping from image IDs to wind speeds.
    
    Args:
        folder_path: Path to augmented images folder
        labels_df: Labels DataFrame
    
    Returns:
        wind_speeds: Dictionary {image_id: wind_speed}
    """
    folder_path_image = os.path.join(folder_path, "augmented/")
    
    # Get image filenames
    image_filenames = [
        f for f in os.listdir(folder_path_image)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    # Extract IDs
    image_ids_df = labels_df['Image ID'].astype(str).tolist()
    image_ids_files = [os.path.splitext(f)[0] for f in image_filenames]
    
    # Find common IDs
    common_ids = list(set(image_ids_df) & set(image_ids_files))
    unique_ids_df = list(set(image_ids_df) - set(image_ids_files))
    unique_ids_files = list(set(image_ids_files) - set(image_ids_df))
    
    print(f"Number of common Image IDs: {len(common_ids)}")
    print(f"Number of unique Image IDs in DataFrame: {len(unique_ids_df)}")
    print(f"Number of unique Image IDs in files: {len(unique_ids_files)}")
    
    # Create wind speed dictionary
    wind_speeds = {}
    for image_id in common_ids:
        wind_speed = labels_df.loc[
            labels_df['Image ID'] == image_id, 'Wind Speed'
        ].iloc[0]
        wind_speeds[image_id] = wind_speed
    
    return wind_speeds


# =========================
# AugMix Batch Processing 
# ========================

def generate_augmix_images(
    folder_path=".",
    labels_df=None,
    batch_size=1000,
    num_aug_per_image=4,
    output_folder="augmented"
):
    """
    Generate AugMix augmented images from training set.
    
    Args:
        folder_path: Root folder path
        labels_df: Labels DataFrame
        batch_size: Number of images to process per batch
        num_aug_per_image: Number of augmented versions per original
        output_folder: Output folder name
    
    Returns:
        augmented_folder: Path to folder with augmented images
    """
    image_folder = os.path.join(folder_path, "train")
    augmented_folder = os.path.join(folder_path, output_folder)
    os.makedirs(augmented_folder, exist_ok=True)
    
    # Build list of image paths
    image_paths = [
        os.path.join(image_folder, fname + ".jpg")
        for fname in labels_df['Image ID']
        if os.path.exists(os.path.join(image_folder, fname + ".jpg"))
    ]

    print(f"Total original images: {len(image_paths)}")
    print(f"Total expected AugMix images: {len(image_paths) * num_aug_per_image}")
    
    # Augment in batches
    for batch_start in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[batch_start: batch_start + batch_size]
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB").resize((256, 256))
                base_filename = os.path.basename(img_path)
                for j in range(num_aug_per_image):
                    aug = augmix(img)
                    aug_filename = f"{base_filename.replace('.jpg', '')}_augmix_{j}.jpg"
                    aug_path = os.path.join(augmented_folder, aug_filename)
                    ToPILImage()(aug).save(aug_path, "JPEG", quality=85)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print("\nAll augmented images saved to:", augmented_folder)

    return augmented_folder


# ==================================
# Image Processing with Rotation 
# ==================================

def process_images_two_sets(folder_path, wind_speeds, sample_fraction=0.1, size=16):
    """
    Process images with rotation augmentation and wind speed labels.
    
    Args:
        folder_path: Path to augmented images folder
        wind_speeds: Dictionary mapping image_id to wind_speed
        sample_fraction: Fraction of images to process (0.1 = 10%)
        size: Target image size (default 16x16)
    
    Returns:
        image_array: List of processed image arrays
        wind_speed_array: List of one-hot encoded labels
    """
    image_array = []
    wind_speed_array = []
    
    filenames = os.listdir(folder_path)
    num_files_to_process = int(len(filenames) * sample_fraction)
    
    # Random selection
    with tqdm(total=num_files_to_process, desc="Selecting files") as pbar_selection:
        selected_filenames = random.sample(filenames, num_files_to_process)
        pbar_selection.update(num_files_to_process)
    
    # Image processing
    with tqdm(total=num_files_to_process, desc="Processing images") as pbar_processing:
        for filename in selected_filenames:
            if filename.endswith(".jpg"):
                file_path = os.path.join(folder_path, filename)
                try:
                    image_id = os.path.splitext(filename)[0]
                    img = Image.open(file_path)

                    # Original image
                    cropped_image = crop_center(img, size, size)
                    original = channel3(cropped_image, size)
                    image_array.append(original)
                    wind_speed_array.append(labels_process(wind_speeds[image_id]))
                    
                    # Rotated image
                    angle = random.choice([90, 180, 270])
                    rotated_image = cropped_image.rotate(angle)
                    rotated = channel3(rotated_image, size)
                    image_array.append(rotated)
                    wind_speed_array.append(labels_process(wind_speeds[image_id]))
                    
                    img.close()
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                finally:
                    pbar_processing.update(1)
    
    return image_array, wind_speed_array

# ==============================
# Main Pipeline Function
# =============================

def prepare_wind_speed_dataset(
    folder_path=".",
    generate_augmix_data=False,
    sample_fraction=0.1,
    image_size=16,
    output_image_file="wind_3D16X16.npy",
    output_label_file="wind_label_3D16X16.npy"
):
    """
    Complete pipeline to prepare wind speed classification dataset.
    
    Args:
        folder_path: Root folder path
        generate_augmix_ Whether to generate AugMix images (slow)
        sample_fraction: Fraction of images to process
        image_size: Target image size
        output_image_file: Output numpy file for images
        output_label_file: Output numpy file for labels
    
    Returns:
        image_array: Processed images
        labels_array: One-hot encoded labels
    """
    # Load metadata
    labels, metadata = load_metadata(folder_path)
    
    # Generate AugMix images if requested
    if generate_augmix:
        augmented_folder = generate_augmix_images(folder_path, labels)
    else:
        augmented_folder = os.path.join(folder_path, "augmented/")

    
    # Create wind speed dictionary
    wind_speeds = create_wind_speed_dict(folder_path, labels)
    
    # Process images with rotation augmentation
    image_array, labels_array = process_images_two_sets(
        augmented_folder, wind_speeds, sample_fraction, image_size
    )
    
    # Save to numpy files
    np.save(output_image_file, image_array)
    np.save(output_label_file, labels_array)
    
    print(f"\nDataset saved:")
    print(f"  Images: {output_image_file} - Shape: {np.array(image_array).shape}")
    print(f"  Labels: {output_label_file} - Shape: {np.array(labels_array).shape}")
    
    return image_array, labels_array
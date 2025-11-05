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
import numpy as np
import pandas as pd
from PIL import Image

folder_path="."
train_file_path = os.path.join(folder_path, "train_labels.csv")
train_file_path_meta = os.path.join(folder_path, "train_metadata.csv")
labels = pd.read_csv(train_file_path)
metadata = pd.read_csv(train_file_path_meta)

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
num_aug_per_image = 4

# Create output folder for augmented images
image_folder = os.path.join(folder_path, "train")
augmented_folder = os.path.join(".", "augmented")
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

#------------------------------------------------------------------

def channel3(img, size):
    empty_3d_array = np.empty((size, size, 3))
    empty_3d_array[:,:,0]=np.array(img)
    empty_3d_array[:,:,1]=np.array(img)
    empty_3d_array[:,:,2]=np.array(img)
    return empty_3d_array

def channel1(img):
    empty_1d_array = np.empty((16, 16, 1))
    empty_1d_array[:,:,0]=np.array(img)
    return empty_1d_array

def crop_center(image, new_width, new_height):
    # Get the current dimensions of the image
    width, height = image.size

    # Calculate the coordinates for the crop
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    # Perform the crop
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

from tqdm import tqdm

def process_images(folder_path):
    image_array = []
    for filename in tqdm(os.listdir(folder_path), desc="Processing images"):
        if filename.endswith(".jpg"):
            file_path = os.path.join(folder_path, filename)
            size = 16     # changing the image size from 128 to 16
            try:
                img = Image.open(file_path)
                cropped_image = crop_center(img, size, size)  # Crop to 16 x 16
                image_array.append(channel3(cropped_image, size))
                img.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return image_array


folder_path_image= os.path.join(folder_path,"augmented/")
filenames = os.listdir(folder_path_image)
image_names = []  # Create an empty list to store image names
df =labels.copy()

image_filenames = [f for f in os.listdir(folder_path_image) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
# Extract the 'Image ID' column from the DataFrame
image_ids_df = df['Image ID'].astype(str).tolist()  # Convert to strings for comparison

# Convert image filenames to IDs by removing extensions
image_ids_files = [os.path.splitext(f)[0] for f in image_filenames]
# Find common and unique elements
common_ids = list(set(image_ids_df) & set(image_ids_files))
unique_ids_df = list(set(image_ids_df) - set(image_ids_files))
unique_ids_files = list(set(image_ids_files) - set(image_ids_df))

# Print the results
print(f"Number of common Image IDs: {len(common_ids)}")
print(f"Number of unique Image IDs in DataFrame: {len(unique_ids_df)}")
print(f"Number of unique Image IDs in files: {len(unique_ids_files)}")

wind_speeds = {}

# Iterate through common IDs and retrieve wind speeds
for image_id in common_ids:
    wind_speed = df.loc[df['Image ID'] == image_id, 'Wind Speed'].iloc[0]
    wind_speeds[image_id] = wind_speed

def labels_process(value_loc):
    storm_array=[]
    # name_to_find = image_id
    # result_loc = labels.loc[labels['Image ID'] == name_to_find, 'Wind Speed'].values
    # value_loc=result_loc[0]
    if ((value_loc>=15) & (value_loc<=45)):
        storm_array = switch_case(1)
    elif ((value_loc>45) & (value_loc<=80)):
        storm_array = switch_case(2)
    elif (value_loc>80 & value_loc<=110):
        storm_array = switch_case(3)
    elif (value_loc>110 & value_loc<=150):
        storm_array = switch_case(4)
    elif (value_loc>150 & value_loc<=190):
        storm_array = switch_case(5)
    return storm_array

def switch_case(argument):
    return {
        1: [1,0,0,0,0],
        2: [0,1,0,0,0],
        3: [0,0,1,0,0],
        4: [0,0,0,1,0],
        5: [0,0,0,0,1]
    }.get(argument, "Invalid option")


from tqdm import tqdm
import random

def process_images(folder_path,wind_speeds):
    image_array = []
    filenames = os.listdir(folder_path)
    num_files_to_process = int(len(filenames) * 0.1)  # Calculate 10%
    wind_speed_array=[]
    # Wrap selection process with tqdm
    with tqdm(total=num_files_to_process, desc="Selecting files") as pbar_selection:
        selected_filenames = random.sample(filenames, num_files_to_process)
        pbar_selection.update(num_files_to_process) # Update selection progress bar

    # Wrap processing loop with tqdm
    with tqdm(total=num_files_to_process, desc="Processing images") as pbar_processing:
        for filename in selected_filenames:
            if filename.endswith(".jpg"):
                file_path = os.path.join(folder_path, filename)
                size = 128
                try:
                    img = Image.open(file_path)
                    cropped_image = crop_center(img, size, size)
                    image_array.append(channel3(cropped_image, size))
                    wind_speed_array.append(labels_process(wind_speeds[image_id]))  # Add wind speed to list
                    img.close()
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                finally:
                    pbar_processing.update(1) # Update processing progress bar
    return image_array,wind_speed_array


def process_images_two_sets(folder_path, wind_speeds):
    image_array = []
    wind_speed_array = []

    filenames = os.listdir(folder_path)
    num_files_to_process = int(len(filenames) * 0.1)  # Process 10%

    # Random selection
    with tqdm(total=num_files_to_process, desc="Selecting files") as pbar_selection:
        selected_filenames = random.sample(filenames, num_files_to_process)
        pbar_selection.update(num_files_to_process)

    # Image processing
    with tqdm(total=num_files_to_process, desc="Processing images") as pbar_processing:
        for filename in selected_filenames:
            if filename.endswith(".jpg"):
                file_path = os.path.join(folder_path, filename)
                size = 16
                try:
                    image_id = os.path.splitext(filename)[0]  # Get ID from filename
                    img = Image.open(file_path)

                    # --- Original Image ---
                    cropped_image = crop_center(img, size, size)
                    original = channel3(cropped_image, size)
                    image_array.append(original)
                    wind_speed_array.append(labels_process(wind_speeds[image_id]))

                    # --- Rotated Image ---
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


image_array,labels_array= process_images_two_sets(folder_path_image,wind_speeds)

file_path_label = "wind_label_3D16X16.npy"
file_path = "wind_3D16X16.npy"
np.save(file_path, image_array)
np.save(file_path_label, labels_array)




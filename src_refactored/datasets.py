# datasets.py (PRE-GENERATED ONLY, NO LABELS)

"""
Purpose: Minimal dataset wrapper for pre-generated noise (no labels, no on-the-fly mode).

Key Features:
  • Returns only (image, noise) tuple
  • Requires pre-generated noise directory (no fallback)
  • Optimized for deterministic training
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os


class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for loading wind speed images with pre-generated noise.
    No labels. Pre-generated noise only.
    """
    def __init__(self, sfilename, pregenerated_noise_dir, transform):
        """
        Args:
            sfilename (str): Path to the .npy file containing images (sprites).
            pregenerated_noise_dir (str): Path to directory with pre-generated noise chunks.
            transform (callable): Torchvision transforms to apply to images.
        """
        # Load the entire dataset into memory (fast for <10GB datasets)
        self.sprites = np.load(sfilename)
        
        # Debug print to verify data dimensions upon initialization
        print(f"sprite shape: {self.sprites.shape}")  # Expected: (N, H, W, C)
        
        self.transform = transform
        self.sprites_shape = self.sprites.shape
        
        # Load noise metadata and validate
        self.pregenerated_noise_dir = pregenerated_noise_dir
        self._load_noise_metadata()
        
        print(f"✓ Dataset initialized with pre-generated noise")
    
    
    def _load_noise_metadata(self):
        """Load metadata about pre-generated noise structure."""
        metadata_path = os.path.join(self.pregenerated_noise_dir, 'metadata.npy')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}\n"
                f"Please run pregenerate_noise.py first to create noise files."
            )
        
        self.noise_metadata = np.load(metadata_path, allow_pickle=True).item()
        
        # Validate metadata matches dataset
        if self.noise_metadata['n_images'] != len(self.sprites):
            raise ValueError(
                f"Noise was generated for {self.noise_metadata['n_images']} images, "
                f"but dataset has {len(self.sprites)} images. "
                f"Please regenerate noise with correct n_images parameter."
            )
        
        self.images_per_file = self.noise_metadata['images_per_file']
        self.n_chunks = self.noise_metadata['n_chunks']
        self.timesteps = self.noise_metadata['timesteps']
        
        # Cache to avoid reloading same chunk repeatedly
        self.current_chunk_idx = -1
        self.current_chunk_data = None
        
        print(f"  Noise metadata loaded: {self.noise_metadata['n_images']} images, "
              f"{self.timesteps} timesteps, {self.n_chunks} chunks")


    def _get_noise_for_index(self, idx):
        """
        Retrieve pre-generated noise for a specific image index.
        Returns noise array of shape (timesteps, channels, height, height).
        """
        # Determine which chunk file this index belongs to
        chunk_idx = idx // self.images_per_file
        local_idx = idx % self.images_per_file
        
        # Load chunk if not already cached
        if chunk_idx != self.current_chunk_idx:
            chunk_path = os.path.join(
                self.pregenerated_noise_dir, 
                f'noise_chunk_{chunk_idx:04d}.npy'
            )
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(
                    f"Noise chunk not found: {chunk_path}\n"
                    f"Expected {self.n_chunks} chunks. Please verify noise generation completed."
                )
            
            self.current_chunk_data = np.load(chunk_path)
            self.current_chunk_idx = chunk_idx
        
        # Return noise for this specific image (all timesteps)
        return self.current_chunk_data[local_idx]


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.sprites)


    def __getitem__(self, idx):
        """
        Retrieves a single sample: (image, noise).
        
        Returns:
            image: Transformed image tensor (C, H, W)
            noise: Pre-generated noise (timesteps, C, H, W)
        """
        # 1. Get image and cast to float32
        image = self.sprites[idx].astype(np.float32) 
        
        # 2. Apply Transformations
        if self.transform:
            image = self.transform(image)
        
        # 3. Ensure output is strictly a FloatTensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        else:
            image = image.float()
        
        # 4. Get Pre-generated Noise
        noise = self._get_noise_for_index(idx)  # (timesteps, C, H, W)
        noise = torch.from_numpy(noise).float()
        
        return (image, noise)


    def getshapes(self):
        """Helper method to inspect dataset dimensions externally."""
        return self.sprites_shape



# Define the standard transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])





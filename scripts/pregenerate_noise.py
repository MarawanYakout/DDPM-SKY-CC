"""
Pre-generate all noise for DDPM training.
WARNING: This will create large amounts of data and take several hours.
"""
import numpy as np
import os
from tqdm import tqdm
import argparse


def pregenerate_noise(n_images, timesteps, height, channels, save_dir, images_per_file=1000):
    """
    Pre-generate noise and save in batched files.

    Args:
        n_images: number of images in dataset
        timesteps: number of diffusion timesteps
        height: image height/width
        channels: number of channels (3 for RGB)
        save_dir: directory to save noise files
        images_per_file: number of images per .npy file (to avoid huge files)
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Pre-generating Noise for DDPM Training")
    print("=" * 60)
    print(f"Dataset size: {n_images} images")
    print(f"Timesteps: {timesteps}")
    print(f"Image dimensions: {height}×{height}×{channels}")
    print(f"Images per file: {images_per_file}")

    # Calculate storage
    bytes_per_noise = height * height * channels * 4  # float32
    total_noise_values = n_images * timesteps
    total_bytes = total_noise_values * bytes_per_noise
    total_gb = total_bytes / (1024 ** 3)

    print(f"\nEstimated storage: {total_gb:.2f} GB")
    print("=" * 60)

    response = input("\nDo you want to continue? This will take hours! (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return

    print("\nGenerating noise...")

    # Generate in chunks to avoid memory issues
    n_chunks = (n_images + images_per_file - 1) // images_per_file

    for chunk_idx in tqdm(range(n_chunks), desc="Chunks"):
        start_idx = chunk_idx * images_per_file
        end_idx = min(start_idx + images_per_file, n_images)
        chunk_size = end_idx - start_idx

        # Generate noise for this chunk: (chunk_size, timesteps, channels, height, height)
        chunk_noise = np.random.randn(
            chunk_size, timesteps, channels, height, height
        ).astype(np.float32)

        # Save chunk
        filename = os.path.join(save_dir, f"noise_chunk_{chunk_idx:04d}.npy")
        np.save(filename, chunk_noise)

        # Print progress
        file_size_mb = os.path.getsize(filename) / (1024 ** 2)
        tqdm.write(
            f"Saved {filename} ({file_size_mb:.1f} MB) - Images {start_idx} to {end_idx-1}"
        )

    # Save metadata
    metadata = {
        "n_images": n_images,
        "timesteps": timesteps,
        "height": height,
        "channels": channels,
        "images_per_file": images_per_file,
        "n_chunks": n_chunks,
    }
    np.save(os.path.join(save_dir, "metadata.npy"), metadata)

    print(f"\n✓ Noise generation complete!")
    print(f"Total files: {n_chunks} chunks + 1 metadata file")
    print(f"Location: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Optional manual override (fallback)
    parser.add_argument("--n_images", type=int, default=None,
                        help="Number of images (used if --images_np is not provided)")
    parser.add_argument("--images_np", type=str, default=None,
                        help="Path to images .npy file; if set, n_images is inferred from this file")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--images_per_file", type=int, default=1000)

    args = parser.parse_args()

    # Infer n_images from images_np if provided
    if args.images_np is not None:
        print(f"Loading images from: {args.images_np}")
        # Use mmap_mode to avoid loading full array into RAM
        imgs = np.load(args.images_np, mmap_mode="r")
        n_images = imgs.shape[0]
        print(f"Inferred n_images from file: {n_images}")
    else:
        if args.n_images is None:
            raise ValueError(
                "You must provide either --images_np or --n_images."
            )
        n_images = args.n_images
        print(f"Using n_images from argument: {n_images}")

    pregenerate_noise(
        n_images=n_images,
        timesteps=args.timesteps,
        height=args.height,
        channels=args.channels,
        save_dir=args.save_dir,
        images_per_file=args.images_per_file,
    )

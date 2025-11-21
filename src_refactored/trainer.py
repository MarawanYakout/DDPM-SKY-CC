
# trainer.py (NO LABELS VERSION)
"""
DDPM Trainer for pre-generated noise without labels.

Key Changes:
  • Expects (image, noise) from dataloader (no labels)
  • Model called without context parameter
  • Simplified for unconditional generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os


class DDPMTrainer:
    def __init__(self, model, dataloader, optimizer, ab_t, timesteps, device, save_dir):
        """
        Args:
            model: The neural network (ContextUnet).
            dataloader: Returns (image, noise).
            optimizer: PyTorch optimizer.
            ab_t: Alpha-bar schedule tensor of shape (timesteps,).
            timesteps: Total diffusion steps.
            device: 'cuda' or 'cpu'.
            save_dir: Directory to save model checkpoints.
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.ab_t = ab_t
        self.timesteps = timesteps
        self.device = device
        self.save_dir = save_dir
        
        # Create save dir if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Trainer initialized with {timesteps} timesteps")
        print(f"Alpha-bar schedule shape: {ab_t.shape}")


    def perturb_input(self, x, t, noise):
        """
        Standard DDPM forward process: 
        x_t = sqrt(ab_t) * x_0 + sqrt(1 - ab_t) * noise
        
        Args:
            x: Clean images (B, C, H, W)
            t: Timestep indices (B,) - values from 0 to timesteps-1
            noise: Noise to add (B, C, H, W)
        
        Returns:
            Perturbed images (B, C, H, W)
        """
        # Get alpha_bar values for the selected timesteps
        ab_t_batch = self.ab_t[t]  # (B,)
        
        # Reshape for broadcasting: (B, 1, 1, 1)
        sqrt_ab_t = ab_t_batch.sqrt()[:, None, None, None]
        sqrt_one_minus_ab_t = (1 - ab_t_batch).sqrt()[:, None, None, None]
        
        return sqrt_ab_t * x + sqrt_one_minus_ab_t * noise


    def train_epoch(self, epoch_idx):
        """
        Trains the model for one full epoch.
        Expects dataloader to return 2 items: (x, noise)
        
        Args:
            epoch_idx: Current epoch number for logging
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch_idx}")
        total_loss = 0.0

        for batch_idx, (x, noise_all) in enumerate(pbar):
            self.optimizer.zero_grad()

            # Move everything to device
            x = x.to(self.device)              # (B, C, H, W)
            noise_all = noise_all.to(self.device)  # (B, timesteps, C, H, W)

            batch_size = x.shape[0]
            
            # 1. Sample Random Timesteps
            # Sample t uniformly from [0, timesteps-1] for 0-indexing
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
            
            # 2. Select Noise for Sampled Timesteps
            # Pre-generated noise: select the noise for each sampled timestep
            # noise_all is (B, timesteps, C, H, W)
            # We need noise[i, t[i], :, :, :] for each i in batch
            batch_indices = torch.arange(batch_size, device=self.device)
            noise = noise_all[batch_indices, t]  # (B, C, H, W)
            
            # 3. Perturb Input
            x_pert = self.perturb_input(x, t, noise)
            
            # 4. Predict Noise (NO CONTEXT/LABELS)
            # Normalize timestep to [0, 1] for model input
            t_norm = t.float() / self.timesteps
            pred_noise = self.model(x_pert, t_norm)  # Model should accept (x, t) only
            
            # 5. Calculate Loss
            loss = F.mse_loss(pred_noise, noise)
            
            # 6. Backpropagation
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.dataloader)
        return avg_loss


    def save_checkpoint(self, epoch):
        """
        Saves the current model state to disk.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Path to saved checkpoint
        """
        path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        return path

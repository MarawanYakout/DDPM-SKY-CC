"""
DDPM training loop with checkpointing.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from src_refactored.diffusion import compute_loss

class DDPMTrainer:
    def __init__(self, model, dataloader, optimizer, ab_t, timesteps, device, save_dir):
        """
        DDPM trainer.
        
        Args:
            model: ContextUnet model
            dataloader: training data loader
            optimizer: torch optimizer
            ab_t: cumulative alpha product schedule
            timesteps: number of diffusion steps
            device: torch device
            save_dir: checkpoint directory
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.ab_t = ab_t
        self.timesteps = timesteps
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        
        for x, c in pbar:
            x = x.to(self.device)
            c = c.to(self.device).float()
            
            # Sample random timesteps
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long()
            
            # Compute loss
            loss = compute_loss(self.model, x, t, c, self.ab_t)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.dataloader)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, f"model_{epoch}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"saved model at {path}")

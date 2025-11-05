"""
CLI script for training DDPM on wind speed data.
Usage: python scripts/train_cli.py --epochs 250 --save_every 4


"""
import argparse
import torch
from torch.utils.data import DataLoader
from src_refactored.context_unet import ContextUnet
from src_refactored.datasets import CustomDataset, transform
from src_refactored.diffusion import make_ddpm_schedule
from src_refactored.trainer import DDPMTrainer

def main():
    parser = argparse.ArgumentParser(description="Train DDPM diffusion model")
    parser.add_argument("--data_np", default="wind_3D16X16.npy", help="Path to data numpy file")
    parser.add_argument("--labels_np", default="wind_label_3D16X16.npy", help="Path to labels numpy file")
    parser.add_argument("--timesteps", type=int, default=500, help="Number of diffusion steps")
    parser.add_argument("--beta1", type=float, default=1e-4, help="Starting beta")
    parser.add_argument("--beta2", type=float, default=0.02, help="Ending beta")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", default="weights_16/", help="Checkpoint save directory")
    parser.add_argument("--height", type=int, default=16, help="Image height")
    parser.add_argument("--n_feat", type=int, default=64, help="Base feature dimension")
    parser.add_argument("--n_cfeat", type=int, default=5, help="Context feature dimension")
    parser.add_argument("--save_every", type=int, default=4, help="Save checkpoint every N epochs")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build schedule
    print("Building diffusion schedule...")
    b_t, a_t, ab_t = make_ddpm_schedule(args.timesteps, args.beta1, args.beta2, device)
    
    # Build model
    print("Building model...")
    model = ContextUnet(
        in_channels=3, 
        n_feat=args.n_feat, 
        n_cfeat=args.n_cfeat, 
        height=args.height
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build dataset
    print("Loading dataset...")
    dataset = CustomDataset(args.data_np, args.labels_np, transform, null_context=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    print(f"Dataset size: {len(dataset)}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Trainer
    trainer = DDPMTrainer(model, dataloader, optimizer, ab_t, args.timesteps, device, args.save_dir)
    
    # Train loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        avg_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            trainer.save_checkpoint(epoch)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()

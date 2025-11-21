
# train_cli.py (NO LABELS VERSION)

"""
CLI script for training DDPM with pre-generated noise only (no labels).

Key Changes:
  • No labels_np parameter
  • Dataset only requires images and noise directory
  • Model must support unconditional generation (no context parameter)
"""

import argparse
import yaml
import torch
import os
from torch.utils.data import DataLoader
from src_refactored.context_unet import ContextUnet
from src_refactored.datasets import CustomDataset, transform
from src_refactored.diffusion import make_ddpm_schedule
from src_refactored.trainer import DDPMTrainer

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable logging.")


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DDPM diffusion model (no labels)")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--data_np", type=str, help="Path to data numpy file")
    parser.add_argument("--pregenerated_noise_dir", type=str, required=True,
                        help="Directory with pre-generated noise files (REQUIRED)")
    parser.add_argument("--timesteps", type=int, help="Number of diffusion steps")
    parser.add_argument("--beta1", type=float, help="Starting beta")
    parser.add_argument("--beta2", type=float, help="Ending beta")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--save_dir", type=str, help="Checkpoint save directory")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--n_feat", type=int, help="Base feature dimension")
    parser.add_argument("--save_every", type=int, help="Save checkpoint every N epochs")

    args = parser.parse_args()
    
    # Set defaults
    config_defaults = {
        'data_np': 'wind_3D16X16.npy',
        'timesteps': 500,
        'beta1': 1e-4,
        'beta2': 0.02,
        'epochs': 250,
        'batch_size': 32,
        'lr': 1e-4,
        'save_dir': 'weights_16/',
        'height': 16,
        'n_feat': 64,
        'save_every': 4
    }
    
    # WandB config
    use_wandb = False
    wandb_config = {}
    
    # Load config file if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        full_cfg = load_config(args.config)
        
        # Map YAML structure to flat arguments
        config_defaults['data_np'] = full_cfg.get('dataset', {}).get('npy_images', config_defaults['data_np'])
        config_defaults['epochs'] = int(full_cfg.get('train', {}).get('epochs', config_defaults['epochs']))
        config_defaults['batch_size'] = int(full_cfg.get('train', {}).get('batch_size', config_defaults['batch_size']))
        config_defaults['lr'] = float(full_cfg.get('train', {}).get('lr', config_defaults['lr']))
        config_defaults['save_every'] = int(full_cfg.get('train', {}).get('save_every', config_defaults['save_every']))
        config_defaults['save_dir'] = full_cfg.get('train', {}).get('save_dir', config_defaults['save_dir'])
        config_defaults['timesteps'] = int(full_cfg.get('diffusion', {}).get('timesteps', config_defaults['timesteps']))
        config_defaults['beta1'] = float(full_cfg.get('diffusion', {}).get('beta1', config_defaults['beta1']))
        config_defaults['beta2'] = float(full_cfg.get('diffusion', {}).get('beta2', config_defaults['beta2']))
        config_defaults['height'] = int(full_cfg.get('model', {}).get('height', config_defaults['height']))
        config_defaults['n_feat'] = int(full_cfg.get('model', {}).get('n_feat', config_defaults['n_feat']))
        
        # Extract WandB config
        wandb_config = full_cfg.get('wandb', {})
        use_wandb = wandb_config.get('enabled', False) and WANDB_AVAILABLE

    # Override config with CLI arguments
    for key, default_val in config_defaults.items():
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            setattr(args, key, arg_val)
        else:
            setattr(args, key, default_val)
    
    # Validate required argument
    if not args.pregenerated_noise_dir:
        raise ValueError("--pregenerated_noise_dir is REQUIRED. This version only supports pre-generated noise.")
    
    # Print configuration
    print("\n" + "="*60)
    print("Training Configuration (NO LABELS, PRE-GENERATED NOISE ONLY)")
    print("="*60)
    print(f"Data file:         {args.data_np}")
    print(f"Noise directory:   {args.pregenerated_noise_dir}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Learning rate:     {args.lr}")
    print(f"Save every:        {args.save_every} epochs")
    print(f"Save dir:          {args.save_dir}")
    print(f"Timesteps:         {args.timesteps}")
    print(f"Beta range:        [{args.beta1}, {args.beta2}]")
    print(f"Model height:      {args.height}")
    print(f"n_feat:            {args.n_feat}")
    print(f"WandB enabled:     {use_wandb}")
    print("="*60 + "\n")
    
    # Initialize WandB
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'DDPM-SKYCC'),
            group=wandb_config.get('group', 'unconditional'),
            name=wandb_config.get('name', 'ddpm-no-labels'),
            config={
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'timesteps': args.timesteps,
                'beta1': args.beta1,
                'beta2': args.beta2,
                'height': args.height,
                'n_feat': args.n_feat,
                'save_every': args.save_every,
                'unconditional': True,
            }
        )
        print(f"✓ WandB initialized: {wandb.run.name}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Build diffusion schedule
    print("Building diffusion schedule...")
    b_t, a_t, ab_t = make_ddpm_schedule(args.timesteps, args.beta1, args.beta2, device)
    print(f"✓ Schedule built: ab_t shape = {ab_t.shape}\n")
    
    # Build model (UNCONDITIONAL - no context features)
    print("Building model...")
    # NOTE: You'll need to modify ContextUnet to support unconditional mode
    # or use a different model class. For now, passing n_cfeat=0
    model = ContextUnet(
        in_channels=3, 
        n_feat=args.n_feat, 
        n_cfeat=0,  # NO CONTEXT for unconditional generation
        height=args.height
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters\n")
    
    if use_wandb:
        wandb.config.update({'model_parameters': num_params})
    
    # Build dataset (NO LABELS)
    print("Loading dataset...")
    dataset = CustomDataset(
        args.data_np,
        args.pregenerated_noise_dir,
        transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"✓ Dataset loaded: {len(dataset)} images\n")
    
    if use_wandb:
        wandb.config.update({'dataset_size': len(dataset)})
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Build trainer
    print("Initializing trainer...")
    trainer = DDPMTrainer(
        model, 
        dataloader, 
        optimizer, 
        ab_t, 
        args.timesteps, 
        device, 
        args.save_dir
    )
    print("✓ Trainer initialized\n")
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    print("="*60 + "\n")
    
    for epoch in range(args.epochs):
        avg_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}/{args.epochs-1} | Loss: {avg_loss:.6f}")
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_loss,
                'learning_rate': args.lr
            })
        
        # Save checkpoints
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_path = trainer.save_checkpoint(epoch)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
            
            # Log checkpoint to WandB
            if use_wandb and os.path.exists(checkpoint_path):
                artifact = wandb.Artifact(f'model-epoch-{epoch}', type='model')
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
    
    # Finish
    if use_wandb:
        wandb.finish()
        print("\n✓ WandB run finished")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()

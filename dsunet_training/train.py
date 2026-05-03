"""
Train DSUNet on CARLA Lane Detection Dataset
"""

import os
import sys
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model.dsunet import DSUNet, count_parameters
from dataset.carla_lane_dataset import get_dataloaders
from utils.losses import CombinedLoss
from utils.metrics import compute_metrics


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    loss_components = {'dice': 0, 'bce': 0, 'focal': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss, loss_dict = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{loss_dict['dice']:.4f}"
        })
    
    # Average losses
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


def validate(model, val_loader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    loss_components = {'dice': 0, 'bce': 0, 'focal': 0}
    all_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
            
            # Compute metrics
            preds = torch.sigmoid(outputs) > 0.5
            metrics = compute_metrics(preds, masks)
            
            # Accumulate
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            for key in all_metrics:
                all_metrics[key] += metrics[key]
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{metrics['iou']:.4f}"
            })
    
    # Average
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    for key in all_metrics:
        all_metrics[key] /= num_batches
    
    return avg_loss, loss_components, all_metrics


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'logs')
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_height, args.image_width)
    )
    
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val:   {len(val_loader.dataset)} samples")
    print(f"Test:  {len(test_loader.dataset)} samples")
    
    # Create model
    print("\nCreating DSUNet model...")
    model = DSUNet(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Loss and optimizer
    criterion = CombinedLoss(
        dice_weight=0.5,
        bce_weight=0.3,
        focal_weight=0.2
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_loss_components, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/IoU', val_metrics['iou'], epoch)
        writer.add_scalar('Metrics/Dice', val_metrics['dice'], epoch)
        writer.add_scalar('Metrics/F1', val_metrics['f1'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val IoU:    {val_metrics['iou']:.4f}")
        print(f"Val Dice:   {val_metrics['dice']:.4f}")
        print(f"Val F1:     {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, output_dir / 'best_model_loss.pth')
            print(f"✅ Saved best model (loss: {val_loss:.4f})")
        
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, output_dir / 'best_model_iou.pth')
            print(f"✅ Saved best model (IoU: {val_metrics['iou']:.4f})")
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    print("\n✅ Training completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val IoU:  {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DSUNet on CARLA Lane Detection')
    
    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    
    # Model
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base channels for DSUNet')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Data loading
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image-height', type=int, default=480,
                        help='Image height')
    parser.add_argument('--image-width', type=int, default=640,
                        help='Image width')
    
    # Misc
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)

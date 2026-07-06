"""
Evaluate trained DSUNet model on test set
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from model.dsunet import DSUNet
from dataset.carla_lane_dataset import get_dataloaders
from utils.metrics import compute_metrics


def evaluate(model, test_loader, device, save_visualizations=False, output_dir=None):
    """Evaluate model on test set"""
    model.eval()
    
    all_metrics = {'iou': [], 'dice': [], 'precision': [], 'recall': [], 'f1': []}
    
    if save_visualizations and output_dir:
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            # Compute metrics for each sample
            for i in range(images.size(0)):
                metrics = compute_metrics(preds[i:i+1], masks[i:i+1])
                
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                
                # Save visualization
                if save_visualizations and output_dir and batch_idx < 10:
                    save_visualization(
                        images[i].cpu(),
                        masks[i].cpu(),
                        preds[i].cpu(),
                        filenames[i],
                        vis_dir
                    )
            
            # Update progress bar
            avg_iou = np.mean(all_metrics['iou'])
            avg_dice = np.mean(all_metrics['dice'])
            pbar.set_postfix({
                'IoU': f"{avg_iou:.4f}",
                'Dice': f"{avg_dice:.4f}"
            })
    
    # Compute statistics
    results = {}
    for key in all_metrics:
        values = all_metrics[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return results


def save_visualization(image, mask, pred, filename, output_dir):
    """Save visualization of prediction"""
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = (image * 255).clamp(0, 255).byte()
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert mask and pred to uint8
    mask = (mask.squeeze().numpy() * 255).astype(np.uint8)
    pred = (pred.squeeze().numpy() * 255).astype(np.uint8)
    
    # Create overlay
    overlay = image.copy()
    overlay[pred > 127] = [0, 255, 0]  # Green for prediction
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Create comparison
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    pred_colored = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    
    comparison = np.hstack([image, mask_colored, pred_colored, blended])
    
    # Save
    output_path = output_dir / f"vis_{filename}"
    cv2.imwrite(str(output_path), comparison)


def main(args):
    # Setup device (supports CUDA + ROCm/AMD)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from utils.device_utils import get_device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_height, args.image_width)
    )
    
    print(f"Test: {len(test_loader.dataset)} samples")
    
    # Create model
    print("\nLoading model...")
    model = DSUNet(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate(
        model,
        test_loader,
        device,
        save_visualizations=args.save_vis,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for metric, stats in results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_dir / 'test_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DSUNet on CARLA Lane Detection')
    
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                        help='Output directory')
    
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base channels for DSUNet')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image-height', type=int, default=480,
                        help='Image height')
    parser.add_argument('--image-width', type=int, default=640,
                        help='Image width')
    
    parser.add_argument('--save-vis', action='store_true',
                        help='Save visualizations')
    
    args = parser.parse_args()
    
    main(args)

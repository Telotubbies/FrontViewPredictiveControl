#!/usr/bin/env python3
"""
run_evaluation.py — วัดผล DSUNet model (IoU/Dice/Precision/Recall/F1)

ใช้:
  python scripts/run_evaluation.py --checkpoint model/lane_unet_final.pth --data-root dsunet_training/dataset/
  python scripts/run_evaluation.py --checkpoint model/best_model_iou.pth --data-root data/test/ --save-vis

Output: JSON สรุป metric + visualizations (ถ้า --save-vis)
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "dsunet_training"))
sys.path.insert(0, str(project_root / "src"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate DSUNet model on test set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="eval_results/latest",
                        help="Output directory for results")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--save-vis", action="store_true",
                        help="Save visualizations")
    args = parser.parse_args()

    # Import evaluation module
    from evaluate import evaluate, save_visualization

    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    from dataset.carla_lane_dataset import get_dataloaders
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_height, args.image_width),
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    from model.dsunet import DSUNet
    model = DSUNet(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Evaluate
    results = evaluate(
        model, test_loader, device,
        save_visualizations=args.save_vis,
        output_dir=args.output_dir,
    )

    # Print results
    print("\n" + "=" * 60)
    print("DSUNet EVALUATION RESULTS")
    print("=" * 60)
    for metric, stats in results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")

    # Save JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Baseline comparison
    baseline = {"iou": 0.861, "dice": 0.916}
    print("\n── Baseline Comparison ──")
    print(f"  IoU:  {results['iou']['mean']:.4f} vs baseline {baseline['iou']} "
          f"({'✅ OK' if results['iou']['mean'] >= baseline['iou'] else '⚠️ BELOW'})")
    print(f"  Dice: {results['dice']['mean']:.4f} vs baseline {baseline['dice']} "
          f"({'✅ OK' if results['dice']['mean'] >= baseline['dice'] else '⚠️ BELOW'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

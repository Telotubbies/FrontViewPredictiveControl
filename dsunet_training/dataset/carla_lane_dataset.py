"""
CARLA Lane Detection Dataset Loader

Loads images and masks from CARLA dataset with train/val/test split
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CARLALaneDataset(Dataset):
    """CARLA Lane Detection Dataset"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (480, 640),
        augment: bool = True
    ):
        """
        Args:
            data_root: Path to dataset root (contains images/, masks/, states.csv)
            split: 'train', 'val', or 'test'
            image_size: (height, width)
            augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Load image and mask paths
        self.image_dir = self.data_root / 'images'
        self.mask_dir = self.data_root / 'masks'
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        # Load states.csv if exists
        states_file = self.data_root / 'states.csv'
        if states_file.exists():
            self.states = pd.read_csv(states_file)
        else:
            self.states = None
        
        # Split dataset
        self._split_dataset()
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"Loaded {len(self.image_files)} samples for {split} split")
    
    def _split_dataset(self):
        """Split dataset into train/val/test (70/15/15)"""
        total = len(self.image_files)
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(total)
        
        # Split indices
        train_end = int(0.70 * total)
        val_end = int(0.85 * total)
        
        if self.split == 'train':
            indices = indices[:train_end]
        elif self.split == 'val':
            indices = indices[train_end:val_end]
        elif self.split == 'test':
            indices = indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Filter image files
        self.image_files = [self.image_files[i] for i in indices]
    
    def _setup_transforms(self):
        """Setup data augmentation and preprocessing"""
        
        if self.augment:
            # Training augmentation
            self.transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Validation/Test - no augmentation
            self.transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Add channel dimension to mask
        mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_path.name
        }


def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (480, 640)
):
    """Create train, val, and test dataloaders"""
    
    # Create datasets
    train_dataset = CARLALaneDataset(
        data_root=data_root,
        split='train',
        image_size=image_size,
        augment=True
    )
    
    val_dataset = CARLALaneDataset(
        data_root=data_root,
        split='val',
        image_size=image_size,
        augment=False
    )
    
    test_dataset = CARLALaneDataset(
        data_root=data_root,
        split='test',
        image_size=image_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    data_root = "/home/supawich/Desktop/CARLA_0.9.16/carla_lstm_mpc_project/data/lane_train_diverse_20260227_121538"
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=2
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Image: {batch['image'].shape}")
    print(f"  Mask:  {batch['mask'].shape}")

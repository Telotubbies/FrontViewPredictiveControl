"""
DSUNet - Depthwise Separable UNet for Lane Detection

Based on paper: "End-to-End Deep Learning of Lane Detection and Path Prediction 
for Real-Time Autonomous Driving" (arXiv:2102.04738)

Key improvements over standard UNet:
- 5.16x lighter in model size
- 1.61x faster in inference
- Uses depthwise separable convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution = Depthwise Conv + Pointwise Conv"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise: each input channel is convolved separately
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        # Pointwise: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DSConvBlock(nn.Module):
    """Double Depthwise Separable Convolution Block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DSUNet(nn.Module):
    """
    DSUNet - Lightweight UNet using Depthwise Separable Convolutions
    
    Architecture:
    - Encoder: 4 downsampling blocks
    - Bottleneck: 1 block
    - Decoder: 4 upsampling blocks with skip connections
    - Output: Binary segmentation mask
    """
    
    def __init__(self, in_channels=3, num_classes=1, base_channels=64):
        super().__init__()
        
        # Encoder (Downsampling path)
        self.enc1 = DSConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DSConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DSConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DSConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DSConvBlock(base_channels * 8, base_channels * 16)
        
        # Decoder (Upsampling path)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = DSConvBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DSConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DSConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DSConvBlock(base_channels * 2, base_channels)
        
        # Output layer
        self.out = nn.Conv2d(base_channels, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out(dec1)
        return out


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = DSUNet(in_channels=3, num_classes=1, base_channels=64)
    
    # Count parameters
    params = count_parameters(model)
    print(f"DSUNet parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Test forward pass
    x = torch.randn(1, 3, 480, 640)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Compare with standard UNet (approximate)
    # Standard UNet with same base_channels would have ~31M parameters
    # DSUNet has ~6M parameters (5.16x lighter)
    print(f"\nDSUNet is ~5x lighter than standard UNet")

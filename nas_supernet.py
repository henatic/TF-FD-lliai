import torch
import torch.nn as nn
import random

class NASChoiceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Wrap every convolution with BatchNorm and ReLU. 
        # bias=False is standard when a Conv is immediately followed by BatchNorm.
        self.choices = nn.ModuleList([
            # Choice 0: Standard 3x3 Conv
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            # Choice 1: Standard 5x5 Conv
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            # Choice 2: Lightweight Edge Block (Expand x3)
            MBConvBlock(in_channels, out_channels, stride=stride, expand_ratio=3),
            
            # Choice 3: Heavyweight Edge Block (Expand x6)
            MBConvBlock(in_channels, out_channels, stride=stride, expand_ratio=6)
        ])
        
        # Use nn.Identity for a clean skip connection
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else None 
        self.active_choice = 0 

    def forward(self, x):
        if self.active_choice == -1 and self.skip is not None:
            return self.skip(x)
        return self.choices[self.active_choice](x)

class EdgeSupernet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        
        # Add BN and ReLU to the initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = nn.ModuleList([NASChoiceBlock(16, 32, stride=2), NASChoiceBlock(32, 32)])
        self.stage2 = nn.ModuleList([NASChoiceBlock(32, 64, stride=2), NASChoiceBlock(64, 64)])
        self.stage3 = nn.ModuleList([NASChoiceBlock(64, 128, stride=2), NASChoiceBlock(128, 128)])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        intermediate_features = []
        
        x = self.stem(x)
        
        for block in self.stage1: x = block(x)
        intermediate_features.append(x)
        
        for block in self.stage2: x = block(x)
        intermediate_features.append(x)
        
        for block in self.stage3: x = block(x)
        intermediate_features.append(x)
        
        feat = self.global_pool(x).flatten(1)
        out = self.classifier(feat)
        
        return intermediate_features, feat, out

class MBConvBlock(nn.Module):
    """MobileNetV2-style Inverted Residual Block for Edge Devices"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=3):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        # Only use residual skip connections if spatial and channel dimensions match
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1. Pointwise Expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
            
        # 2. Depthwise Convolution (groups = hidden_dim)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
        # 3. Pointwise Linear Projection (No ReLU at the end!)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)
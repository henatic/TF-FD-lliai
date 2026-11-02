import torch
import torch.nn as nn
import random

class NASChoiceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Define the possible architectural choices for this specific block
        self.choices = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2),
            # Depthwise separable or MobileNet inverted residuals would go here
        ])
        # If stride > 1, we must project the skip connection to match dimensions
        self.skip = nn.Sequential() if stride == 1 and in_channels == out_channels else None 
        
        self.active_choice = 0 # Controlled by the NAS Controller

    def forward(self, x):
        if self.active_choice == -1 and self.skip is not None:
            return self.skip(x)
        return self.choices[self.active_choice](x)

class EdgeSupernet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Define the stages (where spatial resolution drops)
        self.stage1 = nn.ModuleList([NASChoiceBlock(16, 32, stride=2), NASChoiceBlock(32, 32)])
        self.stage2 = nn.ModuleList([NASChoiceBlock(32, 64, stride=2), NASChoiceBlock(64, 64)])
        self.stage3 = nn.ModuleList([NASChoiceBlock(64, 128, stride=2), NASChoiceBlock(128, 128)])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        intermediate_features = []
        
        x = self.stem(x)
        
        # Pass through stages and collect feature maps natively
        for block in self.stage1: x = block(x)
        intermediate_features.append(x)
        
        for block in self.stage2: x = block(x)
        intermediate_features.append(x)
        
        for block in self.stage3: x = block(x)
        intermediate_features.append(x)
        
        feat = self.global_pool(x).flatten(1)
        out = self.classifier(feat)
        
        # Perfectly matches your new dynamic unpacking syntax!
        return intermediate_features, feat, out
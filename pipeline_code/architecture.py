from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleUNet(nn.Module):
    """
    U-Net architecture with a pretrained ResNet18 encoder.
    This shared definition ensures consistency between training and inference.
    """
    
    def __init__(self):
        super().__init__()
        # Encoder (pretrained ResNet18)
        # We use weights=None first to instantiate, then load if needed, 
        # but standard practice for transfer learning is using the weights parameter.
        # Since 'pretrained' is deprecated, we use 'weights' if available, 
        # but for compatibility with the existing code's logic, we'll stick to the
        # structure seen in train_model.py which used 'pretrained=True' (older syntax)
        # or 'weights=ResNet18_Weights.DEFAULT' (newer).
        # Let's use the robust approach compatible with recent torchvision.
        
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except (ImportError, AttributeError):
            # Fallback for older torchvision
            resnet = models.resnet18(pretrained=True)

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        
        self._initialize_weights()
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 64 channels
        e2 = self.encoder2(e1)  # 64 channels
        e3 = self.encoder3(e2)  # 128 channels
        e4 = self.encoder4(e3)  # 256 channels
        
        # Decoder without skip connections (simplified)
        # Note: True U-Net has skip connections (concatenation). 
        # The previous 'SimpleUNet' in train_model.py didn't use them (comment says 'without skip connections').
        # We will preserve that logic for now to ensure we match the 'better' trained model structure 
        # from the training file, even if it's not a 'true' U-Net.
        
        d1 = self.up1(e4)  # 128
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)  # 64
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)  # 32
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)  # 16
        out = self.final(d4)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize final layer to output negative bias (for solar detection default=No)
        with torch.no_grad():
            nn.init.constant_(self.final.weight, 0.0)
            if self.final.bias is not None:
                nn.init.constant_(self.final.bias, -2.0)


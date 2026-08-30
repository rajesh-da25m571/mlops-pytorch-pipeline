#model.py
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights

class SEResidualBlock(nn.Module):
    def __init__(self, channels, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        if (self.use_se):
            self.se = SqueezeExcitation(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        if (self.use_se):
            out = self.se(out)
        out += residual
        return self.relu(out)

class SEResNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, use_se=False):
        super().__init__()
        
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res1 = SEResidualBlock(128,use_se)
        
        self.layer2_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.layer3_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res3 = SEResidualBlock(512, use_se)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_conv(x)
        x = self.res1(x)
        x = self.layer2_conv(x)
        x = self.layer3_conv(x)
        x = self.res3(x)
        x = self.pool(x)
        return self.fc(x)

def get_pipeline_model(architecture:str, dataset_name: str, num_classes: int = 10,in_channels: int = 3, use_se = False) -> nn.Module:
    """
    Factory engine maps targeted user requirements into the unified architecture.
    """
    name = dataset_name.lower()
    if name == "fashion_mnist":
        return SEResNet(in_channels=1, num_classes=num_classes, use_se=True)
    elif name == "cifar10":
        return SEResNet(in_channels=3, num_classes=num_classes, use_se=False)
    else:
        raise ValueError(f"Unsupported pipeline dataset choice: {dataset_name}")

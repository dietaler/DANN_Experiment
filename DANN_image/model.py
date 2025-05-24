import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32, 112, 112)

            # Conv block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (64, 56, 56)

            # Conv block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (128, 28, 28)

            # Conv block 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (128, 1, 1)
            
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.extractor(x) # 捲積後形狀: (batch, 128, 1, 1)
        x = x.view(x.size(0), -1) # flatten後形狀: (batch, 128)
        return x


class Classifier(nn.Module):
    def __init__(self, feat_dim=128, n_classes=6):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=n_classes),
        )

    def forward(self, x):
        return self.classifier(x)
    
class Discriminator(nn.Module):
    def __init__(self, feat_dim=128):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
        )

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        return self.discriminator(x)
 
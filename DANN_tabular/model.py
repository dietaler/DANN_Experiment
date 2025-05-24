import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF

class Extractor(nn.Module):
    def __init__(self, input_dim=63):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 讓輸出固定為 1 維
        )
        self.flatten_dim = 256  # 最終的 feature 維度

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x


    
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)  # 6 類別
        )

    def forward(self, x):
        return self.classifier(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 2 類別
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        return self.discriminator(reversed_input)

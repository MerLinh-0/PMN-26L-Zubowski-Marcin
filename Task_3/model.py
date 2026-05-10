import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, kernel_size=3, pool_type="max"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=kernel_size//2)

        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.feature_size = 32 * 8 * 8
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.feature_size = 32 * 8 * 8
        elif pool_type == "none":
            self.pool = nn.Identity()
            self.feature_size = 32 * 32 * 32
        else:
            raise ValueError(f"Nieznany typ pooling: {pool_type}")
        
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
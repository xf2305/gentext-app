import torch
import torch.nn as nn
import torch.nn.functional as F


# Part 1: CNN built to the assignment's spec
# (same style as SimpleCNN from Module 4 Practical 3)
class Assignment2CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Assignment2CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3x64x64  -> 16x32x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x32x32 -> 32x16x16
        x = x.view(-1, 32 * 16 * 16)          # flatten  -> 8192
        x = F.relu(self.fc1(x))               # -> 100
        x = self.fc2(x)                       # -> 10 (logits)
        return x


# Quick check: only runs when this file is run directly
if __name__ == "__main__":
    model = Assignment2CNN()
    print(model)

    fake_images = torch.randn(4, 3, 64, 64)  # 4 random 64x64 RGB "images"
    output = model(fake_images)
    print("Output shape:", output.shape)     # should be [4, 10]
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
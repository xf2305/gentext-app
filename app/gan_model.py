import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.fc = nn.Linear(noise_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            # 128 -> 64, 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 -> 1, 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)  
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1 -> 64, 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 128, 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    G = Generator()
    D = Discriminator()
    z = torch.randn(16, 100)
    fake = G(z)
    print("Generator output:", fake.shape)          
    print("Discriminator output:", D(fake).shape)   
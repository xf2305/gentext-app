from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import Generator, Discriminator

# ---------- Hyperparameters ----------
EPOCHS = 10
BATCH_SIZE = 128
NOISE_DIM = 100
LR = 0.0002

# ---------- Device ----------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"   # Mac 的 GPU
else:
    device = "cpu"
print("Using device:", device)

# ---------- Data: MNIST scaled to [-1, 1] to match the generator's Tanh ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------- Models, loss, optimizers ----------
G = Generator(NOISE_DIM).to(device)
D = Discriminator().to(device)
criterion = nn.BCEWithLogitsLoss()  # 判别器输出的是分数，这个损失内部会做 sigmoid
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, NOISE_DIM, device=device)  # 固定噪声，方便对比每个 epoch 的效果
Path("samples").mkdir(exist_ok=True)

# ---------- Training ----------
for epoch in range(1, EPOCHS + 1):
    for real, _ in loader:
        real = real.to(device)
        b = real.size(0)
        ones = torch.ones(b, 1, device=device)
        zeros = torch.zeros(b, 1, device=device)

        # 1) Train discriminator: real -> 1, fake -> 0
        z = torch.randn(b, NOISE_DIM, device=device)
        fake = G(z)
        loss_D = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # 2) Train generator: make D classify fakes as real
        loss_G = criterion(D(fake), ones)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch}/{EPOCHS}  loss_D={loss_D.item():.4f}  loss_G={loss_G.item():.4f}")

    # Save an 8x8 grid of samples each epoch
    G.eval()
    with torch.no_grad():
        samples = G(fixed_noise)
    save_image(samples * 0.5 + 0.5, f"samples/epoch_{epoch:02d}.png", nrow=8)
    G.train()

# ---------- Save generator weights (the API only needs the generator) ----------
torch.save(G.state_dict(), "app/gan_generator.pth")
print("Saved generator to app/gan_generator.pth")


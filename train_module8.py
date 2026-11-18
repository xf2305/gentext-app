# Module 8 Part 1: train the Diffusion model and the Energy-Based Model on MNIST
#   uv run python train_module8.py diffusion
#   uv run python train_module8.py ebm
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.module8_generators import (EnergyModel, Buffer, ebm_train_step, langevin_sample,
                                    DiffusionModel, EBM_WEIGHTS, DIFFUSION_WEIGHTS)

# ---------- Hyperparameters ----------
EPOCHS = 5
BATCH_SIZE = 128

# ---------- Device ----------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"   # Mac 的 GPU
else:
    device = "cpu"
print("Using device:", device)

# ---------- Data: MNIST padded to 32x32 and scaled to [-1, 1] ----------
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
Path("samples").mkdir(exist_ok=True)

which = sys.argv[1] if len(sys.argv) > 1 else "diffusion"

if which == "diffusion":
    model = DiffusionModel().to(device)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for real, _ in loader:
            total += model.train_step(real.to(device), optimizer)
        print(f"Epoch {epoch}/{EPOCHS}  loss={total / len(loader):.4f}")
        save_image(model.generate(64) * 0.5 + 0.5, f"samples/diffusion_epoch_{epoch:02d}.png", nrow=8)
    torch.save(model.cpu().state_dict(), DIFFUSION_WEIGHTS)
    print("Saved", DIFFUSION_WEIGHTS)

elif which == "ebm":
    model = EnergyModel().to(device)
    buffer = Buffer(model, device, BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for real, _ in loader:
            total += ebm_train_step(model, buffer, real.to(device), optimizer)
        print(f"Epoch {epoch}/{EPOCHS}  loss={total / len(loader):.4f}")
        start = torch.rand(64, 1, 32, 32, device=device) * 2 - 1
        samples = langevin_sample(model, start, steps=256, step_size=10, noise_std=0.005)
        save_image(samples * 0.5 + 0.5, f"samples/ebm_epoch_{epoch:02d}.png", nrow=8)
    torch.save(model.cpu().state_dict(), EBM_WEIGHTS)
    print("Saved", EBM_WEIGHTS)

else:
    print("Usage: uv run python train_module8.py [diffusion|ebm]")

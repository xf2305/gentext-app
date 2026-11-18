# Module 8 Part 1: Energy-Based Model + Diffusion Model digit generators
import copy
import io
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

EBM_WEIGHTS = Path(__file__).parent / "ebm.pth"
DIFFUSION_WEIGHTS = Path(__file__).parent / "diffusion.pth"


def to_png(images):
    """images in [-1, 1] -> PNG grid bytes"""
    images = (images * 0.5 + 0.5).clamp(0, 1).cpu()
    grid = make_grid(images, nrow=math.ceil(math.sqrt(images.size(0))), padding=2)
    buffer = io.BytesIO()
    to_pil_image(grid).save(buffer, format="PNG")
    return buffer.getvalue()


# =====================================================================
#  Energy-Based Model
# =====================================================================
def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """(N, 1, 32, 32) image -> (N, 1) energy. Low energy = looks real."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = swish(self.fc1(self.flatten(x)))
        return self.fc2(x)


def langevin_sample(model, imgs, steps, step_size, noise_std):
    """Gradient descent on the INPUT pixels (weights fixed) towards low energy."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        with torch.no_grad():
            imgs = (imgs + torch.randn_like(imgs) * noise_std).clamp(-1.0, 1.0)
        imgs.requires_grad_(True)
        energy = model(imgs)
        grads, = torch.autograd.grad(energy, imgs, grad_outputs=torch.ones_like(energy))
        with torch.no_grad():
            imgs = (imgs - step_size * grads.clamp(-0.03, 0.03)).clamp(-1.0, 1.0)
    for p in model.parameters():
        p.requires_grad_(True)
    return imgs.detach()


class Buffer:
    """Keeps old Langevin samples so we don't start from noise every step."""

    def __init__(self, model, device, batch_size=128):
        self.model, self.device, self.batch_size = model, device, batch_size
        self.examples = [torch.rand((1, 1, 32, 32), device=device) * 2 - 1 for _ in range(batch_size)]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(self.batch_size, 0.05)
        new_imgs = torch.rand((n_new, 1, 32, 32), device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.batch_size - n_new), dim=0)
        imgs = langevin_sample(self.model, torch.cat([new_imgs, old_imgs], dim=0), steps, step_size, noise)
        self.examples = (list(torch.split(imgs, 1, dim=0)) + self.examples)[:8192]
        return imgs


def ebm_train_step(model, buffer, real_imgs, optimizer, alpha=0.1, steps=60, step_size=10, noise=0.005):
    real_imgs = (real_imgs + torch.randn_like(real_imgs) * noise).clamp(-1.0, 1.0)
    fake_imgs = buffer.sample_new_exmps(steps, step_size, noise)
    model.train()
    out = model(torch.cat([real_imgs, fake_imgs], dim=0))
    real_out, fake_out = torch.split(out, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

    cdiv_loss = real_out.mean() - fake_out.mean()                        # contrastive divergence
    reg_loss = alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())  # keep energies small
    loss = cdiv_loss + reg_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    return loss.item()


# =====================================================================
#  Diffusion Model (UNet)
# =====================================================================
def sinusoidal_embedding(x, size=32):
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), size // 2, device=x.device))
    angles = 2.0 * math.pi * freqs.view(1, -1, 1, 1) * x
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)
        self.norm = nn.BatchNorm2d(c_in, affine=False)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)

    def forward(self, x):
        return self.conv2(F.silu(self.conv1(self.norm(x)))) + self.proj(x)


class UNet(nn.Module):
    """noisy image + noise variance -> predicted noise"""

    def __init__(self, widths=(32, 64, 96, 128)):
        super().__init__()
        self.init_conv = nn.Conv2d(1, widths[0], 1)
        self.upsample_noise = nn.Upsample(size=32, mode="nearest")

        c = widths[0] + 32
        self.down = nn.ModuleList()
        for w in widths[:-1]:                         # DownBlocks: 32 -> 16 -> 8 -> 4
            self.down.append(nn.ModuleList([ResidualBlock(c, w), ResidualBlock(w, w)]))
            c = w
        self.mid = nn.Sequential(ResidualBlock(c, widths[-1]), ResidualBlock(widths[-1], widths[-1]))
        c = widths[-1]
        self.up = nn.ModuleList()
        for w in reversed(widths[:-1]):               # UpBlocks: 4 -> 8 -> 16 -> 32
            self.up.append(nn.ModuleList([ResidualBlock(c + w, w), ResidualBlock(w + w, w)]))
            c = w
        self.final_conv = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.final_conv.weight)

    def forward(self, x, noise_var):
        emb = self.upsample_noise(sinusoidal_embedding(noise_var))
        x = torch.cat([self.init_conv(x), emb], dim=1)
        skips = []
        for blocks in self.down:
            for b in blocks:
                x = b(x)
                skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.mid(x)
        for blocks in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            for b in blocks:
                x = b(torch.cat([x, skips.pop()], dim=1))
        return self.final_conv(x)


def diffusion_schedule(t, min_signal=0.02, max_signal=0.95):
    """offset cosine schedule: noise_rate^2 + signal_rate^2 = 1"""
    start, end = math.acos(max_signal), math.acos(min_signal)
    angles = start + t * (end - start)
    return torch.sin(angles), torch.cos(angles)   # noise_rates, signal_rates


class DiffusionModel(nn.Module):
    def __init__(self, ema=0.999):
        super().__init__()
        self.ema = ema
        self.network = UNet()
        self.ema_network = copy.deepcopy(self.network)   # smoothed weights used for generating

    def denoise(self, noisy, noise_rates, signal_rates, training):
        net = self.network if training else self.ema_network
        net.train(training)
        pred_noise = net(noisy, noise_rates ** 2)
        pred_images = (noisy - noise_rates * pred_noise) / signal_rates
        return pred_noise, pred_images

    def train_step(self, images, optimizer):
        noise = torch.randn_like(images)
        t = torch.rand(images.size(0), 1, 1, 1, device=images.device)
        noise_rates, signal_rates = diffusion_schedule(t)
        noisy = signal_rates * images + noise_rates * noise          # corrupt the images

        pred_noise, _ = self.denoise(noisy, noise_rates, signal_rates, training=True)
        loss = F.l1_loss(pred_noise, noise)                          # learn to predict the noise

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for e, p in zip(self.ema_network.parameters(), self.network.parameters()):
                e.mul_(self.ema).add_(p, alpha=1 - self.ema)
            for e, b in zip(self.ema_network.buffers(), self.network.buffers()):
                e.copy_(b)
        return loss.item()

    @torch.no_grad()
    def generate(self, n, steps=20):
        """Start from Gaussian noise, predict the image, re-noise one step less, repeat."""
        device = next(self.parameters()).device
        current = torch.randn(n, 1, 32, 32, device=device)
        for i in range(steps):
            t = torch.full((n, 1, 1, 1), 1.0 - i / steps, device=device)
            noise_rates, signal_rates = diffusion_schedule(t)
            pred_noise, pred_images = self.denoise(current, noise_rates, signal_rates, training=False)
            next_noise, next_signal = diffusion_schedule(t - 1.0 / steps)
            current = next_signal * pred_images + next_noise * pred_noise
        return pred_images.clamp(-1, 1)


# =====================================================================
#  Generators used by the API (same idea as DigitGenerator)
# =====================================================================
class EBMGenerator:
    def __init__(self):
        self.model = EnergyModel()
        if EBM_WEIGHTS.exists():
            self.model.load_state_dict(torch.load(EBM_WEIGHTS, map_location="cpu"))

    def generate(self, n=16, steps=256):
        start = torch.rand(n, 1, 32, 32) * 2 - 1
        return to_png(langevin_sample(self.model, start, steps, step_size=10, noise_std=0.005))


class DiffusionGenerator:
    def __init__(self):
        self.model = DiffusionModel()
        if DIFFUSION_WEIGHTS.exists():
            self.model.load_state_dict(torch.load(DIFFUSION_WEIGHTS, map_location="cpu"))

    def generate(self, n=16, steps=20):
        return to_png(self.model.generate(n, steps))



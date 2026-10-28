# Assignment 3 Part 2: load the trained GAN generator and produce handwritten digits
import io
import math
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from app.gan_model import Generator

WEIGHTS_PATH = Path(__file__).parent / "gan_generator.pth"


class DigitGenerator:
    def __init__(self, weights_path: Path = WEIGHTS_PATH, noise_dim: int = 100):
        self.noise_dim = noise_dim
        self.model = Generator(noise_dim)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def generate(self, n: int = 16) -> bytes:
        """Generate n digits and return them as a PNG grid (bytes)."""
        with torch.no_grad():
            images = self.model(torch.randn(n, self.noise_dim))
        images = images * 0.5 + 0.5  # map Tanh output [-1, 1] back to [0, 1]
        grid = make_grid(images, nrow=math.ceil(math.sqrt(n)), padding=2)

        buffer = io.BytesIO()
        to_pil_image(grid.clamp(0, 1)).save(buffer, format="PNG")
        return buffer.getvalue()
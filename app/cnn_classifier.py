# Assignment 2 Part 2: load the trained CNN and classify a single image
import io
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.cnn_model import Assignment2CNN

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

WEIGHTS_PATH = Path(__file__).parent / "cnn_weights.pth"


class CNNClassifier:
    def __init__(self, weights_path=WEIGHTS_PATH):
        self.device = torch.device("cpu")
        self.model = Assignment2CNN()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()  # evaluation mode (see Part 3, Question 6)

        # Same preprocessing as in training: resize to 64x64, then to tensor
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(image).unsqueeze(0)  # add batch dimension: 1 x 3 x 64 x 64

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        idx = int(probs.argmax())
        return {
            "class_id": idx,
            "class_name": CLASSES[idx],
            "confidence": round(float(probs[idx]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        }
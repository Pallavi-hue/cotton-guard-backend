"""
Cotton Guard - Stage 2 Disease Classifier Service.
Executes EfficientNet-B0 seven-class classification.
ONLY called when Stage 1 Cotton Verifier accepts the image.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

_CURRENT_DIR = Path(__file__).resolve().parent
if (_CURRENT_DIR.parent / "models" / "disease_classifier" / "best_checkpoint.pth").exists():
    BASE_DIR = _CURRENT_DIR.parent
elif (_CURRENT_DIR.parent.parent / "models" / "disease_classifier" / "best_checkpoint.pth").exists():
    BASE_DIR = _CURRENT_DIR.parent.parent
else:
    BASE_DIR = _CURRENT_DIR.parent

MODEL_PATH = BASE_DIR / "models" / "disease_classifier" / "best_checkpoint.pth"


class DiseaseClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        if not model_path.exists():
            raise FileNotFoundError(f"Disease classifier model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.classes = checkpoint["classes"]
        self.image_size = checkpoint["image_size"]

        # EfficientNet-B0 base preprocessing
        weights = models.EfficientNet_B0_Weights.DEFAULT
        preprocess = weights.transforms()
        self.mean = preprocess.mean
        self.std = preprocess.std

        # Build architecture
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(self.classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def predict(self, pil_image: Image.Image) -> dict:
        """
        Classify a verified cotton leaf image into one of 7 conditions.
        Returns predicted disease, confidence, and all class probabilities.
        """
        tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        predicted_idx = int(torch.tensor(probs).argmax().item())
        predicted_class = self.classes[predicted_idx]
        confidence = round(float(probs[predicted_idx]), 4)

        all_probs = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes, probs)
        }

        return {
            "condition": predicted_class,
            "confidence": confidence,
            "is_healthy": (predicted_class == "Healthy Leaf"),
            "class_probabilities": all_probs,
        }

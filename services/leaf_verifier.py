"""
Cotton Guard - Stage 1 Leaf-Content Verification Service.
Enforces leaf-content validation before Cotton-Leaf Verifier.

Architecture:
1. Photometric Safeguards:
   - Skin & face chromaticity check (YCbCr + RGB)
   - High-contrast document / paper page check
   - Low-variance / uniform color / blank frame check
2. Deep Leaf-Content Neural Classifier:
   - MobileNetV3-Small binary classifier (leaf=1, non_leaf=0)
   - Calibrated decision boundary: tau_leaf from leaf_threshold_calibration.json
"""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

_CURRENT_DIR = Path(__file__).resolve().parent
if (_CURRENT_DIR.parent / "models" / "leaf_verifier" / "best_checkpoint.pth").exists():
    BASE_DIR = _CURRENT_DIR.parent
elif (_CURRENT_DIR.parent.parent / "models" / "leaf_verifier" / "best_checkpoint.pth").exists():
    BASE_DIR = _CURRENT_DIR.parent.parent
else:
    BASE_DIR = _CURRENT_DIR.parent

MODEL_PATH = BASE_DIR / "models" / "leaf_verifier" / "best_checkpoint.pth"
CALIBRATION_PATH = BASE_DIR / "models" / "leaf_verifier" / "leaf_threshold_calibration.json"


def detect_photometric_non_leaf(pil_image: Image.Image) -> tuple[bool, str | None, str | None]:
    """
    Fast photometric & structural heuristics to catch obvious non-leaf inputs:
    - Human faces / skin
    - White documents / printed text / receipts
    - Blank / uniform / flat color images
    Returns (is_non_leaf, rejection_reason, message)
    """
    # Downsample for fast photometric calculation
    thumb = pil_image.convert("RGB").resize((112, 112))
    arr = np.asarray(thumb, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    total_pixels = 112 * 112

    # 1. Uniform / Blank check
    std_total = float(np.std(arr))
    if std_total < 12.0:
        return True, "INSUFFICIENT_TEXTURE", "Image lacks recognizable leaf texture or structure. Please capture a clear, focused photograph of a leaf."

    # 2. Plant / Botanical foliage index (Excess Green: 2G - R - B)
    excess_green = (2.0 * g - r - b)
    green_mask = (excess_green > 10.0)
    green_ratio = float(np.count_nonzero(green_mask) / total_pixels)

    # 3. Human Skin Chromaticity Check (YCbCr & RGB rule-based)
    # Standard digital skin model (Fitzpatrick I-VI)
    # Y = 0.299R + 0.587G + 0.114B
    # Cb = 128 - 0.168736R - 0.331264G + 0.5B
    # Cr = 128 + 0.5R - 0.418688G - 0.081312B
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    skin_mask = (
        (cb >= 77.0) & (cb <= 127.0) &
        (cr >= 133.0) & (cr <= 173.0) &
        (r > 80.0) & (g > 40.0) & (b > 20.0) &
        (r > g) & (r > b) & ((r - g) >= 12.0)
    )
    skin_ratio = float(np.count_nonzero(skin_mask) / total_pixels)

    # If significant skin tone and negligible plant foliage:
    if skin_ratio > 0.20 and green_ratio < 0.10:
        return True, "FACE_OR_HUMAN_DETECTED", "Human face or person detected. Cotton Guard only diagnoses cotton leaves. Please photograph a cotton leaf."

    # 4. Document / Paper Receipt Check
    # High white/gray background and low saturation
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = max_c - min_c

    paper_mask = (r > 185.0) & (g > 185.0) & (b > 185.0) & (saturation < 25.0)
    paper_ratio = float(np.count_nonzero(paper_mask) / total_pixels)

    if paper_ratio > 0.60 and green_ratio < 0.05:
        return True, "DOCUMENT_DETECTED", "Document or printed text detected. Cotton Guard only diagnoses cotton leaves. Please photograph a cotton leaf."

    return False, None, None


class LeafContentVerifier:
    def __init__(self, model_path: Path = MODEL_PATH, calibration_path: Path = CALIBRATION_PATH, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if not model_path.exists():
            raise FileNotFoundError(f"Leaf verifier model checkpoint not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.image_size = checkpoint.get("image_size", 224)
        self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
        self.std = checkpoint.get("std", [0.229, 0.224, 0.225])

        # Load calibrated threshold
        if calibration_path.exists():
            cal = json.loads(calibration_path.read_text(encoding="utf-8"))
            self.tau_leaf = float(cal.get("tau_leaf", 0.745))
        else:
            self.tau_leaf = 0.745

        # Build architecture
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)
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
        Verify if an RGB PIL image contains a recognizable plant leaf.
        Executes fast photometric guards followed by deep neural leaf classification.
        """
        # 1. Photometric structural check
        is_photometric_non_leaf, reason, msg = detect_photometric_non_leaf(pil_image)
        if is_photometric_non_leaf:
            return {
                "is_leaf": False,
                "decision": "REJECTED_NON_LEAF",
                "leaf_probability": 0.0,
                "threshold": self.tau_leaf,
                "rejection_reason": reason,
                "message": msg,
            }

        # 2. Deep neural classification
        tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor).squeeze(-1).item()
            prob = 1.0 / (1.0 + torch.exp(torch.tensor(-logit)).item())

        prob = round(float(prob), 4)

        if prob >= self.tau_leaf:
            return {
                "is_leaf": True,
                "decision": "ACCEPTED_LEAF",
                "leaf_probability": prob,
                "threshold": self.tau_leaf,
                "rejection_reason": None,
                "message": "Recognizable plant leaf verified.",
            }
        else:
            return {
                "is_leaf": False,
                "decision": "REJECTED_NON_LEAF",
                "leaf_probability": prob,
                "threshold": self.tau_leaf,
                "rejection_reason": "NON_LEAF_IMAGE_DETECTED",
                "message": "NOT A LEAF. A recognizable plant leaf was not detected in this image. Cotton Guard only diagnoses cotton leaves. Please provide a clear, focused photograph of a cotton leaf.",
            }

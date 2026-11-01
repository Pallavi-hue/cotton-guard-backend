"""
Cotton Guard - Stage 2 Cotton Verifier Service.

Supports both:
1. Candidate V3: 3-Class Multi-Class Domain-Supervised Verifier (MCD-Verifier) with tree-foliage veto.
   - Class 0: Cotton leaves (field & indoor)
   - Class 1: Herbaceous crop leaves
   - Class 2: Tree foliage
2. Legacy Production: Binary single-logit MobileNetV3-Small verifier.
"""

from pathlib import Path
import json
import logging
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

logger = logging.getLogger("cotton_guard.verifier")

_CURRENT_DIR = Path(__file__).resolve().parent
if (_CURRENT_DIR.parent / "models" / "cotton_verifier_candidate_v4" / "best_checkpoint.pth").exists():
    BASE_DIR = _CURRENT_DIR.parent
elif (_CURRENT_DIR.parent.parent / "models" / "cotton_verifier_candidate_v4" / "best_checkpoint.pth").exists():
    BASE_DIR = _CURRENT_DIR.parent.parent
else:
    BASE_DIR = _CURRENT_DIR.parent

# Checkpoint & Calibration Paths
CANDIDATE_V4_MODEL_PATH = BASE_DIR / "models" / "cotton_verifier_candidate_v4" / "best_checkpoint.pth"
CANDIDATE_V4_CALIBRATION_PATH = BASE_DIR / "models" / "cotton_verifier_candidate_v4" / "threshold_calibration.json"
CANDIDATE_V4_CLASS_MAPPING_PATH = BASE_DIR / "models" / "cotton_verifier_candidate_v4" / "class_mapping.json"

CANDIDATE_V3_MODEL_PATH = BASE_DIR / "models" / "cotton_verifier_candidate_v3" / "best_checkpoint.pth"
CANDIDATE_V3_CALIBRATION_PATH = BASE_DIR / "models" / "cotton_verifier_candidate_v3" / "threshold_calibration.json"
CANDIDATE_V3_CLASS_MAPPING_PATH = BASE_DIR / "models" / "cotton_verifier_candidate_v3" / "class_mapping.json"

LEGACY_MODEL_PATH = BASE_DIR / "models" / "cotton_verifier" / "best_checkpoint.pth"
LEGACY_CALIBRATION_PATH = BASE_DIR / "models" / "cotton_verifier" / "threshold_calibration.json"

# Resolve default active model path (environment variable > Candidate V4 > Candidate V3 > Legacy)
ENV_MODEL_PATH = os.environ.get("COTTON_VERIFIER_MODEL_PATH")
ENV_CALIBRATION_PATH = os.environ.get("COTTON_VERIFIER_CALIBRATION_PATH")

if ENV_MODEL_PATH:
    DEFAULT_MODEL_PATH = Path(ENV_MODEL_PATH)
elif CANDIDATE_V4_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = CANDIDATE_V4_MODEL_PATH
elif CANDIDATE_V3_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = CANDIDATE_V3_MODEL_PATH
else:
    DEFAULT_MODEL_PATH = LEGACY_MODEL_PATH

if ENV_CALIBRATION_PATH:
    DEFAULT_CALIBRATION_PATH = Path(ENV_CALIBRATION_PATH)
elif CANDIDATE_V4_MODEL_PATH.exists() and CANDIDATE_V4_CALIBRATION_PATH.exists():
    DEFAULT_CALIBRATION_PATH = CANDIDATE_V4_CALIBRATION_PATH
elif CANDIDATE_V3_MODEL_PATH.exists() and CANDIDATE_V3_CALIBRATION_PATH.exists():
    DEFAULT_CALIBRATION_PATH = CANDIDATE_V3_CALIBRATION_PATH
else:
    DEFAULT_CALIBRATION_PATH = LEGACY_CALIBRATION_PATH

MODEL_PATH = DEFAULT_MODEL_PATH
CALIBRATION_PATH = DEFAULT_CALIBRATION_PATH


class CenterSquareCrop:
    """
    Symmetrically crops the largest centered square from a PIL Image.
    Preserves original physical aspect ratio without distortion or anisotropic squash.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return img.crop((left, top, left + min_dim, top + min_dim))


class CottonVerifier:
    """
    Cotton Verifier Service with automatic architecture detection.
    Dynamically loads either Candidate V3 (3-class MCD) or Legacy (binary) checkpoints.
    Supports validated center-square crop preprocessing with direct-resize rollback option.
    """

    def __init__(
        self,
        model_path: Path = None,
        calibration_path: Path = None,
        device: str = None,
        preprocessing_mode: str = "center_crop",
    ):
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.calibration_path = Path(calibration_path) if calibration_path is not None else DEFAULT_CALIBRATION_PATH
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Validate model file existence
        if not self.model_path.exists():
            raise FileNotFoundError(f"Cotton verifier model not found at {self.model_path}")

        # 2. Load checkpoint safely
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint at {self.model_path}: {e}") from e

        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid model checkpoint at {self.model_path}: missing 'model_state_dict'")

        state_dict = checkpoint["model_state_dict"]
        self.image_size = checkpoint.get("image_size", 224)
        self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
        self.std = checkpoint.get("std", [0.229, 0.224, 0.225])
        self.architecture = checkpoint.get("architecture", "mobilenet_v3_small")

        # 3. Detect number of output classes
        if "classifier.3.weight" in state_dict:
            out_features = state_dict["classifier.3.weight"].shape[0]
        else:
            out_features = checkpoint.get("num_classes", 1)

        self.num_classes = out_features

        # 4. Validate architecture and class mapping
        if self.num_classes == 3:
            self._init_three_class(checkpoint)
        elif self.num_classes == 1:
            self._init_binary(checkpoint)
        else:
            raise ValueError(f"Unsupported number of output classes: {self.num_classes} in {self.model_path}")

        # 5. Build backbone and load weights
        try:
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, self.num_classes)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model weights from {self.model_path}: {e}") from e

        # 6. Build input transform (center_crop validated default, direct_resize rollback option)
        self.preprocessing_mode = preprocessing_mode
        self.transform = self._build_transform(self.preprocessing_mode)

        logger.info(
            "CottonVerifier initialized: architecture=%s, classes=%d, tau_cotton=%.2f, tau_tree_veto=%s, preproc=%s, device=%s",
            self.architecture,
            self.num_classes,
            self.tau_accept,
            str(self.tau_tree_veto),
            self.preprocessing_mode,
            self.device,
        )

    def _build_transform(self, mode: str) -> transforms.Compose:
        """
        Builds the input preprocessing transform for Cotton Verifier.
        - 'center_crop': Aspect-ratio preserving square crop + resize (Phase 2 validated)
        - 'direct_resize': Legacy anisotropic direct resize (rollback option)
        """
        if mode == "center_crop":
            return transforms.Compose([
                CenterSquareCrop(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        elif mode == "direct_resize":
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}. Supported modes: 'center_crop', 'direct_resize'")

    def set_preprocessing_mode(self, mode: str):
        """Allow dynamic switching between 'center_crop' and 'direct_resize' for A/B testing or rollback."""
        self.transform = self._build_transform(mode)
        self.preprocessing_mode = mode
        logger.info("CottonVerifier preprocessing mode updated to: %s", mode)

    def get_preprocessing_mode(self) -> str:
        """Return the currently active preprocessing mode."""
        return self.preprocessing_mode

    def _init_three_class(self, checkpoint: dict):
        """Initialize thresholds and mapping for Candidate V3 3-class MCD verifier."""
        # Class mapping validation
        class_mapping = checkpoint.get("class_mapping")
        if class_mapping is None and CANDIDATE_V3_CLASS_MAPPING_PATH.exists():
            try:
                class_mapping = json.loads(CANDIDATE_V3_CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
            except Exception:
                class_mapping = None

        if class_mapping is None:
            raise ValueError(f"3-class verifier requires 'class_mapping' metadata in checkpoint or at {CANDIDATE_V3_CLASS_MAPPING_PATH}")

        # Normalize keys (handle both integer and string keys)
        normalized_mapping = {int(k): str(v) for k, v in class_mapping.items()}
        if (
            normalized_mapping.get(0) != "cotton"
            or normalized_mapping.get(1) != "herbaceous_crop"
            or normalized_mapping.get(2) != "tree_foliage"
        ):
            raise ValueError(f"Invalid class mapping order for 3-class verifier: {normalized_mapping}")

        self.class_mapping = normalized_mapping

        # Load thresholds from calibration file if present
        # Defaults use the validation-calibrated operating point:
        # tau_cotton: 0.75, tau_tree_veto: 0.15, tau_reject: 0.25, tau_tree_hard_reject: 0.50
        self.tau_cotton = 0.75
        self.tau_tree_veto = 0.15
        self.tau_reject = 0.25
        self.tau_tree_hard_reject = 0.50

        if self.calibration_path and self.calibration_path.exists():
            try:
                cal = json.loads(self.calibration_path.read_text(encoding="utf-8"))
                val_cal = cal.get("validation_calibrated_thresholds", {})
                if val_cal:
                    self.tau_cotton = float(val_cal.get("tau_cotton", 0.75))
                    self.tau_tree_veto = float(val_cal.get("tau_tree_veto", 0.15))
                    self.tau_reject = float(val_cal.get("tau_reject", 0.25))
                elif "tau_cotton" in cal:
                    self.tau_cotton = float(cal["tau_cotton"])
                    self.tau_tree_veto = float(cal.get("tau_tree_veto", 0.15))
                    self.tau_reject = float(cal.get("tau_reject", 0.25))
            except Exception as e:
                raise ValueError(f"Corrupted calibration file at {self.calibration_path}: {e}") from e

        # Set backwards-compatible property aliases
        self.tau_accept = self.tau_cotton

    def _init_binary(self, checkpoint: dict):
        """Initialize thresholds for legacy binary verifier."""
        self.class_mapping = {0: "cotton"}
        self.tau_accept = 0.85
        self.tau_reject = 0.20
        self.tau_tree_veto = None
        self.tau_tree_hard_reject = None

        if self.calibration_path and self.calibration_path.exists():
            try:
                cal = json.loads(self.calibration_path.read_text(encoding="utf-8"))
                self.tau_accept = float(cal.get("tau_accept", 0.85))
                self.tau_reject = float(cal.get("tau_reject", 0.20))
            except Exception as e:
                raise ValueError(f"Corrupted calibration file at {self.calibration_path}: {e}") from e

        self.tau_cotton = self.tau_accept

    def predict(self, pil_image: Image.Image, preprocessing_mode: str = None) -> dict:
        """
        Verify if an RGB PIL image is a cotton leaf.

        Args:
            pil_image: PIL Image in any format (will be converted to RGB).
            preprocessing_mode: Optional override ('center_crop' or 'direct_resize').
                                Defaults to self.preprocessing_mode ('center_crop').

        Returns:
            dict containing:
                - cotton_probability: float
                - decision: "ACCEPTED_COTTON" | "REJECTED_NON_COTTON" | "REJECTED_UNCERTAIN"
                - threshold_accept: float
                - threshold_reject: float
                - (if 3-class) herbaceous_probability, tree_probability, threshold_tree_veto
                - num_classes: int
                - architecture: str
                - preprocessing_mode: str
        """
        active_mode = preprocessing_mode if preprocessing_mode is not None else self.preprocessing_mode
        if active_mode == self.preprocessing_mode:
            transform = self.transform
        else:
            transform = self._build_transform(active_mode)

        tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.num_classes == 3:
                logits = self.model(tensor).squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                p_cot = round(float(probs[0].item()), 4)
                p_herb = round(float(probs[1].item()), 4)
                p_tree = round(float(probs[2].item()), 4)

                # Decision Rule:
                # 1. ACCEPTED_COTTON: p_cotton >= tau_cotton AND p_tree < tau_tree_veto
                # 2. REJECTED_NON_COTTON: p_cotton <= tau_reject OR p_tree >= tau_tree_hard_reject
                # 3. REJECTED_UNCERTAIN: otherwise (requires retake)
                if p_cot >= self.tau_cotton and p_tree < self.tau_tree_veto:
                    decision = "ACCEPTED_COTTON"
                elif p_cot <= self.tau_reject or p_tree >= self.tau_tree_hard_reject:
                    decision = "REJECTED_NON_COTTON"
                else:
                    decision = "REJECTED_UNCERTAIN"

                return {
                    "cotton_probability": p_cot,
                    "herbaceous_probability": p_herb,
                    "tree_probability": p_tree,
                    "decision": decision,
                    "threshold_accept": self.tau_accept,
                    "threshold_reject": self.tau_reject,
                    "threshold_tree_veto": self.tau_tree_veto,
                    "num_classes": 3,
                    "architecture": self.architecture,
                    "preprocessing_mode": active_mode,
                }

            else:  # Binary single-logit legacy
                logit = self.model(tensor).squeeze(-1).item()
                prob = round(1.0 / (1.0 + float(torch.exp(torch.tensor(-logit)).item())), 4)

                if prob >= self.tau_accept:
                    decision = "ACCEPTED_COTTON"
                elif prob <= self.tau_reject:
                    decision = "REJECTED_NON_COTTON"
                else:
                    decision = "REJECTED_UNCERTAIN"

                return {
                    "cotton_probability": prob,
                    "decision": decision,
                    "threshold_accept": self.tau_accept,
                    "threshold_reject": self.tau_reject,
                    "num_classes": 1,
                    "architecture": self.architecture,
                    "preprocessing_mode": active_mode,
                }

    @classmethod
    def create_candidate_v4(cls, device: str = None) -> "CottonVerifier":
        """Factory method to explicitly load Candidate V4 MCD-Verifier."""
        return cls(
            model_path=CANDIDATE_V4_MODEL_PATH,
            calibration_path=CANDIDATE_V4_CALIBRATION_PATH,
            device=device,
        )

    @classmethod
    def create_candidate_v3(cls, device: str = None) -> "CottonVerifier":
        """Factory method to explicitly load Candidate V3 MCD-Verifier."""
        return cls(
            model_path=CANDIDATE_V3_MODEL_PATH,
            calibration_path=CANDIDATE_V3_CALIBRATION_PATH,
            device=device,
        )

    @classmethod
    def create_legacy(cls, device: str = None) -> "CottonVerifier":
        """Factory method to explicitly load legacy production binary verifier for rollback."""
        return cls(
            model_path=LEGACY_MODEL_PATH,
            calibration_path=LEGACY_CALIBRATION_PATH,
            device=device,
            preprocessing_mode="direct_resize",
        )

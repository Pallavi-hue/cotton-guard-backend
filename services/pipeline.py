"""
Cotton Guard - Gated Multi-Stage Diagnosis Pipeline.

Strict Execution Order:
1. Byte & Format Integrity Check (image_validator.py)
   -> If invalid: Return REJECTED_INVALID_IMAGE (HTTP 400)
2. Stage 1 Leaf-Content Verifier (leaf_verifier.py)
   -> If non-leaf (face, person, document, building, object): Return REJECTED_NON_LEAF (HTTP 422)
3. Stage 2 Cotton-Leaf Verifier (verifier_service.py)
   -> If non-cotton leaf: Return REJECTED_NON_COTTON (HTTP 422)
   -> If uncertain: Return REJECTED_UNCERTAIN (HTTP 422)
4. Stage 3 Disease Classifier (disease_service.py)
   -> EXECUTED ONLY IF ACCEPTED AS COTTON
   -> If Healthy Leaf: Return ACCEPTED_COTTON_HEALTHY (HTTP 200)
   -> If Disease: Return ACCEPTED_COTTON_DISEASE (HTTP 200)
5. Stage 4 Advisory Lookup (advisory_service.py)
"""

import logging
import numpy as np
from PIL import Image
from services.image_validator import validate_image
from services.leaf_verifier import LeafContentVerifier
from services.verifier_service import CottonVerifier
from services.disease_service import DiseaseClassifier
from services.advisory_service import get_advisory

logger = logging.getLogger("cotton_guard.pipeline")


def trim_peripheral_border(img: Image.Image) -> Image.Image:
    """Trim scanning or specimen-mounting dark margins to preserve leaf verifier score."""
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    if min(h, w) < 60:
        return img
    max_trim_y, max_trim_x = int(h * 0.15), int(w * 0.15)
    row_means, col_means = arr.mean(axis=(1, 2)), arr.mean(axis=(0, 2))
    top, bottom, left, right = 0, h, 0, w
    while top < max_trim_y and (row_means[top] < 45 or row_means[top] > 215):
        top += 1
    while bottom > h - max_trim_y and (row_means[bottom - 1] < 45 or row_means[bottom - 1] > 215):
        bottom -= 1
    while left < max_trim_x and (col_means[left] < 45 or col_means[left] > 215):
        left += 1
    while right > w - max_trim_x and (col_means[right - 1] < 45 or col_means[right - 1] > 215):
        right -= 1
    if top > 0 or bottom < h or left > 0 or right < w:
        return img.crop((left, top, right, bottom))
    return img


class GatedDiagnosisPipeline:
    def __init__(
        self,
        leaf_verifier: LeafContentVerifier = None,
        verifier: CottonVerifier = None,
        classifier: DiseaseClassifier = None,
    ):
        self.leaf_verifier = leaf_verifier if leaf_verifier is not None else LeafContentVerifier()
        self.verifier = verifier if verifier is not None else CottonVerifier()
        self.classifier = classifier if classifier is not None else DiseaseClassifier()

    def process(self, image_bytes: bytes) -> tuple[dict, int]:
        # 1. Byte & Format Validation Gate
        validation = validate_image(image_bytes)
        if not validation["valid"]:
            logger.info("Image rejected at Byte Validation Gate: %s", validation["reason"])
            return {
                "status": "REJECTED_INVALID_IMAGE",
                "error_code": validation["reason"],
                "message": validation["message"],
            }, 400

        raw_rgb = validation["image_rgb"]
        image_rgb = trim_peripheral_border(raw_rgb)
        w, h = image_rgb.size

        # 2. Leaf-Content Verifier Gate (Stage 1 Leaf Verification)
        leaf_result = self.leaf_verifier.predict(image_rgb)
        leaf_reason = leaf_result.get("rejection_reason") or "NON_LEAF_IMAGE_DETECTED"
        leaf_prob = leaf_result.get("leaf_probability", 0.0)

        # 2a. Hard Photometric Safeguards (FAST-FAIL: Faces, Documents, Blank Frames)
        hard_photometric_guards = {"FACE_OR_HUMAN_DETECTED", "DOCUMENT_DETECTED", "INSUFFICIENT_TEXTURE"}
        if not leaf_result["is_leaf"] and leaf_reason in hard_photometric_guards:
            logger.info("Image rejected at Photometric Safeguard Gate: %s", leaf_reason)
            return {
                "status": "REJECTED_NON_LEAF",
                "display_title": "NOT A LEAF",
                "error_code": leaf_reason,
                "message": "Cotton Guard accepts only cotton leaf images. This image is not a leaf.",
                "leaf_verification": {
                    "is_leaf": False,
                    "leaf_probability": leaf_prob,
                    "threshold_leaf": leaf_result.get("threshold", 0.745),
                    "decision": leaf_result["decision"],
                    "reason": leaf_reason,
                },
            }, 422

        # 3. Cotton-Leaf Verifier Gate (Stage 2 Cotton Verification)
        verifier_result = self.verifier.predict(image_rgb)
        decision = verifier_result["decision"]
        prob = verifier_result["cotton_probability"]

        # Dual-aspect evaluation: If rectangular and center_crop did not accept, try direct_resize
        ar = w / h if h > 0 else 1.0
        if decision != "ACCEPTED_COTTON" and (ar > 1.15 or ar < 0.85):
            dir_res = self.verifier.predict(image_rgb, preprocessing_mode="direct_resize")
            if dir_res["decision"] == "ACCEPTED_COTTON" or dir_res["cotton_probability"] > prob:
                verifier_result = dir_res
                decision = dir_res["decision"]
                prob = dir_res["cotton_probability"]

        # High-resolution regional leaf evaluation (>= 400px)
        if decision != "ACCEPTED_COTTON" and min(w, h) >= 400:
            cw, ch = int(w * 0.70), int(h * 0.70)
            center_crop = image_rgb.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
            crop_leaf = self.leaf_verifier.predict(center_crop)
            if crop_leaf["is_leaf"]:
                crop_v = self.verifier.predict(center_crop, preprocessing_mode="center_crop")
                if crop_v["decision"] == "ACCEPTED_COTTON" and crop_v["cotton_probability"] >= 0.80 and crop_v.get("tree_probability", 0) < 0.15:
                    verifier_result = crop_v
                    decision = crop_v["decision"]
                    prob = crop_v["cotton_probability"]
                    leaf_prob = max(leaf_prob, crop_leaf.get("leaf_probability", 0.8))

        # 2b. Cascaded Evidence Evaluation:
        # If Stage 1 neural classifier rejected, Stage 2 may overrule only if:
        #   1. leaf_prob >= 0.20 (must have recognizable leaf structure, blocking animals/objects where p_leaf < 0.05)
        #   2. Stage 2 independently confirms ACCEPTED_COTTON (p_cotton >= 0.75 AND zero tree-foliage veto).
        # Otherwise, the Stage 1 non-leaf rejection is strictly upheld.
        if not leaf_result["is_leaf"]:
            if decision == "ACCEPTED_COTTON" and leaf_prob >= 0.20:
                logger.info(
                    "Stage 1 neural ambiguity (p_leaf=%.4f >= 0.20) overruled by Stage 2 high-confidence cotton verification (p_cot=%.4f >= %.2f)",
                    leaf_prob, prob, verifier_result["threshold_accept"]
                )
            else:
                logger.info(
                    "Image rejected at Leaf-Content Gate as Non-Leaf: %s (p_leaf=%.4f, Stage 2 decision=%s)",
                    leaf_reason, leaf_prob, decision
                )
                return {
                    "status": "REJECTED_NON_LEAF",
                    "display_title": "NOT A LEAF",
                    "error_code": leaf_reason,
                    "message": "Cotton Guard accepts only cotton leaf images. This image is not a leaf.",
                    "leaf_verification": {
                        "is_leaf": False,
                        "leaf_probability": leaf_prob,
                        "threshold_leaf": leaf_result.get("threshold", 0.745),
                        "decision": leaf_result["decision"],
                        "reason": leaf_reason,
                    },
                }, 422

        if decision == "REJECTED_NON_COTTON":
            logger.info("Image rejected at Verifier Gate as Non-Cotton (p=%.4f)", prob)
            return {
                "status": "REJECTED_NON_COTTON",
                "display_title": "NOT A COTTON LEAF",
                "error_code": "NON_COTTON_LEAF_DETECTED",
                "message": "This is not a cotton leaf. Please upload a genuine cotton leaf.",
                "verification": {
                    "cotton_probability": prob,
                    "threshold_accept": verifier_result["threshold_accept"],
                    "threshold_reject": verifier_result["threshold_reject"],
                    "decision": decision,
                },
            }, 422

        if decision == "REJECTED_UNCERTAIN":
            logger.info("Image rejected at Verifier Gate as Uncertain (p=%.4f)", prob)
            return {
                "status": "REJECTED_UNCERTAIN",
                "error_code": "CONFIDENCE_BELOW_ACCEPTANCE_THRESHOLD",
                "message": "Unable to verify this image as a cotton leaf with sufficient confidence. Please retake the photo with clearer lighting, closer zoom, and a single focused leaf.",
                "verification": {
                    "cotton_probability": prob,
                    "threshold_accept": verifier_result["threshold_accept"],
                    "threshold_reject": verifier_result["threshold_reject"],
                    "decision": decision,
                },
            }, 422

        # 4. Disease Classifier Gate (Stage 3)
        # ARCHITECTURAL GUARANTEE: REACHED ONLY IF ACCEPTED_COTTON
        logger.info("Image passed Verifier Gate as Verified Cotton (p=%.4f). Running Disease Classifier...", prob)
        diagnosis_result = self.classifier.predict(image_rgb)
        condition = diagnosis_result["condition"]
        is_healthy = diagnosis_result["is_healthy"]

        status = "ACCEPTED_COTTON_HEALTHY" if is_healthy else "ACCEPTED_COTTON_DISEASE"
        advisory = get_advisory(condition)

        return {
            "status": status,
            "verification": {
                "status": "VERIFIED_COTTON",
                "cotton_probability": prob,
                "threshold_accept": verifier_result["threshold_accept"],
            },
            "diagnosis": {
                "condition": condition,
                "confidence": diagnosis_result["confidence"],
                "is_healthy": is_healthy,
                "class_probabilities": diagnosis_result["class_probabilities"],
            },
            "advisory": advisory,
            "image_meta": {
                "format": validation["format"],
                "width": validation["width"],
                "height": validation["height"],
            },
        }, 200

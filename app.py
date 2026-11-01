import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from services.pipeline import GatedDiagnosisPipeline
from services.firebase_auth_service import require_firebase_auth

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cotton_guard")

# Lazy-loaded pipeline singleton
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        logger.info("Initializing Cotton Guard Gated Diagnosis Pipeline...")
        pipeline = GatedDiagnosisPipeline()
        logger.info("Pipeline initialized with Leaf Verifier, Cotton Verifier, and Disease Classifier.")
    return pipeline


@app.get("/health")
def health_check():
    pipe = get_pipeline()
    return jsonify({
        "status": "ok",
        "application": "Cotton Guard API",
        "message": "Backend is running",
        "leaf_verifier_ready": pipe.leaf_verifier is not None,
        "verifier_ready": pipe.verifier is not None,
        "disease_classifier_ready": pipe.classifier is not None,
        "prediction_ready": True,
        "thresholds": {
            "tau_leaf": pipe.leaf_verifier.tau_leaf,
            "tau_accept": pipe.verifier.tau_accept,
            "tau_reject": pipe.verifier.tau_reject,
        }
    }), 200


@app.post("/predict")
@require_firebase_auth
def predict():
    """Gated AI inference for cotton leaf disease diagnosis."""

    if "image" not in request.files:
        return jsonify({
            "status": "REJECTED_INVALID_IMAGE",
            "error_code": "IMAGE_REQUIRED",
            "message": "Upload an image using the 'image' field.",
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({
            "status": "REJECTED_INVALID_IMAGE",
            "error_code": "EMPTY_FILENAME",
            "message": "Select an image file.",
        }), 400

    image_bytes = uploaded_file.read(10 * 1024 * 1024 + 1)
    
    pipe = get_pipeline()
    response_data, status_code = pipe.process(image_bytes)
    return jsonify(response_data), status_code


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "status": "REJECTED_INVALID_IMAGE",
        "error_code": "IMAGE_TOO_LARGE",
        "message": "The request exceeds the 10 MB limit.",
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "NOT_FOUND",
        "message": "The requested API endpoint does not exist.",
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.exception("Unexpected server error")
    return jsonify({
        "error": "INTERNAL_SERVER_ERROR",
        "message": "An internal server error occurred.",
    }), 500


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", 5000)))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)

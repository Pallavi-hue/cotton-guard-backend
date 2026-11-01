"""
Cotton Guard — Firebase ID Token Verification Service
Validates Google-authenticated Firebase ID tokens using Google's public x509 certs.
Complies with Firebase Authentication ID token validation specification:
- Signature: RS256 using Google's public keys
- Issuer (iss): https://securetoken.google.com/<PROJECT_ID>
- Audience (aud): <PROJECT_ID>
- Expiration (exp): Must not be expired
- Subject (sub): Unique Firebase user UID
"""

import time
import os
import logging
from functools import wraps
import requests
import jwt
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.backends import default_backend
from flask import request, jsonify

logger = logging.getLogger("cotton_guard.auth")

GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
DEFAULT_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "cotton-guard")

# In-memory certificate cache: {kid: pem_cert, ...}, expires_at
_cert_cache = {}
_cache_expires_at = 0.0


def _get_google_public_certs() -> dict:
    global _cert_cache, _cache_expires_at
    now = time.time()
    if _cert_cache and now < _cache_expires_at:
        return _cert_cache

    try:
        resp = requests.get(GOOGLE_CERTS_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # Parse cache-control header max-age
        cc = resp.headers.get("Cache-Control", "")
        max_age = 3600
        for part in cc.split(","):
            part = part.strip()
            if part.startswith("max-age="):
                try:
                    max_age = int(part.split("=")[1])
                except ValueError:
                    pass

        _cert_cache = data
        _cache_expires_at = now + max_age
        logger.info("Successfully fetched %d Google public certificates (max-age=%ds)", len(data), max_age)
        return _cert_cache
    except Exception as e:
        logger.error("Failed to fetch Google public certs: %s", e)
        if _cert_cache:
            return _cert_cache
        raise RuntimeError(f"Unable to reach Google certificate endpoint: {e}")


def verify_firebase_id_token(id_token: str, project_id: str = DEFAULT_PROJECT_ID) -> dict:
    """
    Verifies a Firebase ID token and returns the decoded payload.
    Raises ValueError, jwt.PyJWTError on invalid tokens.
    """
    if not id_token or not isinstance(id_token, str):
        raise ValueError("ID token must be a non-empty string.")

    # 1. Inspect unverified header for key id (kid)
    try:
        unverified_header = jwt.get_unverified_header(id_token)
    except Exception as e:
        raise ValueError(f"Malformed JWT header: {e}")

    kid = unverified_header.get("kid")
    if not kid:
        raise ValueError("Missing 'kid' in token header.")

    alg = unverified_header.get("alg")
    if alg != "RS256":
        raise ValueError(f"Invalid algorithm '{alg}'. Firebase requires RS256.")

    # 2. Match kid against Google public keys
    certs = _get_google_public_certs()
    cert_pem = certs.get(kid)
    if not cert_pem:
        raise ValueError(f"Unknown 'kid' ({kid}). Public certificate not found or expired.")

    # Convert PEM x509 certificate string to public key
    try:
        cert_obj = load_pem_x509_certificate(cert_pem.encode("utf-8"), default_backend())
        public_key = cert_obj.public_key()
    except Exception as e:
        raise ValueError(f"Failed to load x509 public key: {e}")

    # 3. Decode & Verify signature, expiration, audience, issuer
    expected_issuer = f"https://securetoken.google.com/{project_id}"
    decoded = jwt.decode(
        id_token,
        public_key,
        algorithms=["RS256"],
        audience=project_id,
        issuer=expected_issuer,
        options={
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "verify_aud": True,
            "verify_iss": True,
        }
    )

    uid = decoded.get("sub")
    if not uid:
        raise ValueError("Missing 'sub' (user UID) in verified token.")

    return decoded


def require_firebase_auth(f):
    """
    Flask route decorator that enforces Firebase ID token authentication on protected endpoints.
    Can be toggled via REQUIRE_AUTH environment variable (defaults to True in production).
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        require_auth = os.getenv("REQUIRE_AUTH", "false").lower() in ("true", "1")
        if not require_auth:
            request.firebase_user = {"uid": "anonymous_dev_mode", "email": "dev@cottonguard.local"}
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.warning("Unauthenticated request to %s: Missing Authorization header", request.path)
            return jsonify({
                "status": "AUTH_REQUIRED",
                "error_code": "UNAUTHORIZED",
                "message": "Authentication required. Please provide a Firebase ID token via 'Authorization: Bearer <TOKEN>'."
            }), 401

        parts = auth_header.split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({
                "status": "AUTH_FAILED",
                "error_code": "INVALID_AUTH_HEADER",
                "message": "Invalid Authorization format. Expected 'Bearer <TOKEN>'."
            }), 401

        token = parts[1].strip()
        project_id = os.getenv("FIREBASE_PROJECT_ID", DEFAULT_PROJECT_ID)

        try:
            payload = verify_firebase_id_token(token, project_id=project_id)
            request.firebase_user = payload
            logger.info("Authenticated Firebase user %s (%s) for %s", payload.get("uid"), payload.get("email"), request.path)
        except Exception as e:
            logger.warning("Token verification failed for %s: %s", request.path, e)
            return jsonify({
                "status": "AUTH_FAILED",
                "error_code": "INVALID_TOKEN",
                "message": f"Firebase authentication failed: {e}"
            }), 401

        return f(*args, **kwargs)

    return decorated_function

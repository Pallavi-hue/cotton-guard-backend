# Cotton Guard — Production Backend (Render Deployment)

Production-grade Flask backend for the **Cotton Guard** multi-stage gated AI diagnosis pipeline.

---

## Architecture Overview

```
                      +-----------------------------+
                      |       Image Upload          |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   1. Byte & Format Gate     |  --> HTTP 400 REJECTED_INVALID_IMAGE
                      +-----------------------------+
                                     | Valid image
                                     v
                      +-----------------------------+
                      |   2. Leaf Verifier Gate     |  --> HTTP 422 REJECTED_NON_LEAF
                      |   (MobileNetV3-Small)       |
                      +-----------------------------+
                                     | Leaf score >= 0.70
                                     v
                      +-----------------------------+
                      |   3. Cotton Verifier Gate   |  --> HTTP 422 REJECTED_NON_COTTON
                      |   (Production Candidate V4) |  --> HTTP 422 REJECTED_UNCERTAIN
                      +-----------------------------+
                                     | Cotton score >= 0.75
                                     v
                      +-----------------------------+
                      |  4. Disease Classifier      |  --> HTTP 200 ACCEPTED_COTTON_HEALTHY
                      |  (EfficientNet-B0 7-Class)  |  --> HTTP 200 ACCEPTED_COTTON_DISEASE
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  5. SQLite Advisory Lookup  |
                      |  (advisories.db)            |
                      +-----------------------------+
```

---

## Production Models & Artifacts

- **Stage 1 (Leaf Verifier)**:
  - Model: `models/leaf_verifier/best_checkpoint.pth` (MobileNetV3-Small binary)
  - Calibration: `models/leaf_verifier/leaf_threshold_calibration.json` (`tau_leaf = 0.70`)
- **Stage 2 (Cotton Verifier)**:
  - Model: `models/cotton_verifier_candidate_v4/best_checkpoint.pth` (Production V4 multi-class)
  - Calibration: `models/cotton_verifier_candidate_v4/threshold_calibration.json` (`tau_accept = 0.75`, `tau_reject = 0.25`)
  - Class Mapping: `models/cotton_verifier_candidate_v4/class_mapping.json`
- **Stage 3 (Disease Classifier)**:
  - Model: `models/disease_classifier/best_checkpoint.pth` (EfficientNet-B0 7-class)
- **Agronomic Knowledge Base**:
  - Database: `database/advisories.db` (SQLite read-only reference table)

---

## Security & Authentication

Protected by stateless Firebase Authentication ID token verification against Google's public x509 certificates.
- Zero server-side private key files required.
- Enabled via `REQUIRE_AUTH=true`.
- Client attaches `Authorization: Bearer <FIREBASE_ID_TOKEN>`.

---

## Deployment on Render

### Service Configuration
- **Runtime**: Python 3.12
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`
- **Health Check Path**: `/health`

### Environment Variables
| Variable | Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.12.4` | Enforces compatible Python 3.12 environment |
| `REQUIRE_AUTH` | `true` | Enforces Firebase ID token validation |
| `FIREBASE_PROJECT_ID` | `cotton-guard` | Target Firebase project ID |
| `FLASK_DEBUG` | `false` | Production mode |

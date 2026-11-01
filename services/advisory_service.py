"""
Cotton Guard - Advisory Repository Service.
Fetches agronomic advisory from SQLite database.
"""

from pathlib import Path
import sqlite3

_CURRENT_DIR = Path(__file__).resolve().parent
if (_CURRENT_DIR.parent / "database" / "advisories.db").exists():
    DB_PATH = _CURRENT_DIR.parent / "database" / "advisories.db"
elif (_CURRENT_DIR.parent.parent / "database" / "advisories.db").exists():
    DB_PATH = _CURRENT_DIR.parent.parent / "database" / "advisories.db"
else:
    DB_PATH = _CURRENT_DIR.parent / "database" / "advisories.db"


def get_advisory(condition: str) -> dict:
    if not DB_PATH.exists():
        return {
            "condition": condition,
            "status": "ADVISORY_DB_NOT_FOUND",
            "message": "Agronomic advisory database is not yet initialized.",
        }

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM advisories WHERE condition = ?", (condition,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "condition": condition,
            "scientific_name": "Unknown",
            "severity_level": "Unknown",
            "symptoms": "No specific advisory found.",
            "chemical_control": "Consult local agricultural extension officer.",
            "organic_control": "Consult local agricultural extension officer.",
            "prevention": "Inspect field regularly.",
        }

    return {
        "condition": row["condition"],
        "scientific_name": row["scientific_name"],
        "severity_level": row["severity_level"],
        "symptoms": row["symptoms"],
        "chemical_control": row["chemical_control"],
        "organic_control": row["organic_control"],
        "prevention": row["prevention"],
    }

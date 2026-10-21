import csv
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Anchor data directory to the project root (parent of src/)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MATURITY_CSV = DATA_DIR / "coffee_maturity_predictions.csv"
YIELD_CSV    = DATA_DIR / "coffee_yield_predictions.csv"

# Make sure directory exists
DATA_DIR.mkdir(exist_ok=True)

MATURITY_FIELDS = [
    "timestamp",
    "image_name",
    "total_fruits_detected",
    "ripe_count",
    "unripe_count",
    "overripe_count",
    "predicted_maturity",
    "yolo_confidence",
    "efficientnet_confidence",
    "hybrid_confidence",
]

YIELD_FIELDS = [
    "timestamp",
    "image_name",
    "fruit_count",
    "ripe_count",
    "unripe_count",
    "overripe_count",
    "ripeness_ratio",
    "temperature",
    "humidity",
    "rainfall_today",
    "rainfall_30d",
    "rainfall_90d",
    "avg_temp_30d",
    "predicted_yield",
    "ripening_days",
    "expected_harvest_date",
]


def _init_csv(filepath, fields):
    if not filepath.exists():
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()


def init_csv_files():
    _init_csv(MATURITY_CSV, MATURITY_FIELDS)
    _init_csv(YIELD_CSV, YIELD_FIELDS)


init_csv_files()


def _clean_data(data: dict, fields: list) -> dict:
    clean = {}
    for field in fields:
        if field == "timestamp" and not data.get("timestamp"):
            clean[field] = datetime.now().isoformat()
        else:
            val = data.get(field, "")
            if isinstance(val, float) and pd.isna(val):
                val = 0.0
            clean[field] = val
    return clean


def save_maturity_prediction(data: dict):
    """Write a maturity prediction row to coffee_maturity_predictions.csv."""
    clean = _clean_data(data, MATURITY_FIELDS)
    with open(MATURITY_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MATURITY_FIELDS)
        writer.writerow(clean)
    print("[CSV] Saved maturity prediction")


def save_yield_prediction(data: dict):
    """Write a yield prediction row to coffee_yield_predictions.csv."""
    clean = _clean_data(data, YIELD_FIELDS)
    with open(YIELD_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=YIELD_FIELDS)
        writer.writerow(clean)
    print("[CSV] Saved yield prediction")


def get_maturity_history(limit=1000):
    if not MATURITY_CSV.exists():
        return []
    try:
        df = pd.read_csv(MATURITY_CSV)
        return df.tail(limit).to_dict(orient='records')[::-1]
    except Exception:
        return []


def get_yield_history(limit=1000):
    if not YIELD_CSV.exists():
        return []
    try:
        df = pd.read_csv(YIELD_CSV)
        return df.tail(limit).to_dict(orient='records')[::-1]
    except Exception:
        return []

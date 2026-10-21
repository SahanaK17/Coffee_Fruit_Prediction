"""
Soil Data Service — SoilGrids API Integration
Fetches soil properties (pH, clay, sand, silt, SOC) from the ISRIC SoilGrids v2.0 API.
Provides caching, preprocessing, and feature fusion utilities.
"""

import os
import csv
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# ===== CONSTANTS =====
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPERTIES = ["phh2o", "clay", "sand", "silt", "soc"]
SOIL_COLUMNS = ["ph", "clay", "sand", "silt", "soc"]
DEPTH = "0-5cm"
VALUE = "mean"

# Cache file path
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_FILE = BASE_DIR / "data" / "soil_cache.csv"

# Normalization ranges (based on typical global ranges for 0-5cm depth)
# These are used for min-max normalization
SOIL_RANGES = {
    "ph":   {"min": 3.5,  "max": 9.5},    # pH scale
    "clay": {"min": 0.0,  "max": 80.0},    # percentage
    "sand": {"min": 0.0,  "max": 100.0},   # percentage
    "silt": {"min": 0.0,  "max": 80.0},    # percentage
    "soc":  {"min": 0.0,  "max": 150.0},   # g/kg
}

# Global in-memory cache: (lat, lon) → {ph, clay, sand, silt, soc}
_memory_cache: Dict[str, Dict[str, float]] = {}


def _load_cache_from_disk():
    """Load previously cached soil data from CSV into memory."""
    global _memory_cache
    if not CACHE_FILE.exists():
        return
    try:
        df = pd.read_csv(CACHE_FILE)
        for _, row in df.iterrows():
            key = f"{row['lat']:.4f}_{row['lon']:.4f}"
            _memory_cache[key] = {
                "ph":   float(row["ph"]),
                "clay": float(row["clay"]),
                "sand": float(row["sand"]),
                "silt": float(row["silt"]),
                "soc":  float(row["soc"]),
            }
        print(f"[SOIL] Loaded {len(_memory_cache)} cached soil entries from disk")
    except Exception as e:
        print(f"[SOIL] Warning: Failed to load cache: {e}")


def _save_to_cache(lat: float, lon: float, data: Dict[str, float]):
    """Persist a single soil entry to the CSV cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = CACHE_FILE.exists()
    with open(CACHE_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lat", "lon"] + SOIL_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"lat": round(lat, 4), "lon": round(lon, 4), **data})


def _parse_soilgrids_response(response_json: dict) -> Dict[str, Optional[float]]:
    """
    Parse the SoilGrids API response and convert raw values to proper units.
    
    Conversion factors (all use d_factor=10):
    - phh2o: pH*10 → actual pH (divide by 10)
    - clay:  g/kg  → % (divide by 10)
    - sand:  g/kg  → % (divide by 10)
    - silt:  g/kg  → % (divide by 10)
    - soc:   dg/kg → g/kg (divide by 10)
    """
    result = {col: None for col in SOIL_COLUMNS}
    
    try:
        layers = response_json.get("properties", {}).get("layers", [])
        for layer in layers:
            name = layer.get("name", "")
            depths = layer.get("depths", [])
            
            # Find the 0-5cm depth
            for depth_entry in depths:
                if depth_entry.get("label") == DEPTH:
                    raw_value = depth_entry.get("values", {}).get(VALUE)
                    if raw_value is not None:
                        # All properties use d_factor = 10
                        converted = raw_value / 10.0
                        
                        # Map API name to our column name
                        if name == "phh2o":
                            result["ph"] = converted
                        elif name in SOIL_COLUMNS:
                            result[name] = converted
                    break
    except Exception as e:
        print(f"[SOIL] Error parsing response: {e}")
    
    return result


def get_soil_data(lat: float, lon: float) -> Dict[str, Optional[float]]:
    """
    Fetch soil properties for a given (lat, lon) coordinate.
    
    Uses a multi-level cache:
    1. In-memory dictionary (fastest)
    2. CSV file on disk (persistent)
    3. SoilGrids API call (fallback)
    
    Returns dict with keys: ph, clay, sand, silt, soc
    Values are in proper units (pH scale, %, %, %, g/kg)
    Returns None for any property that couldn't be fetched.
    """
    # Load cache on first call
    if not _memory_cache:
        _load_cache_from_disk()
    
    # Check memory cache (round to 4 decimal places for matching)
    cache_key = f"{lat:.4f}_{lon:.4f}"
    if cache_key in _memory_cache:
        print(f"[SOIL] Cache HIT for ({lat:.4f}, {lon:.4f})")
        return _memory_cache[cache_key].copy()
    
    # Call SoilGrids API
    print(f"[SOIL] API call for ({lat:.4f}, {lon:.4f})...")
    try:
        params = [
            ("lat", lat),
            ("lon", lon),
            ("depth", DEPTH),
            ("value", VALUE),
        ]
        for prop in SOIL_PROPERTIES:
            params.append(("property", prop))
            
        response = requests.get(SOILGRIDS_URL, params=params, timeout=15)
        response.raise_for_status()
        
        response_json = response.json()
        if not response_json.get("properties", {}).get("layers"):
            print("[SOIL] Warning: SoilGrids API returned empty data (no layers)")
            
        data = _parse_soilgrids_response(response_json)
        
        # Only cache if we got at least some valid data
        valid_count = sum(1 for v in data.values() if v is not None)
        if valid_count > 0:
            _memory_cache[cache_key] = data
            _save_to_cache(lat, lon, {k: v if v is not None else -1 for k, v in data.items()})
            print(f"[SOIL] Successfully fetched {valid_count}/5 properties")
        else:
            print(f"[SOIL] Warning: No valid soil data returned for ({lat}, {lon})")
        
        return data
        
    except requests.exceptions.Timeout:
        print(f"[SOIL] API timeout for ({lat}, {lon})")
    except requests.exceptions.ConnectionError:
        print(f"[SOIL] Connection error for ({lat}, {lon})")
    except requests.exceptions.HTTPError as e:
        print(f"[SOIL] HTTP error for ({lat}, {lon}): {e}")
    except Exception as e:
        print(f"[SOIL] Unexpected error for ({lat}, {lon}): {e}")
    
    # Return None values on failure
    return {col: None for col in SOIL_COLUMNS}


def preprocess_soil_features(
    soil_data: Dict[str, Optional[float]],
    impute_values: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Preprocess soil features: handle missing values + normalize.
    
    Args:
        soil_data: Dict with keys [ph, clay, sand, silt, soc]
        impute_values: Optional dict of mean values for imputation.
                       If None, uses sensible global defaults.
    
    Returns:
        numpy array of shape (5,) with normalized [0, 1] values
        Order: [ph, clay, sand, silt, soc]
    """
    # Default imputation values (global means for coffee-growing regions)
    defaults = impute_values or {
        "ph":   5.8,    # Slightly acidic, typical for coffee
        "clay": 25.0,   # %
        "sand": 45.0,   # %
        "silt": 30.0,   # %
        "soc":  25.0,   # g/kg
    }
    
    features = []
    for col in SOIL_COLUMNS:
        value = soil_data.get(col)
        
        # Handle missing values (None or -1 sentinel)
        if value is None or value < 0:
            value = defaults.get(col, 0.0)
        
        # Min-max normalize to [0, 1]
        r = SOIL_RANGES[col]
        normalized = (value - r["min"]) / (r["max"] - r["min"])
        normalized = max(0.0, min(1.0, normalized))  # Clamp
        
        features.append(normalized)
    
    return np.array(features, dtype=np.float32)


def merge_features(
    image_features: np.ndarray,
    soil_features: np.ndarray,
) -> np.ndarray:
    """
    Concatenate image features with soil features.
    
    Args:
        image_features: numpy array from CNN feature extractor (e.g., shape (1280,))
        soil_features:  numpy array from preprocess_soil_features (shape (5,))
    
    Returns:
        Concatenated feature vector (e.g., shape (1285,))
    """
    return np.concatenate([image_features, soil_features])


def compute_imputation_means(cache_path: Optional[str] = None) -> Dict[str, float]:
    """
    Compute mean values from the cached soil data for imputation.
    Falls back to global defaults if no cache available.
    """
    path = Path(cache_path) if cache_path else CACHE_FILE
    defaults = {"ph": 5.8, "clay": 25.0, "sand": 45.0, "silt": 30.0, "soc": 25.0}
    
    if not path.exists():
        return defaults
    
    try:
        df = pd.read_csv(path)
        means = {}
        for col in SOIL_COLUMNS:
            valid_vals = df[col][df[col] >= 0]  # Exclude -1 sentinel
            means[col] = float(valid_vals.mean()) if len(valid_vals) > 0 else defaults[col]
        return means
    except Exception:
        return defaults


def get_soil_features_for_prediction(lat: float, lon: float) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
    """
    Convenience function: fetch + preprocess soil data in one call.
    
    Returns:
        (normalized_features_array, raw_soil_data_dict)
    """
    raw_data = get_soil_data(lat, lon)
    features = preprocess_soil_features(raw_data)
    return features, raw_data

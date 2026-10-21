"""
Enhanced Yield Module — Intelligent Decision System
=====================================================
Features:
  - 12-feature XGBoost prediction (11 + soil_score)
  - Soil quality scoring (soil type, water retention, drainage)
  - Scientific yield formula fallback
  - Profit / Loss indicator
  - Reason identification engine
  - Suggestion engine
  - Potential yield improvement estimation
"""

import joblib
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta
import requests

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
YIELD_MODEL_PATH = MODELS_DIR / "yield" / "yield_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "yield" / "feature_names.joblib"

# Agronomy constants
AVG_CHERRY_WEIGHT_G = 1.8       # grams per cherry
PLANT_DENSITY = 2500            # plants per hectare

# Default economics
DEFAULT_PRICE_PER_KG = 200       # ₹ per kg
DEFAULT_COST_PER_HECTARE = 250000  # ₹ per hectare

# Global model instances
_yield_model = None
_feature_names = None
_effnet_extractor = None


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_yield_model():
    global _yield_model, _feature_names
    if _yield_model is None:
        if os.path.exists(YIELD_MODEL_PATH):
            _yield_model = joblib.load(YIELD_MODEL_PATH)
            if os.path.exists(FEATURE_NAMES_PATH):
                _feature_names = joblib.load(FEATURE_NAMES_PATH)
            print(f"[INFO] Yield model loaded from {YIELD_MODEL_PATH}")
        else:
            print(f"[WARNING] Yield model not found at {YIELD_MODEL_PATH}. Agronomy formula will be used.")
    return _yield_model, _feature_names


def get_effnet_extractor(device='cpu'):
    """EfficientNet-B0 used as a pretrained visual feature extractor."""
    global _effnet_extractor
    if _effnet_extractor is None:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Identity()
        model.to(device)
        model.eval()
        _effnet_extractor = model
    return _effnet_extractor


# ─────────────────────────────────────────────
# Vision extraction
# ─────────────────────────────────────────────
def detect_coffee_fruits(image, yolo_instance):
    """Run YOLO detection on an image path or array."""
    results = yolo_instance(image, verbose=False)[0]
    return results


def extract_visual_features(detections) -> dict:
    """
    Extract fruit_count, ripe_count, unripe_count, overripe_count,
    and ripeness_ratio from YOLO detection results.
    """
    boxes   = detections.boxes.xyxy.cpu().numpy()
    scores  = detections.boxes.conf.cpu().numpy()
    cls_ids = detections.boxes.cls.cpu().numpy().astype(int)
    names   = detections.names

    mask = scores >= 0.3
    filtered_classes = cls_ids[mask]

    fruit_count = len(filtered_classes)

    counts = {'Unripe': 0, 'Ripe': 0, 'Overripe': 0}
    for cls_id in filtered_classes:
        label = names.get(cls_id, "Unknown")
        if label in counts:
            counts[label] += 1

    ripe_count    = counts['Ripe']
    unripe_count  = counts['Unripe']
    overripe_count = counts['Overripe']
    ripeness_ratio = ripe_count / fruit_count if fruit_count > 0 else 0.0

    return {
        "fruit_count":    fruit_count,
        "ripe":           ripe_count,
        "ripe_count":     ripe_count,
        "unripe":         unripe_count,
        "unripe_count":   unripe_count,
        "overripe":       overripe_count,
        "overripe_count": overripe_count,
        "ripeness_ratio": float(ripeness_ratio),
    }


# ─────────────────────────────────────────────
# Soil Quality Scoring
# ─────────────────────────────────────────────
_SOIL_TYPE_SCORES = {
    'loamy':      0.9,
    'clay':       0.7,
    'sandy':      0.5,
    'red soil':   0.75,
    'black soil': 0.85,
}

_WATER_RETENTION_SCORES = {
    'low':    0.4,
    'medium': 0.7,
    'high':   0.9,
}

_DRAINAGE_SCORES = {
    'poor':     0.5,
    'moderate': 0.7,
    'good':     0.9,
}

DEFAULT_SOIL_SCORE = 0.7

def compute_soil_index(soil_type: str, drainage: str) -> float:
    """
    Step 3: Feature Engineering
    Convert soil_type and drainage into numerical scores
    Compute soil_index (0–1 scale)
    """
    st = _SOIL_TYPE_SCORES.get(soil_type.lower().strip(), 0.7)
    dr = _DRAINAGE_SCORES.get(drainage.lower().strip(), 0.7)
    return round((st + dr) / 2, 4)

# Keep old compute_soil_score for backward compatibility
def compute_soil_score(
    soil_type: str = 'loamy',
    water_retention: str = 'medium',
    drainage: str = 'moderate',
) -> float:
    """
    Compute a normalized soil quality score (0–1) from farmer-friendly inputs.
    Formula: (soil_type_score + water_score + drainage_score) / 3
    """
    st = _SOIL_TYPE_SCORES.get(soil_type.lower().strip(), 0.7)
    wr = _WATER_RETENTION_SCORES.get(water_retention.lower().strip(), 0.7)
    dr = _DRAINAGE_SCORES.get(drainage.lower().strip(), 0.7)
    return round((st + wr + dr) / 3, 4)

KARNATAKA_SOIL_DATA = {
    "chikkamagaluru": {"soil_type": "loamy", "drainage": "good", "water_retention": "medium", "organic_carbon": "high"},
    "kodagu": {"soil_type": "clay", "drainage": "moderate", "water_retention": "high", "organic_carbon": "medium"},
    "hassan": {"soil_type": "sandy", "drainage": "good", "water_retention": "low", "organic_carbon": "low"},
    "mysuru": {"soil_type": "red soil", "drainage": "moderate", "water_retention": "medium", "organic_carbon": "medium"},
    "shivamogga": {"soil_type": "loamy", "drainage": "good", "water_retention": "medium", "organic_carbon": "high"},
    "dakshina kannada": {"soil_type": "loamy", "drainage": "good", "water_retention": "high", "organic_carbon": "high"}
}

def get_soil_data(district: str) -> dict:
    """Step 2: Soil Dataset Integration
    get_soil_data(district) returns soil features for karnataka districts.
    """
    if not district:
        return {"soil_type": "loamy", "drainage": "moderate", "water_retention": "medium", "organic_carbon": "medium"}
    
    district_lower = district.lower().replace(" district", "")
    for dist_key, soil_info in KARNATAKA_SOIL_DATA.items():
        if dist_key in district_lower:
            return soil_info
            
    # Add fallback default soil if district not found
    return {"soil_type": "loamy", "drainage": "moderate", "water_retention": "medium", "organic_carbon": "medium"}

def get_soil_data_by_location(lat: float, lon: float) -> dict:
    """Step 1: Reverse Geocoding via OpenStreetMap Nominatim"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {'User-Agent': 'CoffeeAI-Yield-Predictor'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        address = data.get('address', {})
        district = address.get('state_district', address.get('county', ''))
        # state = address.get('state', '')
        
        soil_data = get_soil_data(district)
        soil_data["detected_district"] = district.title() if district else "Unknown"
        return soil_data
        
        # If not found but we got a district
        if district:
            return {"soil_type": "loamy", "drainage": "moderate", "water_retention": "medium", "detected_district": district.title()}
    except Exception as e:
        print(f"[WARNING] Reverse geocoding failed: {e}")
        
    # Complete fallback
    return {"soil_type": "loamy", "drainage": "moderate", "water_retention": "medium", "detected_district": "Unknown (Fallback)"}


# ─────────────────────────────────────────────
# Feature vector  (12-feature model: 11 + soil_score)
# ─────────────────────────────────────────────
def _get_current_season() -> int:
    """Determine season from current month: 0=summer, 1=monsoon, 2=winter."""
    month = datetime.utcnow().month
    if month in (3, 4, 5):      # Mar-May
        return 0  # summer
    elif month in (6, 7, 8, 9):  # Jun-Sep
        return 1  # monsoon
    else:                         # Oct-Feb
        return 2  # winter


def create_feature_vector(features_dict: dict) -> np.ndarray:
    """
    Build X = [fruit_count, ripeness_ratio, temperature, humidity,
               rainfall_30d, rainfall_90d,
               interaction_1, interaction_2, interaction_3,
               rainfall_trend, season, soil_score]
    Backward compatible: if soil_score is missing, default 0.7 is used.
    """
    fc   = features_dict.get('fruit_count', 0)
    rr   = features_dict.get('ripeness_ratio', 0.0)
    temp = features_dict.get('temperature', 25.0)
    hum  = features_dict.get('humidity', 75.0)
    r30  = features_dict.get('rainfall_last_30_days', 90.0)
    r90  = features_dict.get('rainfall_last_90_days', 270.0)
    soil = features_dict.get('soil_score', DEFAULT_SOIL_SCORE)

    # Engineered features
    interaction_1 = fc * rr
    interaction_2 = r30 / (temp + 1)
    interaction_3 = hum * rr
    rainfall_trend = r90 - r30
    season = _get_current_season()
    
    # Step 4: Integrate Into Existing Prediction Pipeline
    # features = [fruit_count, ripeness_ratio, temperature, humidity, rainfall..., soil_index]
    soil_index = features_dict.get('soil_index', soil)

    vec = [
        fc, rr, temp, hum, r30, r90,
        interaction_1, interaction_2, interaction_3,
        rainfall_trend, season, soil_index
    ]
    return np.array(vec, dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────
# Agronomy formula backup
# ─────────────────────────────────────────────
def agronomy_yield_formula(fruit_count: int) -> float:
    """
    Scientific fallback:
    yield_kg_per_plant = fruit_count × avg_cherry_weight_g / 1000
    """
    yield_g_per_plant  = fruit_count * AVG_CHERRY_WEIGHT_G
    yield_kg_per_plant = yield_g_per_plant / 1000.0
    return round(yield_kg_per_plant, 3)


# ─────────────────────────────────────────────
# ML yield prediction
# ─────────────────────────────────────────────
def predict_coffee_yield(feature_vector: np.ndarray, soil_score: float = DEFAULT_SOIL_SCORE) -> float:
    """
    Predict coffee yield in kg/ha using the trained XGBoost model.
    Falls back to the agronomy formula if the model is unavailable.
    The final yield is adjusted by soil quality: final_yield = predicted * soil_score.
    """
    model, _ = load_yield_model()
    if model is None:
        fruit_count = int(feature_vector[0][0])
        # Base yield per hectare using agronomy formula times plant density
        base_yield_ha = agronomy_yield_formula(fruit_count) * PLANT_DENSITY
    else:
        # Use only the original 11 features if model expects 11
        try:
            base_yield_ha = float(model.predict(feature_vector[:, :11])[0])
        except Exception:
            base_yield_ha = float(model.predict(feature_vector)[0])

    # Apply soil quality adjustment
    final_yield_ha = base_yield_ha * soil_score
    return round(final_yield_ha, 3)


def calculate_prediction_confidence(prediction: float, feature_vector: np.ndarray) -> float:
    """Heuristic confidence score based on feature plausibility."""
    base_conf  = 0.88
    fruit_count = feature_vector[0][0]
    ripeness    = feature_vector[0][1]
    temp        = feature_vector[0][2]

    if fruit_count < 2 or fruit_count > 180:
        base_conf -= 0.10
    if prediction < 200 or prediction > 5000:
        base_conf -= 0.12
    if ripeness > 0.5:
        base_conf += 0.03
    if 20 <= temp <= 28:
        base_conf += 0.02

    confidence = max(0.60, min(0.98, base_conf + np.random.uniform(-0.02, 0.04)))
    return float(confidence)


# ─────────────────────────────────────────────
# Ripening prediction
# ─────────────────────────────────────────────
def predict_ripening_days(temperature: float, unripe_count: int, fruit_count: int) -> dict:
    """Estimate days remaining until unripe fruits ripen."""
    if fruit_count == 0 or unripe_count == 0:
        return {
            "ripening_days": 0,
            "expected_harvest_date": datetime.utcnow().strftime("%Y-%m-%d"),
        }

    if temperature > 28:
        ripening_days = 12
    elif temperature >= 22:
        ripening_days = 14
    else:
        ripening_days = 15

    harvest_date = (datetime.utcnow() + timedelta(days=ripening_days)).strftime("%Y-%m-%d")

    return {
        "ripening_days":         ripening_days,
        "expected_harvest_date": harvest_date,
    }


# ─────────────────────────────────────────────
# Feature importances (12 features: 11 + Soil Score)
# ─────────────────────────────────────────────
FEATURE_LABELS_12 = [
    "Fruit Count", "Ripeness Ratio",
    "Temperature", "Humidity",
    "Rainfall 30d", "Rainfall 90d",
    "Fruit×Ripeness", "Rain/Temp",
    "Humidity×Ripeness", "Rainfall Trend",
    "Season", "Soil Score",
]


def get_feature_importances() -> dict:
    """Return feature importances from the ML model, or sensible defaults."""
    model, names = load_yield_model()
    if model is not None and hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        labels = FEATURE_LABELS_12[:len(imp)]
        result = {
            labels[i]: float(imp[i])
            for i in range(len(labels))
        }
        # Ensure Soil Score is always present
        if 'Soil Score' not in result:
            result['Soil Score'] = 0.08
        return result

    # Defaults when model is unavailable
    return {
        "Fruit Count":        0.25,
        "Ripeness Ratio":     0.13,
        "Temperature":        0.09,
        "Humidity":           0.07,
        "Rainfall 30d":       0.06,
        "Rainfall 90d":       0.05,
        "Fruit×Ripeness":     0.09,
        "Rain/Temp":          0.05,
        "Humidity×Ripeness":  0.04,
        "Rainfall Trend":     0.04,
        "Season":             0.03,
        "Soil Score":         0.10,
    }


# ─────────────────────────────────────────────
# PROFIT INDICATOR
# ─────────────────────────────────────────────
def compute_plant_profit_indicator(
    predicted_yield_ha: float,
    price_per_kg: float = DEFAULT_PRICE_PER_KG,
    estimated_plant_maintenance_cost: float = 100.0,
) -> dict:
    """
    Calculate economics at the hectare level.
    """
    # Convert per-plant upkeep cost to per-hectare upkeep cost
    cost_per_hectare = estimated_plant_maintenance_cost * PLANT_DENSITY
    
    revenue = predicted_yield_ha * price_per_kg
    profit = revenue - cost_per_hectare
    
    # Break-even is when revenue == cost
    break_even_yield_ha = cost_per_hectare / price_per_kg if price_per_kg > 0 else 0

    status = "PROFIT" if profit >= 0 else "LOSS"

    return {
        "status": status,
        "revenue": round(revenue, 2),
        "profit": round(profit, 2),
        "break_even_yield": round(break_even_yield_ha, 3),
        "price_per_kg": price_per_kg,
        "estimated_plant_maintenance_cost": estimated_plant_maintenance_cost,
    }


# ─────────────────────────────────────────────
# REASON IDENTIFICATION ENGINE (v2)
# ─────────────────────────────────────────────

# Each entry: (reason_text, suggestion_text)
_REASON_SUGGESTION_PAIRS = {
    "high_temp":      ("High temperature stress reducing yield",
                       "Use shade nets or mulching to reduce heat"),
    "low_humidity":   ("Low humidity affecting fruit development",
                       "Use micro-irrigation or misting systems"),
    "low_rain_30":    ("Insufficient recent rainfall",
                       "Increase irrigation frequency"),
    "low_rain_90":    ("Low long-term rainfall affecting yield",
                       "Adopt water conservation techniques"),
    "low_ripeness":   ("Low ripeness ratio indicating incomplete maturity",
                       "Delay harvesting to allow proper ripening"),
    "low_fruit":      ("Low fruit density detected",
                       "Improve fertilization and pruning practices"),
    "poor_soil":      ("Poor soil quality affecting yield",
                       "Apply organic compost or soil treatment"),
    "waterlogging_risk": ("Clay soil with poor drainage poses waterlogging risk",
                       "Improve field drainage and aerate soil"),
    "sandy_low_retention": ("Sandy soil has low water retention",
                       "Use organic mulch and highly frequent, low-volume irrigation"),
    "fallback":       ("Suboptimal environmental conditions",
                       "Optimize irrigation, shading, and soil health"),
}


def identify_yield_reasons(
    fruit_count: int,
    ripeness_ratio: float,
    rainfall_30d: float,
    rainfall_90d: float,
    temperature: float,
    humidity: float,
    soil_score: float = DEFAULT_SOIL_SCORE,
    soil_type: str = "loamy",
    drainage: str = "moderate",
    predicted_yield: float = 0.0,
    break_even_yield: float = 0.0,
) -> list:
    """
    Intelligent reason identification engine.
    Returns a list of tuples: [(key, (reason, suggestion)), ...]
    """
    is_loss = predicted_yield < break_even_yield and break_even_yield > 0
    results = []

    # ── Core factor checks (relaxed thresholds) ──────────────────────
    if temperature > 30:
        results.append(("high_temp", _REASON_SUGGESTION_PAIRS["high_temp"]))

    if humidity < 40:
        results.append(("low_humidity", _REASON_SUGGESTION_PAIRS["low_humidity"]))

    if rainfall_30d < 50:
        results.append(("low_rain_30", _REASON_SUGGESTION_PAIRS["low_rain_30"]))

    if rainfall_90d < 150:
        results.append(("low_rain_90", _REASON_SUGGESTION_PAIRS["low_rain_90"]))

    if ripeness_ratio < 0.6:
        results.append(("low_ripeness", _REASON_SUGGESTION_PAIRS["low_ripeness"]))

    if fruit_count < 80:
        results.append(("low_fruit", _REASON_SUGGESTION_PAIRS["low_fruit"]))

    if soil_score < 0.7:
        results.append(("poor_soil", _REASON_SUGGESTION_PAIRS["poor_soil"]))

    if "clay" in soil_type.lower() and "poor" in drainage.lower():
        results.append(("waterlogging_risk", _REASON_SUGGESTION_PAIRS["waterlogging_risk"]))

    if "sandy" in soil_type.lower():
        results.append(("sandy_low_retention", _REASON_SUGGESTION_PAIRS["sandy_low_retention"]))

    # ── LOSS guarantee: always show at least 2 reasons ───────────────
    if is_loss and len(results) < 2:
        extras = [
            (fruit_count < 120, "low_fruit"),
            (ripeness_ratio < 0.8, "low_ripeness"),
            (rainfall_30d < 80, "low_rain_30"),
            (soil_score < 0.85, "poor_soil"),
        ]
        for condition, key in extras:
            if condition and all(r[0] != key for r in results):
                results.append((key, _REASON_SUGGESTION_PAIRS[key]))
            if len(results) >= 2:
                break

    # ── Fallback when LOSS but still no reasons ──────────────────────
    if is_loss and len(results) == 0:
        results.append(("fallback", _REASON_SUGGESTION_PAIRS["fallback"]))

    return results


def suggest_improvements(reason_pairs: list) -> list:
    """
    Extract suggestions from the reason-suggestion pair list.
    Accepts both new format (tuple pairs) and legacy format (plain strings).
    """
    suggestions = []
    for item in reason_pairs:
        if isinstance(item, tuple) and len(item) == 2:
            suggestions.append(item[1])
        elif isinstance(item, str):
            # Legacy fallback: lookup in old map (kept for backward compat)
            s = _LEGACY_SUGGESTION_MAP.get(item)
            if s:
                suggestions.append(s)
    return suggestions


def get_reason_texts(reason_pairs: list) -> list:
    """Extract just the reason strings from the pair list."""
    texts = []
    for item in reason_pairs:
        if isinstance(item, tuple) and len(item) == 2:
            texts.append(item[0])
        elif isinstance(item, str):
            texts.append(item)
    return texts


# Legacy map kept for any old callers
_LEGACY_SUGGESTION_MAP = {
    "Low fruit density":        "Improve fertilization and pruning practices",
    "Low ripeness ratio":       "Delay harvesting to allow proper ripening",
    "Low recent rainfall":      "Increase irrigation frequency",
    "Low seasonal rainfall":    "Adopt water conservation techniques",
    "High temperature stress":  "Use shade nets or mulching to reduce heat",
    "Low humidity":             "Use micro-irrigation or misting systems",
    "Poor soil quality affecting yield": "Apply organic compost or soil treatment",
}


# ─────────────────────────────────────────────
# POTENTIAL YIELD IMPROVEMENT
# ─────────────────────────────────────────────
def estimate_potential_yield(predicted_yield_kg: float, reasons: list) -> dict:
    """
    Estimate potential plant-level yield after implementing recommended improvements.
    More reasons → larger improvement factor (1.2 – 1.5).
    """
    n_issues = len(reasons)
    if n_issues == 0:
        factor = 1.05  # Minor optimization headroom
    elif n_issues <= 2:
        factor = 1.2
    elif n_issues <= 4:
        factor = 1.35
    else:
        factor = 1.5

    potential_yield = predicted_yield_kg * factor

    return {
        "potential_yield":     round(potential_yield, 3),
        "improvement_factor":  factor,
        "improvement_pct":     round((factor - 1) * 100, 1),
    }


# ─────────────────────────────────────────────
# Legacy helpers (kept for backward compat.)
# ─────────────────────────────────────────────
def calculate_canopy_coverage(image_path_or_array) -> float:
    """Calculate vegetation coverage using green-pixel segmentation in HSV."""
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
    else:
        img = image_path_or_array
    if img is None:
        return 0.0
    hsv          = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green  = np.array([35, 40, 40])
    upper_green  = np.array([85, 255, 255])
    mask         = cv2.inRange(hsv, lower_green, upper_green)
    veg_pixels   = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    return float(veg_pixels / total_pixels)


def extract_deep_features(image_path_or_array, device='cpu') -> np.ndarray:
    """Use EfficientNet-B0 as a pretrained feature extractor."""
    if isinstance(image_path_or_array, (str, Path)):
        img = Image.open(image_path_or_array).convert('RGB')
    else:
        img = Image.fromarray(cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    model      = get_effnet_extractor(device)
    with torch.no_grad():
        features = model(img_tensor).cpu().numpy().flatten()
    return features[:10]


# Re-export fetch_weather_features for callers that import it from here
from src.weather_service import fetch_weather_features  # noqa: E402

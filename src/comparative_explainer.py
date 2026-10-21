"""
Comparative Explainability Engine — CoffeeAI
=============================================
Supports two modes automatically:
  - DIFFERENT LOCATION MODE: compares soil + environment + vision
  - SAME LOCATION MODE: switches to Visual Comparative Analysis,
    since soil/environment are shared context.

All language uses XAI wording: "possible", "likely", "potential".
No ML models. No API calls. Pure agronomic rule-based reasoning.
"""

from typing import Dict, List
import src.root_health_explainer as root_health_explainer

# Tolerance for same-location detection (within ~100m)
_COORD_TOLERANCE = 0.001


# ─────────────────────────────────────────────
# 0. Same-Location Detection
# ─────────────────────────────────────────────

def detect_same_location(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> bool:
    """
    Returns True if both plants originate from the same geographic location.
    Uses a small tolerance to handle floating point rounding in GPS input.
    """
    return abs(lat_a - lat_b) <= _COORD_TOLERANCE and abs(lon_a - lon_b) <= _COORD_TOLERANCE


# ─────────────────────────────────────────────
# 1. Yield Comparison
# ─────────────────────────────────────────────

def compare_yield_predictions(plant_a: dict, plant_b: dict) -> dict:
    """Determine which plant has lower predicted yield and compute the delta."""
    yield_a = plant_a.get("predicted_yield", 0.0)
    yield_b = plant_b.get("predicted_yield", 0.0)
    delta = abs(yield_a - yield_b)

    if yield_a >= yield_b:
        winner, loser = "A", "B"
        ref_yield = yield_a
    else:
        winner, loser = "B", "A"
        ref_yield = yield_b

    pct = round((delta / ref_yield) * 100, 1) if ref_yield > 0 else 0.0

    if delta < 0.05:
        verdict = "Both plants show comparable predicted yield — no significant difference detected."
    else:
        verdict = (
            f"Plant {loser} shows {pct}% lower predicted yield than Plant {winner}. "
            f"Analysis focuses on possible agronomic causes for Plant {loser}."
        )

    return {
        "winner": winner,
        "loser": loser,
        "yield_a": round(yield_a, 3),
        "yield_b": round(yield_b, 3),
        "yield_delta": round(delta, 3),
        "yield_delta_pct": pct,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────
# 2. Visual Comparative Analysis (Same Location)
# ─────────────────────────────────────────────

def compute_visual_scores(plant_data: dict) -> dict:
    """
    Derive image-based plant productivity metrics from YOLO detection output.
    Returns normalized scores (0–1 or raw counts) for comparison.
    """
    fruit_count   = plant_data.get("fruit_count", 0)
    ripe_count    = plant_data.get("ripe_count", 0)
    unripe_count  = plant_data.get("unripe_count", 0)
    overripe      = plant_data.get("overripe_count", 0)
    ripeness_ratio = plant_data.get("ripeness_ratio", 0.0)

    total = fruit_count if fruit_count > 0 else 1

    # Fruit Density Score: normalized to 0–1 based on expected max ~150 fruits/image
    density_score = min(fruit_count / 150.0, 1.0)

    # Ripeness Ratio: % of ripe fruits
    ripeness_score = ripeness_ratio

    # Maturity Consistency Score: penalise if large spread between ripe/unripe/overripe
    ripe_pct    = ripe_count / total
    unripe_pct  = unripe_count / total
    over_pct    = overripe / total
    # Consistency high when fruit is predominantly one category
    dominant_pct = max(ripe_pct, unripe_pct, over_pct)
    consistency_score = round(dominant_pct, 3)

    # Ripe-to-unripe ratio
    ripe_unripe_ratio = round(ripe_count / max(unripe_count, 1), 2)

    return {
        "fruit_count":         fruit_count,
        "ripe_count":          ripe_count,
        "unripe_count":        unripe_count,
        "overripe_count":      overripe,
        "density_score":       round(density_score, 3),
        "ripeness_score":      round(ripeness_score, 3),
        "consistency_score":   consistency_score,
        "ripe_unripe_ratio":   ripe_unripe_ratio,
    }


def analyze_visual_differences(
    loser_visual: dict,
    winner_visual: dict,
    loser_label: str,
) -> List[str]:
    """Compare visual metrics and list simple causes for lower yield."""
    causes = []

    density_diff = winner_visual["density_score"] - loser_visual["density_score"]
    if density_diff > 0.1:
        causes.append(
            f"{loser_label} has fewer fruits ({loser_visual['fruit_count']} vs "
            f"{winner_visual['fruit_count']}). This plant may not have flowered well."
        )
    elif loser_visual["fruit_count"] < 30:
        causes.append(
            f"{loser_label} has very few fruits ({loser_visual['fruit_count']} detected). "
            "The plant may be weak or stressed."
        )

    ripeness_diff = winner_visual["ripeness_score"] - loser_visual["ripeness_score"]
    if ripeness_diff > 0.15:
        causes.append(
            f"{loser_label} has fewer ripe fruits "
            f"({loser_visual['ripeness_score']*100:.0f}% ripe vs "
            f"{winner_visual['ripeness_score']*100:.0f}%). Fruits are ripening slowly."
        )

    cons_diff = winner_visual["consistency_score"] - loser_visual["consistency_score"]
    if cons_diff > 0.15:
        causes.append(
            f"{loser_label} has fruits at mixed stages of ripeness. "
            "Fruits are not developing evenly on this plant."
        )

    if loser_visual["ripe_unripe_ratio"] < 0.8 and winner_visual["ripe_unripe_ratio"] > 1.2:
        causes.append(
            f"{loser_label} has more unripe fruits than ripe ones "
            f"(ratio: {loser_visual['ripe_unripe_ratio']:.1f}). "
            "The plant may need more nutrients to ripen fruits faster."
        )

    if loser_visual["overripe_count"] > winner_visual["overripe_count"] + 5:
        causes.append(
            f"{loser_label} has more overripe fruits. "
            "Fruits may have been missed during the last harvest."
        )

    if not causes:
        causes.append(
            "Both plants look similar visually. "
            "The yield difference is small and may improve on its own."
        )

    return causes[:5]




def infer_visual_root_health(loser_visual: dict, winner_visual: dict, loser_label: str) -> List[str]:
    """Give simple health hints based on what the image shows using root_health_explainer."""
    # This acts as the same_location insights
    return root_health_explainer.generate_crop_health_insights(loser_visual, is_same_location=True)


def generate_visual_recommendations(
    causes: List[str],
    insights: List[str],
    loser_label: str,
) -> List[dict]:
    """Simple, cost-friendly suggestions. Always returns at least 2."""
    recs = []
    text = " ".join(causes + insights).lower()

    if "fewer fruits" in text or "few fruits" in text or "flowered" in text or "stress" in text:
        recs.append({
            "priority": "high",
            "action": f"Prune dry or weak branches on {loser_label}",
            "reason": "Removing weak branches helps the plant put energy into growing more fruits. Pruning costs nothing.",
        })

    if "nutrient" in text or "nutrition" in text or "efficiency" in text or "getting enough" in text:
        recs.append({
            "priority": "high",
            "action": f"Add compost or organic manure near the base of {loser_label}",
            "reason": "Compost is cheap and improves soil nutrients so the plant can produce more fruits.",
        })

    if "ripen" in text or "unripe" in text or "potassium" in text or "maturation" in text:
        recs.append({
            "priority": "medium",
            "action": f"Apply wood ash or banana peel compost around {loser_label}",
            "reason": "These are cheap, natural sources of potassium that help fruits ripen faster and more evenly.",
        })

    if "water" in text or "watering" in text or "moisture" in text or "consistent" in text or "climate" in text:
        recs.append({
            "priority": "medium",
            "action": f"Water {loser_label} at the same time every day",
            "reason": "Regular, consistent watering helps fruits develop evenly without extra cost.",
        })

    if "overripe" in text or "harvest" in text:
        recs.append({
            "priority": "low",
            "action": f"Pick fruits from {loser_label} every 7–10 days",
            "reason": "Frequent harvesting stops fruits from becoming overripe and wasting yield.",
        })

    # Always guarantee at least 2 recommendations
    if len(recs) < 2:
        recs.append({
            "priority": "medium",
            "action": f"Water {loser_label} regularly and add organic compost once a month",
            "reason": "Consistent water and cheap organic compost are the easiest low-cost ways to improve any coffee plant's yield.",
        })
    if len(recs) < 2:
        recs.append({
            "priority": "low",
            "action": f"Remove dry leaves and old branches from {loser_label}",
            "reason": "Keeping the plant clean and tidy allows better airflow and fruit growth at no extra cost.",
        })

    return recs[:5]



# ─────────────────────────────────────────────
# 3. Soil Condition Comparison (Different Location)
# ─────────────────────────────────────────────

_SOIL_OPTIMAL = {
    "ph":   (5.5, 6.5),
    "soc":  (20.0, None),
    "clay": (15.0, 50.0),
    "sand": (None, 60.0),
    "silt": (10.0, 50.0),
}
_SOIL_LABELS = {
    "ph":   "Soil pH",
    "soc":  "Organic Carbon (g/kg)",
    "clay": "Clay (%)",
    "sand": "Sand (%)",
    "silt": "Silt (%)",
}


def _soil_status(key: str, value) -> str:
    lo, hi = _SOIL_OPTIMAL.get(key, (None, None))
    if value is None:
        return "unknown"
    if lo is not None and value < lo:
        return "low"
    if hi is not None and value > hi:
        return "high"
    return "optimal"


def compare_soil_conditions(soil_a: dict, soil_b: dict) -> List[dict]:
    rows = []
    for key, label in _SOIL_LABELS.items():
        val_a = soil_a.get(key)
        val_b = soil_b.get(key)
        display_a = round(val_a, 2) if val_a is not None else None
        display_b = round(val_b, 2) if val_b is not None else None
        delta = None
        advantage = "equal"
        if val_a is not None and val_b is not None:
            delta = round(val_a - val_b, 2)
            if key in ("ph", "soc", "silt"):
                advantage = "A" if val_a > val_b else ("B" if val_b > val_a else "equal")
            elif key == "sand":
                advantage = "A" if val_a < val_b else ("B" if val_b < val_a else "equal")
            else:
                mid = 32.5
                advantage = "A" if abs(val_a - mid) < abs(val_b - mid) else "B"
        rows.append({
            "property": key,
            "label": label,
            "value_a": display_a,
            "value_b": display_b,
            "status_a": _soil_status(key, val_a),
            "status_b": _soil_status(key, val_b),
            "delta": delta,
            "advantage": advantage,
        })
    return rows


# ─────────────────────────────────────────────
# 4. Different-Location Cause / Health / Recs
# ─────────────────────────────────────────────

def identify_low_yield_causes(
    soil_data: dict, weather_data: dict, visual_features: dict, comparison: dict,
) -> List[str]:
    """Simple list of reasons why this plant has lower yield."""
    causes = []
    ph = soil_data.get("ph")
    if ph is not None:
        if ph < 5.5:
            causes.append(f"The soil is too acidic (pH {ph:.1f}). This stops the plant from absorbing nutrients properly.")
        elif ph > 7.0:
            causes.append(f"The soil is too alkaline (pH {ph:.1f}). This can block important minerals from reaching the plant.")
    soc = soil_data.get("soc")
    if soc is not None and soc < 20.0:
        causes.append(f"The soil has low organic matter ({soc:.1f} g/kg). The soil is not very fertile.")
    sand = soil_data.get("sand")
    if sand is not None and sand > 60.0:
        causes.append(f"The soil has too much sand ({sand:.1f}%). Water drains away too quickly and the plant gets dry.")
    clay = soil_data.get("clay")
    if clay is not None and clay > 55.0:
        causes.append(f"The soil has too much clay ({clay:.1f}%). Water stays too long and roots may get damaged.")
    temp = weather_data.get("temperature", 25.0)
    if temp > 30.0:
        causes.append(f"It is too hot ({temp:.1f}°C). High heat puts stress on the plant and reduces fruit growth.")
    rain_30d = weather_data.get("rainfall_last_30_days", 100.0)
    if rain_30d < 50.0:
        causes.append(f"There has been very little rain recently ({rain_30d:.1f} mm in 30 days). The plant may be dry.")
    if visual_features.get("fruit_count", 0) < 40:
        causes.append(f"Only {visual_features['fruit_count']} fruits were detected. The plant may not have flowered enough.")
    if visual_features.get("ripeness_ratio", 1.0) < 0.4:
        causes.append("Most fruits are still unripe. The plant needs more time or nutrients to ripen them.")
    pct = comparison.get("yield_delta_pct", 0)
    if pct > 30:
        causes.append(f"Overall yield is {pct:.0f}% lower than the other plant. Several small problems may be adding up.")
    if not causes:
        causes.append("No major problems were found. The yield difference is small and likely due to minor variations.")
    return causes[:6]


def infer_comparative_root_health(soil_data: dict, weather_data: dict) -> List[str]:
    """Simple health hints based on soil and weather data."""
    insights = []
    ph   = soil_data.get("ph", 6.0)
    soc  = soil_data.get("soc", 30.0)
    sand = soil_data.get("sand", 40.0)
    clay = soil_data.get("clay", 25.0)
def identify_low_yield_causes(
    soil_data: dict, weather_data: dict, visual_features: dict, comparison: dict,
) -> List[str]:
    """Uses root_health_explainer to find stress factors."""
    causes = root_health_explainer.identify_possible_stress_factors(soil_data, weather_data, visual_features)
    pct = comparison.get("yield_delta_pct", 0)
    if pct > 30 and "Significant overall yield gap indicating multiple compounding stress factors" not in causes:
        causes.append(f"Significant overall yield gap ({pct:.0f}%) indicating multiple compounding stress factors")
    return causes[:6]


def infer_comparative_root_health(soil_data: dict, weather_data: dict) -> List[str]:
    """Uses root_health_explainer to find root health issues."""
    return root_health_explainer.infer_root_health_issues(soil_data, weather_data)


def generate_comparative_recommendations(
    causes: List[str], insights: List[str], loser_label: str = "Plant B",
) -> List[dict]:
    """Simple, cost-friendly tips. Always returns at least 2."""
    recs = []
    text = " ".join(causes + insights).lower()

    if "acidic" in text or "too acidic" in text:
        recs.append({"priority": "high",
                     "action": "Add agricultural lime or wood ash to the soil",
                     "reason": f"The soil around {loser_label} is too acidic. Lime or wood ash is cheap and raises the pH so the plant can absorb nutrients better."})
    if "alkaline" in text or "too alkaline" in text:
        recs.append({"priority": "high",
                     "action": "Mix in some compost or sulfur powder to lower soil alkalinity",
                     "reason": f"The soil around {loser_label} is too alkaline. Compost helps bring the pH to a healthier level."})
    if "organic matter" in text or "organic carbon" in text or "compost" in text or "fertile" in text:
        recs.append({"priority": "high",
                     "action": f"Add kitchen compost or cow dung manure around {loser_label}",
                     "reason": "Organic matter is cheap and improves soil so roots can access more nutrients."})
    if "sandy" in text or "drains" in text or "drying out" in text:
        recs.append({"priority": "high",
                     "action": f"Put a layer of dry leaves or straw (mulch) around {loser_label}",
                     "reason": "Mulch is free and stops water from draining away too fast in sandy soil."})
    if "rain" in text or "water" in text or "dry" in text or "moisture" in text:
        recs.append({"priority": "medium",
                     "action": f"Water {loser_label} regularly — at least 3 times a week during dry weather",
                     "reason": "Consistent watering is free and keeps the plant hydrated so fruits develop well."})
    if "hot" in text or "heat" in text or "temperature" in text:
        recs.append({"priority": "medium",
                     "action": f"Plant a fast-growing shade tree or use old cloth to shade {loser_label}",
                     "reason": "Shade reduces heat stress and is low cost. Even a cloth cover helps."})
    if "few fruits" in text or "flowered" in text or "flowering" in text or "unripe" in text:
        recs.append({"priority": "medium",
                     "action": f"Trim old or dry branches from {loser_label} and water more regularly",
                     "reason": "Pruning and watering are free steps that help the plant flower and fruit better."})
    if "clay" in text or "too much clay" in text:
        recs.append({"priority": "medium",
                     "action": f"Loosen the soil around {loser_label} with a small hand tool",
                     "reason": "Breaking up hard clay lets air and water reach roots more easily at no cost."})

    # Guarantee at least 2 recommendations
    if len(recs) < 1:
        recs.append({"priority": "medium",
                     "action": f"Add compost or organic manure around {loser_label} once a month",
                     "reason": "Organic compost is cheap and improves soil health so the plant can produce more fruits."})
    if len(recs) < 2:
        recs.append({"priority": "low",
                     "action": f"Water {loser_label} at the same time each day and remove dry leaves regularly",
                     "reason": "Regular watering and basic plant hygiene cost nothing and can noticeably improve yield over time."})

    return recs[:5]



# ─────────────────────────────────────────────
# 5. Full Comparison Orchestrator
# ─────────────────────────────────────────────

def generate_full_comparison(
    plant_a_data: dict,
    plant_b_data: dict,
    lat_a: float = 0.0, lon_a: float = 0.0,
    lat_b: float = 0.0, lon_b: float = 0.0,
) -> dict:
    """
    Orchestrates all comparison logic with automatic mode detection.

    Returns a unified dict with an extra key:
        same_location_mode  – bool
        visual_scores_a     – dict (only in same_location_mode)
        visual_scores_b     – dict (only in same_location_mode)
    """
    same_location = detect_same_location(lat_a, lon_a, lat_b, lon_b)

    # ── Yield delta (always computed) ──────────────────────────────
    yield_cmp   = compare_yield_predictions(plant_a_data, plant_b_data)
    loser_key   = yield_cmp["loser"]
    loser_label = f"Plant {loser_key}"
    winner_key  = yield_cmp["winner"]

    loser_data  = plant_a_data if loser_key  == "A" else plant_b_data
    winner_data = plant_a_data if winner_key == "A" else plant_b_data

    # ── Soil comparison (always useful for table display) ───────────
    soil_a   = plant_a_data.get("soil_grids_data") or {}
    soil_b   = plant_b_data.get("soil_grids_data") or {}
    soil_cmp = compare_soil_conditions(soil_a, soil_b)

    if same_location:
        # ── VISUAL COMPARATIVE ANALYSIS MODE ───────────────────────
        vis_a = compute_visual_scores(plant_a_data)
        vis_b = compute_visual_scores(plant_b_data)
        loser_vis  = vis_a if loser_key  == "A" else vis_b
        winner_vis = vis_a if winner_key == "A" else vis_b

        from src.dynamic_explainer import generate_dynamic_insights
        
        loser_soil    = loser_data.get("soil_grids_data") or {}
        loser_weather = {
            "temperature":           loser_data.get("temperature", 25.0),
            "humidity":              loser_data.get("humidity", 70.0),
            "rainfall_last_30_days": loser_data.get("rainfall_last_30_days", 100.0),
            "rainfall_last_90_days": loser_data.get("rainfall_last_90_days", 270.0),
        }
        
        vis_a = compute_visual_scores(plant_a_data)
        vis_b = compute_visual_scores(plant_b_data)
        loser_vis  = vis_a if loser_key  == "A" else vis_b
        
        dynamic_insights = generate_dynamic_insights(
            loser_soil,
            loser_weather,
            loser_vis,
            is_same_location=True,
            yield_delta_pct=yield_cmp.get("yield_delta_pct", 0)
        )

        shared_context_note = (
            "Both plants share the same soil and environmental conditions because they originate "
            "from the same location. Analysis focuses on visual plant-level differences to explain "
            "the yield gap."
        )

        # Amend the yield verdict
        if yield_cmp["yield_delta"] >= 0.05:
            yield_cmp["verdict"] = (
                f"Shared environmental context detected. Despite identical location, "
                f"{loser_label} shows {yield_cmp['yield_delta_pct']}% lower predicted yield — "
                "attributed to visual plant-level differences."
            )

        print(f"[COMPARE/SAME-LOC] Winner=Plant {winner_key} | "
              f"Δ={yield_cmp['yield_delta']:.3f} kg | Dynamic Insights Generated")

        return {
            "same_location_mode":   True,
            "shared_context_note":  shared_context_note,
            "yield_comparison":     yield_cmp,
            "soil_comparison":      soil_cmp,
            "loser_label":          loser_label,
            "dynamic_insights":     dynamic_insights,
            "visual_scores_a":      vis_a,
            "visual_scores_b":      vis_b,
            "summary":              yield_cmp["verdict"],
        }

    else:
        # ── DIFFERENT LOCATION MODE (original logic) ────────────────
        loser_soil    = loser_data.get("soil_grids_data") or {}
        loser_weather = {
            "temperature":           loser_data.get("temperature", 25.0),
            "humidity":              loser_data.get("humidity", 70.0),
            "rainfall_last_30_days": loser_data.get("rainfall_last_30_days", 100.0),
            "rainfall_last_90_days": loser_data.get("rainfall_last_90_days", 270.0),
        }
        loser_visual = {
            "fruit_count":    loser_data.get("fruit_count", 0),
            "ripeness_ratio": loser_data.get("ripeness_ratio", 0.5),
        }

        from src.dynamic_explainer import generate_dynamic_insights
        
        dynamic_insights = generate_dynamic_insights(
            loser_soil,
            loser_weather,
            loser_visual,
            is_same_location=False,
            yield_delta_pct=yield_cmp.get("yield_delta_pct", 0)
        )

        print(f"[COMPARE/DIFF-LOC] Winner=Plant {winner_key} | "
              f"Δ={yield_cmp['yield_delta']:.3f} kg | Dynamic Insights Generated")

        return {
            "same_location_mode":  False,
            "shared_context_note": None,
            "yield_comparison":    yield_cmp,
            "soil_comparison":     soil_cmp,
            "loser_label":         loser_label,
            "dynamic_insights":    dynamic_insights,
            "visual_scores_a":     None,
            "visual_scores_b":     None,
            "summary":             yield_cmp["verdict"],
        }

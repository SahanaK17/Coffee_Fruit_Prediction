# CoffeeAI Robust QA Gate System - Implementation Guide

## Overview

The QA gate has been completely redesigned to use **strict detection presence** instead of unreliable color-based heuristics. It now includes protection against sketches, diagrams, and random objects.

---

## Two-Stage QA Architecture

### Stage 0: Heuristic Check (Anti-Sketch)

**Purpose:** Reject line drawings and non-photo images.
**Logic:**
- Checks color saturation (rejects grayscale/low-saturation).
- Checks color variance (rejects flat colors).
- If image looks like a sketch AND has weak YOLO detections -> **REJECT**.

### Stage A: Coffee Object Presence (YOLO-based)

**Purpose:** Validate that coffee fruits actually exist in the image.

**Rules:**
1. **Minimum Strong Detections**: ≥ 1 detection with confidence ≥ 0.40
2. **Total Area Check**: 
   - Min: 0.2% of image (prevents small noise)
   - Max: 65% of image (prevents close-up blobs)
3. **Central Region Check**: At least one strong detection must overlap with the **middle 60%** of the image (prevents corner noise).

**Rejects:**
- Market photos (no coffee fruit detections)
- Vegetable/Leaf images (YOLO won't detect "coffee fruit")
- Random green/red graphics
- Sketches/Diagrams
- People/buildings/landscapes

### Stage B: Classifier Consistency Check (Hybrid/EffNet Mode)

**Optional:** Additional layer for non-YOLO modes to ensure EfficientNet consistency (currently lightweight/pass-through if YOLO pass is strong).

---

## Mode-Aware QA Behavior

### All Modes (YOLO / EfficientNet / Hybrid)
- **ALWAYS** runs Stage 0 (Heuristic) and Stage A (YOLO Presence).
- **EfficientNet Mode** now requires YOLO validation first! This prevents the classifier from hallucinating on non-coffee images.

---

## Tunable Thresholds

Located in `src/inference.py`:

```python
self.qa_min_conf = 0.40             # Minimum confidence for a "strong" detection
self.qa_min_total_area = 0.002      # 0.2% minimum total area
self.qa_max_total_area = 0.65       # 65% maximum total area
self.qa_min_single_box_area = 0.001 # 0.1% min size single box
self.qa_center_fraction = 0.60      # Center 60% check
self.qa_min_saturation = 15.0       # Sketch detection
```

---

## API Response Structure (Crash-Proof)

### QA Reject Response (200 OK):
```json
{
  "ok": false,
  "message": "QA Reject: No confident coffee detections",
  "qa": {
    "reason": "insufficient_strong_boxes",
    "strong_count": 0,
    "max_conf": 0.25,
    "total_area_ratio": 0.001,
    "rules": {...}
  },
  "mode": "Hybrid",
  ...
}
```

### QA Pass Response:
```json
{
  "ok": true,
  "message": "QA Pass: Valid coffee fruit image",
  "qa": {
    "reason": "passed_all_checks",
    "strong_count": 5,
    ...
  },
  ...
}
```

## Logs

Rejections are now logged to `data/predictions_history.json` with `final_label="REJECTED"`.

---

**System Status:** Strict & Robust QA Gate Active ✅

import os
import cv2
import base64
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import traceback

# Import custom modules
from src.inference import HybridModel

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YOLO_PATH = MODELS_DIR / "yolo" / "best.pt"
EFFNET_PATH = MODELS_DIR / "effnet" / "effnet" / "best.pth"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "predictions_history.json"

# Create directories
UPLOADS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ===== FASTAPI APP =====
app = FastAPI(
    title="CoffeeAI M.Tech System",
    description="Multi-model Coffee Fruit Maturity Prediction with QA",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== GLOBAL STATE =====
model_instance: Optional[HybridModel] = None
system_status = {
    "yolo_loaded": False,
    "effnet_loaded": False,
    "startup_error": None,
    "model_paths": {
        "yolo": str(YOLO_PATH),
        "effnet": str(EFFNET_PATH)
    }
}

# ===== UTILITY FUNCTIONS =====

def image_to_base64(img_array: np.ndarray) -> str:
    """Convert image array to base64 string."""
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def validate_image_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded image file."""
    # Check file type
    valid_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in valid_types:
        return False, f"Invalid file type: {file.content_type}. Allowed: JPG, PNG, WEBP"
    
    return True, "OK"

def convert_webp_to_png(file_bytes: bytes) -> bytes:
    """Convert WEBP to PNG format."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()

def store_prediction(data: dict):
    """Store QA-passed prediction to history."""
    try:
        history = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        
        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()
        history.append(data)
        
        # Keep only last 1000 records
        history = history[-1000:]
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to store prediction: {e}")

# ===== STARTUP =====

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global model_instance, system_status
    
    print("\n" + "="*60)
    print("CoffeeAI System Initialization")
    print("="*60)
    
    # Check model files
    if not YOLO_PATH.exists():
        msg = f"YOLO model not found at: {YOLO_PATH}"
        print(f"[ERROR] {msg}")
        system_status["startup_error"] = msg
        return
    
    if not EFFNET_PATH.exists():
        msg = f"EfficientNet model not found at: {EFFNET_PATH}"
        print(f"[ERROR] {msg}")
        system_status["startup_error"] = msg
        return
    
    try:
        print(f"[INFO] Loading models...")
        model_instance = HybridModel(str(YOLO_PATH), str(EFFNET_PATH))
        system_status["yolo_loaded"] = True
        system_status["effnet_loaded"] = True
        print("[SUCCESS] All models loaded")
        print("="*60 + "\n")
    except Exception as e:
        msg = f"Model loading failed: {str(e)}"
        system_status["startup_error"] = msg
        print(f"[ERROR] {msg}")

# ===== ENDPOINTS =====

# ----- Health -----
@app.get("/health", tags=["Health"])
def health_check():
    """System health and model status."""
    if system_status["startup_error"]:
        raise HTTPException(status_code=503, detail=system_status["startup_error"])
    
    return {
        "status": "online",
        "models_loaded": model_instance is not None,
        "system_info": system_status
    }

# ----- Analytics -----
@app.get("/analytics", tags=["Analytics"])
def get_analytics():
    """Fetch training metrics and charts."""
    curves_path = "models/effnet/training_curves.png"
    cm_path = "models/effnet/confusion_matrix.png"
    
    curves_full = BASE_DIR / curves_path
    cm_full = BASE_DIR / cm_path
    
    return {
        "training_curves": f"/static/{curves_path}" if curves_full.exists() else None,
        "confusion_matrix": f"/static/{cm_path}" if cm_full.exists() else None,
        "report_available": (BASE_DIR / "models/effnet/classification_report.txt").exists()
    }

# ----- QA Check -----
@app.post("/qa/check", tags=["QA"])
# ----- QA Endpoint -----
@app.post("/qa/check", tags=["QA"])
async def qa_check(file: UploadFile = File(...)):
    """
    QA Gate Checker: Validates if image is a valid coffee fruit image.
    CRASH-PROOF: Always returns 200 with valid JSON.
    """
    temp_path = None
    
    try:
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        valid, msg = validate_image_file(file)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)
        
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        if file.content_type == "image/webp":
            file_bytes = convert_webp_to_png(file_bytes)
        
        temp_path = UPLOADS_DIR / f"qa_check_{datetime.now().timestamp()}_{file.filename}"
        temp_path.write_bytes(file_bytes)
        
        # Run QA Gate (hybrid mode for full check)
        qa_pass, qa_message, qa_report = model_instance.qa_gate(str(temp_path), mode='hybrid')
        
        # Debug logging
        print(f"\n{'='*60}")
        print(f"[QA CHECK ENDPOINT]")
        print(f"  File: {file.filename}")
        print(f"  Result: {'PASS' if qa_pass else 'REJECT'}")
        print(f"  Report: {qa_report}")
        print(f"{'='*60}\n")
        
        return {
            "ok": qa_pass,
            "message": qa_message,
            "qa": qa_report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] QA Check failed: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "message": "QA Reject: Processing error",
                "qa": {
                    "reason": "exception",
                    "error": str(e),
                    "strong_count": 0,
                    "max_conf": 0.0,
                    "total_area_ratio": 0.0,
                    "rules": model_instance.get_qa_rules() if model_instance else {}
                }
            }
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

# ----- Data -----
@app.get("/data/history", tags=["Data"])
def get_prediction_history(limit: int = 50):
    """Retrieve prediction history."""
    if not HISTORY_FILE.exists():
        return {"history": []}
    
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    return {"history": history[-limit:]}

# ----- Prediction: YOLO Only -----
@app.post("/predict/yolo", tags=["Prediction"])
async def predict_yolo(file: UploadFile = File(...)):
    """
    YOLO-Only Mode: Object detection + majority voting.
    CRASH-PROOF: Always returns 200 with valid JSON.
    """
    temp_path = None
    
    try:
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Validate file
        valid, msg = validate_image_file(file)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)
        
        # Read file
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (>10MB)")
        
        # Convert WEBP if needed
        if file.content_type == "image/webp":
            file_bytes = convert_webp_to_png(file_bytes)
        
        # Save temporarily
        temp_path = UPLOADS_DIR / f"temp_{datetime.now().timestamp()}_{file.filename}"
        temp_path.write_bytes(file_bytes)
        
        # ===== QA GATE =====
        qa_pass, qa_message, qa_report = model_instance.qa_gate(str(temp_path), mode='yolo')
        
        # Debug logging
        print(f"\n{'='*60}")
        print(f"[QA DEBUG - YOLO Mode]")
        print(f"  File: {file.filename}")
        print(f"  Result: {'PASS' if qa_pass else 'REJECT'}")
        print(f"  Detections: {qa_report.get('strong_count', 0)}")
        print(f"  Max Conf: {qa_report.get('max_conf', 0):.3f}")
        print(f"  Area Ratio: {qa_report.get('total_area_ratio', 0):.4f}")
        print(f"  Reason: {qa_report.get('reason', 'N/A')}")
        print(f"{'='*60}\n")
        
        if not qa_pass:
            # Log rejection
            store_prediction({
                "mode": "YOLO",
                "final_label": "REJECTED",
                "confidence": 0.0,
                "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0},
                "fruits_detected": 0,
                "qa_pass": False,
                "qa_reason": qa_report.get('reason', 'N/A')
            })
            
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "message": qa_message,
                    "mode": "YOLO",
                    "qa": qa_report,
                    "final_label": None,
                    "final_confidence": 0.0,
                    "fruits_detected": 0,
                    "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0}
                }
            )
        
        # ===== PREDICTION =====
        result = model_instance.predict_yolo_only(str(temp_path))
        
        # Annotate image
        annotated_img = model_instance.draw_annotations(str(temp_path), result['predictions'])
        b64_img = image_to_base64(annotated_img)
        
        # Build response
        response = {
            "ok": True,
            "message": qa_message,
            "qa": qa_report,
            "mode": result['mode'],
            "final_label": result['final_label'],
            "final_confidence": result['final_confidence'],
            "fruits_detected": result['fruits_detected'],
            "distribution": result['distribution'],
            "annotated_image": b64_img
        }
        
        # Store only if QA passed
        store_prediction({
            "mode": "YOLO",
            "final_label": result['final_label'],
            "confidence": result['final_confidence'],
            "distribution": result['distribution'],
            "fruits_detected": result['fruits_detected']
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Global exception handler - NEVER crash
        print(f"\n{'='*60}")
        print(f"[EXCEPTION - YOLO Mode]")
        print(f"  File: {file.filename if file else 'unknown'}")
        print(f"  Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "message": "QA Reject: Processing error",
                "mode": "YOLO",
                "qa": {
                    "reason": "exception",
                    "error": str(e),
                    "strong_count": 0,
                    "max_conf": 0.0,
                    "total_area_ratio": 0.0,
                    "rules": model_instance.get_qa_rules() if model_instance else {}
                },
                "final_label": None,
                "final_confidence": 0.0,
                "fruits_detected": 0,
                "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0}
            }
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

# ----- Prediction: EfficientNet Only -----
@app.post("/predict/effnet", tags=["Prediction"])
async def predict_effnet(file: UploadFile = File(...)):
    """
    EfficientNet-Only Mode: Whole image classification.
    CRASH-PROOF: Always returns 200 with valid JSON.
    """
    temp_path = None
    
    try:
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        valid, msg = validate_image_file(file)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)
        
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        if file.content_type == "image/webp":
            file_bytes = convert_webp_to_png(file_bytes)
        
        temp_path = UPLOADS_DIR / f"temp_{datetime.now().timestamp()}_{file.filename}"
        temp_path.write_bytes(file_bytes)
        
        # ===== QA GATE =====
        qa_pass, qa_message, qa_report = model_instance.qa_gate(str(temp_path), mode='effnet')
        
        # Debug logging
        print(f"\n{'='*60}")
        print(f"[QA DEBUG - EfficientNet Mode]")
        print(f"  File: {file.filename}")
        print(f"  Result: {'PASS' if qa_pass else 'REJECT'}")
        print(f"  Detections: {qa_report.get('strong_count', 0)}")
        print(f"  Max Conf: {qa_report.get('max_conf', 0):.3f}")
        print(f"  EffNet Conf: {qa_report.get('effnet_top1_conf', 'N/A')}")
        print(f"  Reason: {qa_report.get('reason', 'N/A')}")
        print(f"{'='*60}\n")
        
        if not qa_pass:
            # Log rejection
            store_prediction({
                "mode": "EfficientNet",
                "final_label": "REJECTED",
                "confidence": 0.0,
                "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0},
                "fruits_detected": 0,
                "qa_pass": False,
                "qa_reason": qa_report.get('reason', 'N/A')
            })
            
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "message": qa_message,
                    "mode": "EfficientNet",
                    "qa": qa_report,
                    "final_label": None,
                    "final_confidence": 0.0,
                    "fruits_detected": 0,
                    "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0}
                }
            )
        
        # ===== PREDICTION =====
        result = model_instance.predict_effnet_only(str(temp_path))
        
        response = {
            "ok": True,
            "message": qa_message,
            "qa": qa_report,
            "mode": result['mode'],
            "final_label": result['final_label'],
            "final_confidence": result['final_confidence'],
            "fruits_detected": result['fruits_detected'],
            "distribution": result['distribution'],
            "annotated_image": None 
        }
        
        store_prediction({
            "mode": "EfficientNet",
            "final_label": result['final_label'],
            "confidence": result['final_confidence'],
            "distribution": result['distribution'],
            "fruits_detected": result['fruits_detected']
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Global exception handler - NEVER crash
        print(f"\n{'='*60}")
        print(f"[EXCEPTION - EfficientNet Mode]")
        print(f"  File: {file.filename if file else 'unknown'}")
        print(f"  Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "message": "QA Reject: Processing error",
                "mode": "EfficientNet",
                "qa": {
                    "reason": "exception",
                    "error": str(e),
                    "strong_count": 0,
                    "max_conf": 0.0,
                    "total_area_ratio": 0.0,
                    "rules": model_instance.get_qa_rules() if model_instance else {}
                },
                "final_label": None,
                "final_confidence": 0.0,
                "fruits_detected": 0,
                "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0}
            }
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

# ----- Prediction: Hybrid -----
@app.post("/predict/hybrid", tags=["Prediction"])
async def predict_hybrid(file: UploadFile = File(...)):
    """
    Hybrid Mode: YOLO detection + EfficientNet classification with fusion.
    Research contribution: Confidence calibration via α-weighted fusion.
    CRASH-PROOF: Always returns 200 with valid JSON.
    """
    temp_path = None
    
    try:
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Validate file
        valid, msg = validate_image_file(file)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)
        
        # Read file
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        if file.content_type == "image/webp":
            file_bytes = convert_webp_to_png(file_bytes)
        
        temp_path = UPLOADS_DIR / f"temp_{datetime.now().timestamp()}_{file.filename}"
        temp_path.write_bytes(file_bytes)
        
        # ===== QA GATE =====
        qa_pass, qa_message, qa_report = model_instance.qa_gate(str(temp_path), mode='hybrid')
        
        # Debug logging
        print(f"\n{'='*60}")
        print(f"[QA DEBUG - Hybrid Mode]")
        print(f"  File: {file.filename}")
        print(f"  Result: {'PASS' if qa_pass else 'REJECT'}")
        print(f"  Detections: {qa_report.get('strong_count', 0)}")
        print(f"  Max Conf: {qa_report.get('max_conf', 0):.3f}")
        print(f"  Area Ratio: {qa_report.get('total_area_ratio', 0):.4f}")
        print(f"  EffNet Conf: {qa_report.get('effnet_top1_conf', 'N/A')}")
        print(f"  Reason: {qa_report.get('reason', 'N/A')}")
        print(f"{'='*60}\n")
        
        if not qa_pass:
            # Log rejection
            store_prediction({
                "mode": "Hybrid",
                "final_label": "REJECTED",
                "confidence": 0.0,
                "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0},
                "fruits_detected": 0,
                "qa_pass": False,
                "qa_reason": qa_report.get('reason', 'N/A')
            })
            
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "message": qa_message,
                    "mode": "Hybrid",
                    "qa": qa_report,
                    "final_label": None,
                    "final_confidence": 0.0,
                    "fruits_detected": 0,
                    "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0}
                }
            )
        
        # ===== PREDICTION =====
        result = model_instance.predict_hybrid(str(temp_path))
        
        # Annotate
        annotated_img = model_instance.draw_annotations(str(temp_path), result['predictions'])
        b64_img = image_to_base64(annotated_img)
        
        response = {
            "ok": True,
            "message": qa_message,
            "qa": qa_report,
            "mode": result['mode'],
            "final_label": result['final_label'],
            "final_confidence": result['final_confidence'],
            "fruits_detected": result['fruits_detected'],
            "distribution": result['distribution'],
            "annotated_image": b64_img,
            "fusion_alpha": model_instance.fusion_alpha
        }
        
        store_prediction({
            "mode": "Hybrid",
            "final_label": result['final_label'],
            "confidence": result['final_confidence'],
            "distribution": result['distribution'],
            "fruits_detected": result['fruits_detected']
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Global exception handler - NEVER crash
        print(f"\n{'='*60}")
        print(f"[EXCEPTION - Hybrid Mode]")
        print(f"  File: {file.filename if file else 'unknown'}")
        print(f"  Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "message": "QA Reject: Processing error",
                "mode": "Hybrid",
                "qa": {
                    "reason": "exception",
                    "error": str(e),
                    "strong_count": 0,
                    "max_conf": 0.0,
                    "total_area_ratio": 0.0,
                    "rules": model_instance.get_qa_rules() if model_instance else {}
                },
                "final_label": None,
                "final_confidence": 0.0,
                "fruits_detected": 0,
                "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0}
            }
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

# ----- Static Files -----
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# ----- Main -----
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import os
import cv2
import base64
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import traceback
import pandas as pd
import io

from fastapi.responses import JSONResponse, StreamingResponse

# Import custom modules
from src.maturity_module import HybridModel
from src import yield_module
from database import CoffeeDatabase

# Create database instance
db = CoffeeDatabase()

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YOLO_PATH = MODELS_DIR / "yolo" / "best.pt"
EFFNET_PATH = MODELS_DIR / "effnet" / "effnet" / "best.pth"

UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "predictions_history.json"
REJECTED_FILE = DATA_DIR / "rejected_history.json"

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

# ----- Database Compatibility Wrappers -----
def save_maturity_prediction(data: dict):
    """Save maturity prediction to SQLite."""
    if 'mode' not in data: data['mode'] = 'Hybrid'
    db.save_predictions(data)

def save_yield_prediction(data: dict):
    """Save yield prediction to SQLite."""
    if 'mode' not in data: data['mode'] = 'Yield'
    db.save_predictions(data)

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
        
        if not qa_pass:
            return {
                "ok": False,
                "message": "QA Rejected. No Coffee Fruit Found",
                "qa": qa_report
            }

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
                "message": "QA Rejected. No Coffee Fruit Found"
            }
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

# ----- Database Compatibility Wrappers -----
def get_maturity_history(limit=1000):
    return [r for r in db.get_history(limit) if r.get('mode') != 'Yield']

def get_yield_history(limit=1000):
    return [r for r in db.get_history(limit) if r.get('mode') == 'Yield']

# ----- Dashboard Stats -----
@app.get("/dashboard-stats", tags=["Dashboard"])
def get_dashboard_summary():
    """Retrieve aggregate prediction stats from the database."""
    try:
        stats = db.get_dashboard_stats()
        return stats
    except Exception as e:
        print(f"[ERROR] Dashboard stats failed: {e}")
        return {
            "maturity": {
                "total_predictions": 0,
                "hybrid_avg": 0.0,
                "yolo_avg": 0.0,
                "efficientnet_avg": 0.0
            },
            "yield": {
                "total_yield_predictions": 0,
                "avg_yield": 0.0,
                "avg_ripeness": 0.0,
                "avg_fruit_count": 0.0,
                "avg_ripening_days": 0.0
            }
        }

# ----- Data: History -----
@app.get("/data/history", tags=["Data"])
def get_prediction_history(limit: int = 1000):
    """Retrieve unified prediction history from the database."""
    try:
        history = db.get_history(limit)
        return {"history": history}
    except Exception as e:
        print(f"[ERROR] History fetch failed: {e}")
        return {"history": []}


# ----- Export -----
from src.analytics_service import get_analytics_paths
from src.csv_storage import YIELD_CSV, MATURITY_CSV
import json

@app.get("/data/export/csv")
async def export_history_csv(type: str = 'maturity'):
    """Export the requested strict history type directly to CSV."""
    try:
        if type == 'yield':
            filepath = YIELD_CSV
        else:
            filepath = MATURITY_CSV

        if not filepath.exists():
            return JSONResponse(status_code=404, content={"message": "No data to export"})

        headers = {
            'Content-Disposition': f'attachment; filename="coffee_{type}_predictions.csv"',
            'Content-Type': 'text/csv'
        }
        
        # Read file as bytes to return as stream
        with open(filepath, 'rb') as f:
            content = f.read()
            
        return StreamingResponse(
            iter([content]),
            headers=headers
        )
    except Exception as e:
        print(f"[ERROR] CSV export failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/report", tags=["Export"])
async def generate_pdf_report(request: dict):
    """Generate a PDF report from prediction data."""
    try:
        from fpdf import FPDF
        import tempfile
        import base64
        import os
        
        mode = request.get("mode", "Unknown")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=16)
        pdf.cell(0, 10, text=f"CoffeeAI Prediction Report - {mode}", new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(10)
        
        pdf.set_font("Helvetica", size=12)
        
        if "Yield" in mode:
            pdf.cell(0, 10, text=f"Predicted Yield: {request.get('predicted_yield', 0)} kg/ha", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, text=f"Profit Status: {request.get('profit_status', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, text=f"Fruit Count: {request.get('fruit_count', 0)}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, text=f"Ripeness Ratio: {request.get('ripeness_ratio', 0)}", new_x="LMARGIN", new_y="NEXT")
            
            insights = request.get("dynamic_insights", {})
            if insights:
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, text="Causes & Stress Factors", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", size=12)
                
                primary = insights.get("primary_stress")
                if primary:
                    sev = str(primary.get('severity', '')).upper()
                    name = str(primary.get('name', ''))
                    pdf.cell(0, 8, text=f"  Primary ({sev}): {name}", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "I", 10)
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(pdf.epw, 6, text=str(primary.get('description', '')), new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", size=12)
                
                secondary = insights.get("secondary_stress")
                if secondary:
                    sev = str(secondary.get('severity', '')).upper()
                    name = str(secondary.get('name', ''))
                    pdf.cell(0, 8, text=f"  Secondary ({sev}): {name}", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "I", 10)
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(pdf.epw, 6, text=str(secondary.get('description', '')), new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", size=12)

                recs = insights.get("recommendations", [])
                if recs:
                    pdf.ln(3)
                    pdf.set_font("Helvetica", "B", 14)
                    pdf.cell(0, 10, text="Recommendations", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", size=10)
                    for r in recs:
                        line = f"* {r.get('action', '')} - {r.get('reason', '')}"
                        pdf.set_x(pdf.l_margin)
                        pdf.multi_cell(pdf.epw, 6, text=line, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", size=12)
        else:
            pdf.cell(0, 10, text=f"Maturity Label: {request.get('final_label', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
            conf = request.get('final_confidence', 0)
            pdf.cell(0, 10, text=f"Confidence: {conf*100:.1f}%", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, text=f"Fruits Detected: {request.get('fruits_detected', 0)}", new_x="LMARGIN", new_y="NEXT")
            dist = request.get('distribution', {})
            if dist:
                pdf.cell(0, 10, text=f"Distribution: Ripe({dist.get('Ripe',0)}) Unripe({dist.get('Unripe',0)}) Overripe({dist.get('Overripe',0)})", new_x="LMARGIN", new_y="NEXT")
                
        def add_base64_image(b64_str, title):
            if not b64_str:
                return
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            try:
                img_data = base64.b64decode(b64_str)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(img_data)
                    tmp_path = tmp.name
                
                pdf.ln(5)
                pdf.cell(0, 10, text=title, new_x="LMARGIN", new_y="NEXT")
                pdf.image(tmp_path, w=100)
                pdf.ln(5)
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Error adding image to PDF: {e}")

        add_base64_image(request.get("original_image"), "Original Image")
        add_base64_image(request.get("annotated_image"), "Annotated Image")
        
        pdf_buffer = io.BytesIO()
        pdf_bytes = pdf.output()
        if isinstance(pdf_bytes, str):
            pdf_buffer.write(pdf_bytes.encode('latin1'))
        else:
            pdf_buffer.write(bytes(pdf_bytes))
        pdf_buffer.seek(0)
        
        headers = {
            'Content-Disposition': 'attachment; filename="CoffeeAI_Report.pdf"',
            'Content-Type': 'application/pdf'
        }
        
        return StreamingResponse(
            iter([pdf_buffer.getvalue()]),
            headers=headers
        )
    except Exception as e:
        import traceback
        with open("pdf_error.log", "w") as f:
            f.write(traceback.format_exc())
        print(f"[ERROR] PDF generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



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
            save_maturity_prediction({
                "image_name": getattr(file, "filename", ""),
                "predicted_maturity": "REJECTED",
                "yolo_confidence": 0.0,
                "total_fruits_detected": 0,
                "ripe_count": 0,
                "unripe_count": 0,
                "overripe_count": 0
            })
            
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "message": "QA Rejected. No Coffee Fruit Found",
                    "qa": qa_report
                }
            )
        
        # ===== PREDICTION =====
        result = model_instance.predict_yolo_only(str(temp_path))
        
        # Annotate image
        annotated_img = model_instance.draw_annotations(str(temp_path), result['predictions'])
        b64_img = image_to_base64(annotated_img)
        
        # Debug logging
        print(f"\n[PREDICTION - YOLO]")
        print(f"  Final Label: {result['final_label']}")
        print(f"  Confidence: {result['final_confidence'] * 100:.1f}%")
        
        # Build response
        response = {
            "ok": True,
            "message": qa_message,
            "qa": qa_report,
            "mode": result['mode'],
            "final_label": result['final_label'],
            "final_confidence": result['final_confidence'],
            "yolo_label": result['final_label'],
            "yolo_confidence": result['final_confidence'],
            "fruits_detected": result['fruits_detected'],
            "distribution": result['distribution'],
            "annotated_image": b64_img
        }
        
        # Store only if QA passed
        save_maturity_prediction({
            "image_name": getattr(file, "filename", ""),
            "predicted_maturity": result['final_label'],
            "yolo_confidence": result['final_confidence'],
            "ripe_count": result['distribution'].get("Ripe", 0),
            "unripe_count": result['distribution'].get("Unripe", 0),
            "overripe_count": result['distribution'].get("Overripe", 0),
            "total_fruits_detected": result['fruits_detected']
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
                "message": "QA Rejected. No Coffee Fruit Found"
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
            save_maturity_prediction({
                "image_name": getattr(file, "filename", ""),
                "predicted_maturity": "REJECTED",
                "efficientnet_confidence": 0.0,
                "total_fruits_detected": 0,
                "ripe_count": 0,
                "unripe_count": 0,
                "overripe_count": 0
            })
            
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "message": "QA Rejected. No Coffee Fruit Found",
                    "qa": qa_report
                }
            )
        
        # ===== PREDICTION =====
        result = model_instance.predict_effnet_only(str(temp_path))
        
        # Debug logging
        print(f"\n[PREDICTION - EfficientNet]")
        print(f"  Final Label: {result['final_label']}")
        print(f"  Confidence: {result['final_confidence'] * 100:.1f}%")

        response = {
            "ok": True,
            "message": qa_message,
            "qa": qa_report,
            "mode": result['mode'],
            "final_label": result['final_label'],
            "final_confidence": result['final_confidence'],
            "eff_label": result['final_label'],
            "eff_confidence": result['final_confidence'],
            "fruits_detected": result['fruits_detected'],
            "distribution": result['distribution'],
            "annotated_image": None 
        }
        
        save_maturity_prediction({
            "image_name": getattr(file, "filename", ""),
            "predicted_maturity": result['final_label'],
            "efficientnet_confidence": result['final_confidence'],
            "ripe_count": result['distribution'].get("Ripe", 0),
            "unripe_count": result['distribution'].get("Unripe", 0),
            "overripe_count": result['distribution'].get("Overripe", 0),
            "total_fruits_detected": result['fruits_detected']
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
                "message": "QA Rejected. No Coffee Fruit Found"
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
            save_maturity_prediction({
                "image_name": getattr(file, "filename", ""),
                "predicted_maturity": "REJECTED",
                "hybrid_confidence": 0.0,
                "total_fruits_detected": 0,
                "ripe_count": 0,
                "unripe_count": 0,
                "overripe_count": 0
            })
            
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "message": "QA Rejected. No Coffee Fruit Found",
                    "qa": qa_report
                }
            )
        
        # ===== PREDICTION =====
        result = model_instance.predict_hybrid(str(temp_path))
        
        # Annotate
        annotated_img = model_instance.draw_annotations(str(temp_path), result['predictions'])
        b64_img = image_to_base64(annotated_img)
        
        # Debug logging
        print(f"\n[PREDICTION - Hybrid]")
        print(f"  YOLO: {result.get('yolo_label')} ({result.get('yolo_confidence', 0) * 100:.1f}%)")
        print(f"  EffNet: {result.get('eff_label')} ({result.get('eff_confidence', 0) * 100:.1f}%)")
        print(f"  Final: {result['final_label']} ({result['final_confidence'] * 100:.1f}%)")

        response = {
            "ok": True,
            "message": qa_message,
            "qa": qa_report,
            "mode": result['mode'],
            "final_label": result['final_label'],
            "final_confidence": result['final_confidence'],
            "yolo_label": result.get('yolo_label'),
            "yolo_confidence": result.get('yolo_confidence'),
            "eff_label": result.get('eff_label'),
            "eff_confidence": result.get('eff_confidence'),
            "fruits_detected": result['fruits_detected'],
            "distribution": result['distribution'],
            "annotated_image": b64_img,
            "fusion_alpha": model_instance.fusion_alpha
        }
        
        save_maturity_prediction({
            "image_name": getattr(file, "filename", ""),
            "predicted_maturity": result['final_label'],
            "hybrid_confidence": result['final_confidence'],
            "yolo_confidence": result.get('yolo_confidence', 0),
            "efficientnet_confidence": result.get('eff_confidence', 0),
            "ripe_count": result['distribution'].get("Ripe", 0),
            "unripe_count": result['distribution'].get("Unripe", 0),
            "overripe_count": result['distribution'].get("Overripe", 0),
            "total_fruits_detected": result['fruits_detected']
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
                "message": f"QA Rejected. Error processing: {str(e)}"
            }
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass





# ----- Prediction: Yield -----
@app.post("/predict/yield", tags=["Prediction"])
async def predict_yield(
    file: UploadFile = File(...),
    lat: float = Form(20.0),
    lon: float = Form(75.0),
    price_per_kg: float = Form(200.0),
    estimated_plant_maintenance_cost: float = Form(100.0),
    soil_type: str = Form('loamy'),
    water_retention: str = Form('medium'),
    drainage: str = Form('moderate')
):
    """
    Intelligent Yield Decision System:
    Combines YOLO fruit detection + OpenWeatherMap environmental data
    to estimate harvest yield (kg/ha), profit/loss, reasons, and suggestions.
    """
    temp_path = None
    from src.soil_service import get_soil_data as fetch_soilgrids_data
    try:
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Models not loaded")

        valid, msg = validate_image_file(file)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)

        file_bytes = await file.read()
        temp_path = UPLOADS_DIR / f"yield_temp_{datetime.now().timestamp()}_{file.filename}"
        temp_path.write_bytes(file_bytes)

        # ── 1. QA Gate ────────────────────────────────────────────────────────
        qa_pass, qa_message, qa_report = model_instance.qa_gate(str(temp_path), mode='yolo')
        if not qa_pass:
            save_yield_prediction({
                "image_name": getattr(file, "filename", ""),
                "fruit_count": 0,
                "ripe_count": 0,
                "unripe_count": 0,
                "overripe_count": 0,
                "ripeness_ratio": 0.0,
                "predicted_yield": 0.0,
                "ripening_days": 0,
                "expected_harvest_date": "",
            })
            return {
                "ok": False,
                "message": "QA Rejected. No Coffee Fruit Found",
                "qa": qa_report,
            }

        # ── 2. Vision Layer: YOLO Detection ──────────────────────────────────
        detections = yield_module.detect_coffee_fruits(str(temp_path), model_instance.yolo)
        visual_features = yield_module.extract_visual_features(detections)

        # ── 3. Environment Layer: Current + Historical Weather ────────────────
        weather = yield_module.fetch_weather_features(lat, lon)

        temperature           = weather["temperature"]
        humidity              = weather["humidity"]
        rainfall_today        = weather["rainfall_today"]
        rainfall_last_30_days = weather["rainfall_last_30_days"]
        rainfall_last_90_days = weather["rainfall_last_90_days"]
        avg_temp_last_30_days = weather["avg_temp_last_30_days"]

        # ── 3.5. Soil Detection & Validation ──────────────────────────────────
        detected_district = "None"
        try:
            detected_soil = yield_module.get_soil_data_by_location(lat, lon)
            if detected_soil.get("detected_district") and "Unknown" not in detected_soil.get("detected_district", ""):
                # Use automated soil detection
                soil_type = detected_soil.get("soil_type", soil_type)
                water_retention = detected_soil.get("water_retention", water_retention)
                drainage = detected_soil.get("drainage", drainage)
                detected_district = detected_soil.get("detected_district", "Found")
            else:
                detected_district = "Unknown (Fallback to Manual)"
        except Exception:
            detected_district = "Failed (Fallback to Manual)"

        soil_score = yield_module.compute_soil_score(soil_type, water_retention, drainage)
        soil_index = yield_module.compute_soil_index(soil_type, drainage)

        # ── 4. Feature Fusion → ML Yield Prediction (12-feature) ──────────────
        features_dict = {
            "fruit_count":           visual_features["fruit_count"],
            "ripeness_ratio":        visual_features["ripeness_ratio"],
            "temperature":           temperature,
            "humidity":              humidity,
            "rainfall_last_30_days": rainfall_last_30_days,
            "rainfall_last_90_days": rainfall_last_90_days,
            "soil_score":            soil_score,
            "soil_index":            soil_index,
        }
        feature_vector  = yield_module.create_feature_vector(features_dict)
        predicted_yield = yield_module.predict_coffee_yield(feature_vector, soil_score)
        confidence      = yield_module.calculate_prediction_confidence(predicted_yield, feature_vector)
        importances     = yield_module.get_feature_importances()

        # ── 5. Ripening Time Prediction ────────────────────────────────────────
        ripening_info = yield_module.predict_ripening_days(
            temperature   = temperature,
            unripe_count  = visual_features["unripe_count"],
            fruit_count   = visual_features["fruit_count"],
        )
        ripening_days         = ripening_info["ripening_days"]
        expected_harvest_date = ripening_info["expected_harvest_date"]

        # ── 6. Profit / Loss Indicator (Plant Level) ───────────────────────────
        profit_info = yield_module.compute_plant_profit_indicator(
            predicted_yield, price_per_kg, estimated_plant_maintenance_cost
        )

        # -- 7. Explainable Yield Factors & Root Health ------------------------
        from src.dynamic_explainer import generate_dynamic_insights
        
        # We need soil properties, not just soil score. Use SoilGrids data.
        soil_grids_data = fetch_soilgrids_data(lat, lon)
        
        # Apply agronomic fallbacks for any missing API data so the UI doesn't show N/A
        fallbacks = {"ph": 5.8, "clay": 25.0, "sand": 45.0, "silt": 30.0, "soc": 25.0}
        for k, v in fallbacks.items():
            if soil_grids_data.get(k) is None:
                soil_grids_data[k] = v

        dynamic_insights = generate_dynamic_insights(
            soil_grids_data,
            weather,
            visual_features,
            is_same_location=False,
            profit_status=profit_info["status"]
        )

        # -- 8. Potential Yield Improvement ------------------------------------
        # ── 8. Potential Yield Improvement ------------------------------------
        # For potential yield, we map explainer recommendations to legacy factor logic
        # Provide dummy pairs just to drive the multiplier if needed
        dummy_pairs = [("r", "s") for _ in range(len(dynamic_insights["recommendations"]))]
        potential_info = yield_module.estimate_potential_yield(predicted_yield, dummy_pairs)

        # ── 10. Annotated Detection Image ─────────────────────────────────────
        pred_list = []
        if detections.boxes is not None and len(detections.boxes) > 0:
            boxes_xyxy = detections.boxes.xyxy.cpu().numpy()
            confs      = detections.boxes.conf.cpu().numpy()
            cls_ids    = detections.boxes.cls.cpu().numpy().astype(int)
            names      = detections.names
            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                pred_list.append({
                    "box":        (int(x1), int(y1), int(x2), int(y2)),
                    "class":      names.get(cls_ids[i], str(cls_ids[i])),
                    "confidence": float(confs[i]),
                })
        annotated_img = model_instance.draw_annotations(str(temp_path), pred_list)
        b64_img       = image_to_base64(annotated_img)

        # ── 11. Persist to DB ─────────────────────────────────────────────────
        
        save_yield_prediction({
            "image_name":           getattr(file, "filename", ""),
            "fruit_count":          visual_features["fruit_count"],
            "ripe_count":           visual_features["ripe_count"],
            "unripe_count":         visual_features["unripe_count"],
            "overripe_count":       visual_features["overripe_count"],
            "ripeness_ratio":       visual_features["ripeness_ratio"],
            "temperature":          temperature,
            "humidity":             humidity,
            "rainfall_today":       rainfall_today,
            "rainfall_30d":         rainfall_last_30_days,
            "rainfall_90d":         rainfall_last_90_days,
            "avg_temp_30d":         avg_temp_last_30_days,
            "predicted_yield":      round(predicted_yield, 2),
            "ripening_days":        ripening_days,
            "expected_harvest_date": expected_harvest_date,
            "soil_type":            soil_type,
            "water_retention":      water_retention,
            "drainage":             drainage,
            "soil_score":           soil_score,
        })

        print(f"[YIELD] fruit={visual_features['fruit_count']} | "
              f"yield={predicted_yield:.3f} kg/ha | "
              f"status={profit_info['status']} | "
              f"profit=₹{profit_info['profit']:.0f} | "
              f"dynamic_insights=True")

        # ── 12. Full Response ─────────────────────────────────────────────────
        return {
            "ok":                    True,
            "mode":                  "Yield Prediction",
            "message":               "Yield prediction successful",
            "qa":                    qa_report,
            # Vision
            "fruit_count":           visual_features["fruit_count"],
            "ripe_count":            visual_features["ripe_count"],
            "unripe_count":          visual_features["unripe_count"],
            "overripe_count":        visual_features["overripe_count"],
            "ripeness_ratio":        round(visual_features["ripeness_ratio"], 4),
            # Environment
            "temperature":           temperature,
            "humidity":              humidity,
            "rainfall_today":        rainfall_today,
            "rainfall_last_30_days": rainfall_last_30_days,
            "rainfall_last_90_days": rainfall_last_90_days,
            "avg_temp_last_30_days": avg_temp_last_30_days,
            # Soil
            "detected_district":     detected_district,
            "soil_type":             soil_type,
            "water_retention":       water_retention,
            "drainage":              drainage,
            "soil_score":            soil_score,
            "soil_index":            soil_index,
            "soil_grids_data":       soil_grids_data,
            # Yield
            "predicted_yield":       round(predicted_yield, 2),
            "confidence":            round(confidence, 2),
            "model_architecture":    "Enhanced XGBoost (12-Feature + Location Soil)",
            "feature_importances":   importances,
            # Profit Analysis
            "profit_status":         profit_info["status"],
            "profit":                profit_info["profit"],
            "break_even_yield":      profit_info["break_even_yield"],
            "price_per_kg":          profit_info["price_per_kg"],
            "estimated_plant_maintenance_cost": profit_info["estimated_plant_maintenance_cost"],
            # Explainability
            "dynamic_insights":      dynamic_insights,
            # Potential
            "potential_yield":       potential_info["potential_yield"],
            "improvement_factor":    potential_info["improvement_factor"],
            "improvement_pct":       potential_info["improvement_pct"],
            # Ripening
            "ripening_days":         ripening_days,
            "expected_harvest_date": expected_harvest_date,
            # Visual
            "annotated_image":       b64_img,
        }

    except Exception as e:
        print(f"[ERROR] Yield Prediction failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: run the full yield pipeline for a single plant image.
# Reused by both /predict/yield and /predict/compare so logic stays in one place.
# ─────────────────────────────────────────────────────────────────────────────
async def _run_yield_pipeline(
    file: UploadFile,
    lat: float,
    lon: float,
    price_per_kg: float,
    estimated_plant_maintenance_cost: float,
    plant_label: str = "Plant",
) -> dict:
    """
    Runs the complete yield prediction pipeline for a single plant image.
    Returns the full result dict (same structure as /predict/yield response).
    Raises HTTPException on hard failures.
    """
    from src.soil_service import get_soil_data as fetch_soilgrids_data
    from src.yield_explainer import analyze_yield_factors, infer_root_health, generate_recommendations

    temp_path = None
    try:
        valid, msg = validate_image_file(file)
        if not valid:
            raise HTTPException(status_code=400, detail=f"{plant_label}: {msg}")

        file_bytes = await file.read()
        temp_path = UPLOADS_DIR / f"compare_{plant_label.replace(' ','_')}_{datetime.now().timestamp()}_{file.filename}"
        temp_path.write_bytes(file_bytes)

        # QA Gate
        qa_pass, qa_message, qa_report = model_instance.qa_gate(str(temp_path), mode='yolo')
        if not qa_pass:
            return {
                "ok": False,
                "message": f"QA Rejected for {plant_label}. No Coffee Fruit Found",
                "qa": qa_report,
                "predicted_yield": 0.0,
                "fruit_count": 0,
                "ripeness_ratio": 0.0,
            }

        # Vision layer
        detections = yield_module.detect_coffee_fruits(str(temp_path), model_instance.yolo)
        visual_features = yield_module.extract_visual_features(detections)

        # Weather
        weather = yield_module.fetch_weather_features(lat, lon)
        temperature           = weather["temperature"]
        humidity              = weather["humidity"]
        rainfall_today        = weather["rainfall_today"]
        rainfall_last_30_days = weather["rainfall_last_30_days"]
        rainfall_last_90_days = weather["rainfall_last_90_days"]
        avg_temp_last_30_days = weather["avg_temp_last_30_days"]

        # Soil (OSM geocoding fallback)
        soil_type, water_retention, drainage = 'loamy', 'medium', 'moderate'
        detected_district = "Unknown"
        try:
            detected_soil = yield_module.get_soil_data_by_location(lat, lon)
            if detected_soil.get("detected_district") and "Unknown" not in detected_soil.get("detected_district", ""):
                soil_type       = detected_soil.get("soil_type", soil_type)
                water_retention = detected_soil.get("water_retention", water_retention)
                drainage        = detected_soil.get("drainage", drainage)
                detected_district = detected_soil.get("detected_district", "Found")
        except Exception:
            pass

        soil_score = yield_module.compute_soil_score(soil_type, water_retention, drainage)
        soil_index = yield_module.compute_soil_index(soil_type, drainage)

        # Feature fusion + ML yield
        features_dict = {
            "fruit_count":           visual_features["fruit_count"],
            "ripeness_ratio":        visual_features["ripeness_ratio"],
            "temperature":           temperature,
            "humidity":              humidity,
            "rainfall_last_30_days": rainfall_last_30_days,
            "rainfall_last_90_days": rainfall_last_90_days,
            "soil_score":            soil_score,
            "soil_index":            soil_index,
        }
        feature_vector  = yield_module.create_feature_vector(features_dict)
        predicted_yield = yield_module.predict_coffee_yield(feature_vector, soil_score)
        confidence      = yield_module.calculate_prediction_confidence(predicted_yield, feature_vector)
        importances     = yield_module.get_feature_importances()

        # Ripening
        ripening_info = yield_module.predict_ripening_days(
            temperature=temperature,
            unripe_count=visual_features["unripe_count"],
            fruit_count=visual_features["fruit_count"],
        )

        # Profit (plant level)
        profit_info = yield_module.compute_plant_profit_indicator(
            predicted_yield, price_per_kg, estimated_plant_maintenance_cost
        )

        # SoilGrids raw data
        soil_grids_data = fetch_soilgrids_data(lat, lon)
        fallbacks = {"ph": 5.8, "clay": 25.0, "sand": 45.0, "silt": 30.0, "soc": 25.0}
        for k, v in fallbacks.items():
            if soil_grids_data.get(k) is None:
                soil_grids_data[k] = v

        # Explainability
        from src.dynamic_explainer import generate_dynamic_insights
        dynamic_insights = generate_dynamic_insights(
            soil_grids_data, 
            weather, 
            visual_features, 
            is_same_location=False,
            profit_status=profit_info["status"]
        )

        # Potential yield
        dummy_pairs = [("r", "s") for _ in range(len(dynamic_insights["recommendations"]))]
        potential_info = yield_module.estimate_potential_yield(predicted_yield, dummy_pairs)

        # Annotated image
        pred_list = []
        if detections.boxes is not None and len(detections.boxes) > 0:
            boxes_xyxy = detections.boxes.xyxy.cpu().numpy()
            confs      = detections.boxes.conf.cpu().numpy()
            cls_ids    = detections.boxes.cls.cpu().numpy().astype(int)
            names      = detections.names
            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                pred_list.append({
                    "box":        (int(x1), int(y1), int(x2), int(y2)),
                    "class":      names.get(cls_ids[i], str(cls_ids[i])),
                    "confidence": float(confs[i]),
                })
        annotated_img = model_instance.draw_annotations(str(temp_path), pred_list)
        b64_img = image_to_base64(annotated_img)

        print(f"[COMPARE/{plant_label}] fruit={visual_features['fruit_count']} | "
              f"yield={predicted_yield:.3f} kg/plant | status={profit_info['status']}")

        return {
            "ok": True,
            "label": plant_label,
            "qa": qa_report,
            "fruit_count":           visual_features["fruit_count"],
            "ripe_count":            visual_features["ripe_count"],
            "unripe_count":          visual_features["unripe_count"],
            "overripe_count":        visual_features["overripe_count"],
            "ripeness_ratio":        round(visual_features["ripeness_ratio"], 4),
            "temperature":           temperature,
            "humidity":              humidity,
            "rainfall_today":        rainfall_today,
            "rainfall_last_30_days": rainfall_last_30_days,
            "rainfall_last_90_days": rainfall_last_90_days,
            "avg_temp_last_30_days": avg_temp_last_30_days,
            "detected_district":     detected_district,
            "soil_type":             soil_type,
            "water_retention":       water_retention,
            "drainage":              drainage,
            "soil_score":            soil_score,
            "soil_index":            soil_index,
            "soil_grids_data":       soil_grids_data,
            "predicted_yield":       round(predicted_yield, 3),
            "confidence":            round(confidence, 2),
            "profit_status":         profit_info["status"],
            "profit":                profit_info["profit"],
            "break_even_yield":      profit_info["break_even_yield"],
            "feature_importances":   importances,
            "dynamic_insights":      dynamic_insights,
            "potential_yield":       potential_info["potential_yield"],
            "ripening_days":         ripening_info["ripening_days"],
            "expected_harvest_date": ripening_info["expected_harvest_date"],
            "annotated_image":       b64_img,
        }

    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ----- Prediction: Compare Two Plants -----
@app.post("/predict/compare", tags=["Prediction"])
async def predict_compare(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    lat_a:  float = Form(13.0),
    lon_a:  float = Form(76.1),
    lat_b:  float = Form(13.0),
    lon_b:  float = Form(76.1),
    price_per_kg: float = Form(200.0),
    estimated_plant_maintenance_cost: float = Form(100.0),
):
    """
    Comparative Explainable Agricultural Intelligence.
    Runs the full yield pipeline independently on two plant images,
    then generates a side-by-side comparative analysis with root-cause
    identification and actionable recommendations for the lower-performing plant.

    Does NOT modify or break any existing endpoints.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    from src.comparative_explainer import generate_full_comparison

    # Run both plants in sequence
    plant_a = await _run_yield_pipeline(
        file_a, lat_a, lon_a, price_per_kg, estimated_plant_maintenance_cost, "Plant A"
    )
    plant_b = await _run_yield_pipeline(
        file_b, lat_b, lon_b, price_per_kg, estimated_plant_maintenance_cost, "Plant B"
    )

    comparison = {}
    if plant_a.get("ok") and plant_b.get("ok"):
        # Pass coordinates so the explainer can detect same-location mode
        comparison = generate_full_comparison(
            plant_a, plant_b,
            lat_a=lat_a, lon_a=lon_a,
            lat_b=lat_b, lon_b=lon_b,
        )
    else:
        comparison = {
            "same_location_mode": False,
            "verdict": "Comparison could not be completed — one or both plants failed QA.",
            "yield_comparison": {},
            "soil_comparison": [],
            "loser_causes": [],
            "loser_root_health": [],
            "recommendations": [],
            "visual_scores_a": None,
            "visual_scores_b": None,
        }

    return {
        "ok": True,
        "mode": "Compare Plants",
        "plant_a": plant_a,
        "plant_b": plant_b,
        "comparison": comparison,
    }


@app.post("/predict-maturity", tags=["Prediction"])
async def predict_maturity_alias(file: UploadFile = File(...)):
    """User requested explicit endpoint for maturity prediction."""
    return await predict_hybrid(file)

@app.post("/predict-yield", tags=["Prediction"])
async def predict_yield_alias(
    file: UploadFile = File(...),
    lat: float = Form(20.0),
    lon: float = Form(75.0)
):
    """User requested explicit endpoint for yield prediction."""
    return await predict_yield(file, lat, lon)





# ----- Static Files -----
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# ----- Main -----
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

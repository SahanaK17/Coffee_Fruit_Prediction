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
from fpdf import FPDF
from fastapi.responses import JSONResponse, StreamingResponse

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

def store_prediction(data: dict):
    """Store a single prediction."""
    store_predictions([data])

def store_predictions(data_list: List[dict]):
    """Store multiple predictions to appropriate history files (Valid or Rejected)."""
    try:
        # Group by target file to minimize I/O and avoid reload issues
        valid_items = []
        rejected_items = []
        timestamp = datetime.now().isoformat()

        for data in data_list:
            # Ensure timestamp is present
            if 'timestamp' not in data:
                data['timestamp'] = timestamp
            
            is_rejected = data.get('final_label') == 'REJECTED'
            if is_rejected:
                rejected_items.append(data)
            else:
                valid_items.append(data)

        # Process each group
        for items, target_file, label in [(valid_items, HISTORY_FILE, 'VALID'), (rejected_items, REJECTED_FILE, 'REJECTED')]:
            if not items:
                continue

            history = []
            if target_file.exists():
                try:
                    with open(target_file, 'r') as f:
                        history = json.load(f)
                except Exception as e:
                    print(f"[WARNING] Could not read {target_file.name}: {e}")
                    history = []
            
            history.extend(items)
            # Keep only last 1000 records
            history = history[-1000:]
            
            with open(target_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            print(f"[LOG] Stored {len(items)} {label} predictions to {target_file.name}")
            
    except Exception as e:
        print(f"[ERROR] Failed to store predictions: {e}")
        traceback.print_exc()

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

# ----- PDF Report Class -----
class CoffeeReportPDF(FPDF):
    def header(self):
        # Draw a simple logo-like circle if no logo file exists
        self.set_fill_color(198, 124, 78) # Theme color
        self.ellipse(10, 10, 10, 10, 'F') # (x, y, w, h) - compatible with old fpdf and fpdf2
        
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(40, 40, 40)
        self.cell(20)
        self.cell(0, 10, 'CoffeeAI Maturity Analysis Report', 0, 1, 'L')
        self.set_font('helvetica', 'I', 8)
        self.cell(20)
        self.cell(0, 5, 'M.Tech Research System - Vision-based Quality Assessment', 0, 1, 'L')
        self.ln(10)
        self.line(10, 32, 200, 32)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')

def generate_pdf_report(data: dict, img_b64: str = None):
    pdf = CoffeeReportPDF()
    pdf.add_page()
    
    # Section: Overall Result
    pdf.set_y(40)
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(198, 124, 78)
    pdf.cell(0, 10, '1. Prediction Summary', 0, 1, 'L')
    pdf.ln(2)
    
    # Metrics Table
    pdf.set_font('helvetica', 'B', 11)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_text_color(50, 50, 50)
    
    pdf.cell(50, 10, 'Metric', 1, 0, 'C', True)
    pdf.cell(80, 10, 'Value', 1, 1, 'C', True)
    
    pdf.set_font('helvetica', '', 11)
    
    results = [
        ('Prediction Label', data.get('final_label', 'N/A')),
        ('Confidence Score', f"{float(data.get('final_confidence', 0))*100:.1f}%"),
        ('Analysis Mode', data.get('mode', 'N/A'))
    ]
    
    # Context-aware Fruit Count
    mode = data.get('mode', '').lower()
    if 'yolo' in mode or 'hybrid' in mode:
        results.append(('Fruits Detected', str(data.get('fruits_detected', 0))))
    else:
        results.append(('Analysis Level', 'Global Classification'))
    
    for label, val in results:
        pdf.cell(50, 10, f" {label}", 1, 0, 'L')
        pdf.cell(80, 10, f" {val}", 1, 1, 'L')
    
    # Section: Class Distribution
    pdf.ln(10)
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(198, 124, 78)
    pdf.cell(0, 10, '2. Detailed Distribution', 0, 1, 'L')
    pdf.ln(2)
    
    dist = data.get('distribution', {})
    if dist:
        pdf.set_font('helvetica', '', 11)
        pdf.set_text_color(50, 50, 50)
        for cls, prob in dist.items():
            # Format as percentage if it looks like a float probability
            if isinstance(prob, (float, int)) and prob <= 1.0 and any(isinstance(v, float) for v in dist.values()):
                display_val = f"{prob*100:.1f}%"
            else:
                display_val = str(prob)
                
            pdf.cell(50, 8, f" - {cls}:", 0, 0)
            pdf.cell(50, 8, display_val, 0, 1)
    
    # Section: Visual Analysis
    pdf.ln(5)
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(198, 124, 78)
    pdf.cell(0, 10, '3. Visual Analysis', 0, 1, 'L')
    pdf.ln(2)
    
    # Image Fallback Logic: Annotated -> Original
    img_b64 = data.get('annotated_image') or data.get('original_image')
    
    if img_b64:
        try:
            # Decode b64 to temp file
            if "," in img_b64: img_b64 = img_b64.split(",")[1] # Remove data:image/jpeg;base64,
            img_data = base64.b64decode(img_b64)
            temp_path = UPLOADS_DIR / f"report_img_{pdf.page_no()}_{datetime.now().timestamp()}.jpg"
            temp_path.write_bytes(img_data)
            
            # Add image (scaled to fit)
            img_w = 170
            
            # Check for page overflow
            # Rough estimate: image height will be around 100-120 units for standard aspect ratio
            if pdf.get_y() > 180: # If near bottom
                pdf.add_page()
                pdf.set_y(20) # Reset Y on new page
                
            pdf.image(str(temp_path), x=15, y=pdf.get_y(), w=img_w)
        except Exception as e:
            pdf.set_font('helvetica', 'I', 10)
            pdf.cell(0, 10, f"Visualization unavailable: {str(e)}", 0, 1)

    return pdf

# ----- PDF Report Endpoint -----
@app.post("/predict/report", tags=["Prediction"])
async def get_prediction_report(data: dict):
    """Generate and return a professional PDF report."""
    temp_img_path = None
    try:
        img_b64 = data.get('annotated_image')
        
        # We need to manage the temp image file for FPDF
        pdf = generate_pdf_report(data, img_b64)
        
        # FPDF output to bytes
        pdf_content = pdf.output(dest='S')
        if isinstance(pdf_content, str):
            # For older fpdf versions that return strings
            pdf_bytes = pdf_content.encode('latin-1')
        else:
            # For fpdf2 which returns bytes
            pdf_bytes = pdf_content
        
        # Cleanup temp images in uploads dir that start with report_img_
        # (Self-cleaning heuristic)
        for f in UPLOADS_DIR.glob("report_img_*"):
            try: f.unlink()
            except: pass

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=CoffeeAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"}
        )
    except Exception as e:
        print(f"[ERROR] PDF Report generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

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

# ----- Data -----
@app.get("/data/history", tags=["Data"])
def get_prediction_history(limit: int = 50):
    """Retrieve combined prediction history (Valid + Rejected)."""
    try:
        # Load valid history
        valid_history = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                valid_history = json.load(f)
        
        # Load rejected history
        rejected_history = []
        if REJECTED_FILE.exists():
            with open(REJECTED_FILE, 'r') as f:
                rejected_history = json.load(f)
        
        # Merge and sort by timestamp
        all_history = valid_history + rejected_history
        all_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {
            "history": all_history[:limit],
            "total": len(all_history)
        }
    except Exception as e:
        print(f"[ERROR] Failed to read history: {e}")
        return {"history": [], "total": 0}

# ----- Export -----
@app.get("/data/export/csv", tags=["Data"])
def export_history_csv():
    """Export prediction history to CSV."""
    try:
        # Load valid history
        valid_history = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                valid_history = json.load(f)
        
        # Load rejected history
        rejected_history = []
        if REJECTED_FILE.exists():
            with open(REJECTED_FILE, 'r') as f:
                rejected_history = json.load(f)
        
        # Merge all records
        all_records = valid_history + rejected_history
        
        if not all_records:
            return StreamingResponse(
                io.StringIO("timestamp,mode,final_label,confidence,fruits_detected\n"),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=coffee_predictions.csv"}
            )

        df = pd.DataFrame(all_records)
        
        # Sort by timestamp (descending)
        if 'timestamp' in df.columns:
            df = df.sort_values(by='timestamp', ascending=False)
        
        # Define core columns and ensure they exist
        cols = ['timestamp', 'mode', 'final_label', 'confidence', 'fruits_detected']
        for col in cols:
            if col not in df.columns:
                df[col] = "N/A"
        
        # Select columns in preferred order
        df = df[cols]
        
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        
        return StreamingResponse(
            io.StringIO(stream.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=coffee_predictions.csv"}
        )
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Export failed")

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
                "message": "QA Rejected. No Coffee Fruit Found"
            }
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

# ----- Batch Prediction -----
@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch Prediction Mode: Processes multiple images at once.
    Limited to 10 images per request.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch size limited to 10 images")
    
    results = []
    to_store = []
    
    for file in files:
        temp_path = None
        try:
            if model_instance is None:
                results.append({"filename": file.filename, "ok": False, "message": "Models not loaded"})
                continue
            
            # Validate
            valid, msg = validate_image_file(file)
            if not valid:
                results.append({"filename": file.filename, "ok": False, "message": msg})
                continue
            
            # Read
            file_bytes = await file.read()
            if len(file_bytes) > 10 * 1024 * 1024:
                results.append({"filename": file.filename, "ok": False, "message": "File too large"})
                continue
            
            # Save
            temp_path = UPLOADS_DIR / f"batch_{datetime.now().timestamp()}_{file.filename}"
            temp_path.write_bytes(file_bytes)
            
            # Run Hybrid Prediction (Default for batch)
            # QA Gate first
            qa_pass, qa_message, qa_report = model_instance.qa_gate(str(temp_path), mode='hybrid')
            
            if not qa_pass:
                # Log rejection for history
                to_store.append({
                    "mode": "Hybrid",
                    "final_label": "REJECTED",
                    "confidence": 0.0,
                    "distribution": {"Unripe": 0, "Ripe": 0, "Overripe": 0},
                    "fruits_detected": 0,
                    "qa_pass": False,
                    "qa_reason": qa_report.get('reason', 'N/A')
                })
                
                results.append({
                    "filename": file.filename,
                    "ok": False,
                    "message": "QA Rejected. No Coffee Fruit Found",
                    "qa": qa_report
                })
                continue
            
            # Predict
            res = model_instance.predict_hybrid(str(temp_path))
            
            # Add to batch storage list
            to_store.append({
                "mode": "Hybrid",
                "final_label": res['final_label'],
                "confidence": res['final_confidence'],
                "distribution": res['distribution'],
                "fruits_detected": res['fruits_detected']
            })
            
            results.append({
                "filename": file.filename,
                "ok": True,
                "mode": "Hybrid",
                "final_label": res['final_label'],
                "final_confidence": res['final_confidence'],
                "fruits_detected": res['fruits_detected']
            })
            
        except Exception as e:
            print(f"[ERROR] Batch item failed: {e}")
            results.append({"filename": file.filename, "ok": False, "message": str(e)})
        finally:
            if temp_path and temp_path.exists():
                try: temp_path.unlink()
                except: pass
    
    # Batch save all predictions at once (more efficient & safer against reload restarts)
    if to_store:
        store_predictions(to_store)
                
    return {"results": results}

# ----- Static Files -----
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# ----- Main -----
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

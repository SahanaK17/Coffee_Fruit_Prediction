from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
from src.inference import HybridModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = r"c:\Users\hp\OneDrive\projects\Coffee-prediction"
MODEL_YOLO = os.path.join(BASE_DIR, "models", "yolo", "best.pt")
MODEL_EFFNET = os.path.join(BASE_DIR, "models", "effnet", "effnet", "best.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Model (Lazy load or at startup)
model = None
try:
    if os.path.exists(MODEL_YOLO) and os.path.exists(MODEL_EFFNET):
        model = HybridModel(MODEL_YOLO, MODEL_EFFNET)
        print("Models loaded successfully.")
    else:
        print("Models not found. Training might be required.")
except Exception as e:
    print(f"Error loading models: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}
        
    predictions = model.predict(file_path)
    return {"predictions": predictions}

@app.get("/metrics")
async def get_metrics():
    # Return paths or data for training curves and confusion matrix
    return {
        "curves": "/static/models/effnet/training_curves.png",
        "confusion_matrix": "/static/models/effnet/confusion_matrix.png",
        "report": "/static/models/effnet/classification_report.txt"
    }

# Serve static files for frontend and model metrics
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

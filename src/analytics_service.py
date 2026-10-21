import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def get_analytics_paths():
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

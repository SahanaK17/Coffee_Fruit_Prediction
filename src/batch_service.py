import os
from typing import List, Dict, Any
from datetime import datetime
from fastapi import UploadFile
import traceback

from src import yield_module

async def process_batch(
    files: List[UploadFile],
    mode: str,
    lat: float,
    lon: float,
    model_instance,
    validate_func,
    save_maturity_func,
    save_yield_func,
    uploads_dir
) -> Dict[str, Any]:
    """
    Process a batch of uploaded images for either Maturity or Yield Prediction.
    """
    results = []
    to_store = []
    
    for file in files:
        temp_path = None
        try:
            if model_instance is None:
                results.append({"filename": file.filename, "ok": False, "message": "Models not loaded"})
                continue
            
            # Validate
            valid, msg = validate_func(file)
            if not valid:
                results.append({"filename": file.filename, "ok": False, "message": msg})
                continue
            
            # Read
            file_bytes = await file.read()
            if len(file_bytes) > 10 * 1024 * 1024:
                results.append({"filename": file.filename, "ok": False, "message": "File too large"})
                continue
            
            # Save
            temp_path = uploads_dir / f"batch_{datetime.now().timestamp()}_{file.filename}"
            temp_path.write_bytes(file_bytes)
            
            # QA Gate
            qa_pass, qa_message, qa_report = model_instance.qa_gate(str(temp_path), mode='yolo' if mode == 'yield' else 'hybrid')
            
            if not qa_pass:
                if mode == "yield":
                    save_yield_func({
                        "image_name": file.filename,
                        "predicted_yield_kg_per_hectare": 0.0,
                        "fruit_count": 0,
                        "ripeness_ratio": 0.0
                    })
                else:
                    save_maturity_func({
                        "image_name": file.filename,
                        "predicted_maturity": "REJECTED",
                        "hybrid_confidence": 0.0,
                        "total_fruits_detected": 0,
                        "ripe_count": 0,
                        "unripe_count": 0,
                        "overripe_count": 0
                    })
                
                results.append({
                    "filename": file.filename,
                    "ok": False,
                    "message": "QA Rejected. No Coffee Fruit Found",
                    "qa": qa_report
                })
                continue
            
            if mode == 'yield':
                # yield logic
                detections = yield_module.detect_coffee_fruits(str(temp_path), model_instance.yolo)
                visual_features = yield_module.extract_visual_features(detections)
                weather_features = yield_module.fetch_weather_features(lat, lon)
                
                features_dict = {
                    'fruit_count': visual_features['fruit_count'],
                    'ripeness_ratio': visual_features['ripeness_ratio'],
                    'temperature': weather_features['temperature'],
                    'humidity': weather_features['humidity'],
                    'rainfall_last_30_days': weather_features['rainfall_last_30_days'],
                    'rainfall_last_90_days': weather_features['rainfall_last_90_days'],
                }
                feature_vector = yield_module.create_feature_vector(features_dict)
                predicted_yield = yield_module.predict_coffee_yield(feature_vector)
                confidence = yield_module.calculate_prediction_confidence(predicted_yield, feature_vector)

                # Ripening prediction
                ripening_info = yield_module.predict_ripening_days(
                    temperature=weather_features['temperature'],
                    unripe_count=visual_features['unripe'],
                    fruit_count=visual_features['fruit_count'],
                )

                res_dict = {
                    "filename": file.filename,
                    "ok": True,
                    "mode": "Yield",
                    "final_label": f"{predicted_yield:.2f} kg/ha",
                    "final_confidence": confidence,
                    "fruit_count": visual_features['fruit_count'],
                    "ripeness_ratio": round(visual_features['ripeness_ratio'], 4),
                    "temperature": weather_features['temperature'],
                    "humidity": weather_features['humidity'],
                    "rainfall_today": weather_features['rainfall_today'],
                    "rainfall_last_30_days": weather_features['rainfall_last_30_days'],
                    "rainfall_last_90_days": weather_features['rainfall_last_90_days'],
                    "predicted_yield": round(predicted_yield, 2),
                    "distribution": {
                        "Ripe": visual_features['ripe'],
                        "Unripe": visual_features['unripe'],
                        "Overripe": visual_features['overripe']
                    }
                }
                save_yield_func({
                    "image_name": file.filename,
                    "fruit_count": visual_features['fruit_count'],
                    "ripeness_ratio": visual_features['ripeness_ratio'],
                    "temperature": weather_features['temperature'],
                    "humidity": weather_features['humidity'],
                    "rainfall_today": weather_features['rainfall_today'],
                    "rainfall_30d": weather_features['rainfall_last_30_days'],
                    "rainfall_90d": weather_features['rainfall_last_90_days'],
                    "predicted_yield": round(predicted_yield, 2),
                    "ripening_days": ripening_info['ripening_days'],
                    "expected_harvest_date": ripening_info['expected_harvest_date'],
                })
                results.append(res_dict)
            else:
                # Predict Maturity
                res = model_instance.predict_hybrid(str(temp_path))
                
                save_maturity_func({
                    "image_name": file.filename,
                    "predicted_maturity": res['final_label'],
                    "hybrid_confidence": res['final_confidence'],
                    "total_fruits_detected": res['fruits_detected'],
                    "ripe_count": res['distribution'].get("Ripe", 0),
                    "unripe_count": res['distribution'].get("Unripe", 0),
                    "overripe_count": res['distribution'].get("Overripe", 0)
                })
                
                results.append({
                    "filename": file.filename,
                    "ok": True,
                    "mode": "Hybrid",
                    "final_label": res['final_label'],
                    "final_confidence": res['final_confidence'],
                    "fruits_detected": res['fruits_detected'],
                    "distribution": res['distribution']
                })
                
        except Exception as e:
            print(f"[ERROR] Batch item failed: {e}")
            results.append({"filename": file.filename, "ok": False, "message": str(e)})
        finally:
            if temp_path and temp_path.exists():
                try: temp_path.unlink()
                except: pass

    return {"results": results}

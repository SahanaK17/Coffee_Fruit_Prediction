from ultralytics import YOLO
import os

def train_yolo(data_yaml, output_dir, epochs=50, imgsz=640):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO("yolo11n.pt") # YOLOv11 small
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=output_dir,
        name="coffee_yolo",
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    # Evaluate
    metrics = model.val(data=data_yaml, split='test')
    print("Test Metrics:", metrics)
    
    # Export for later use
    model.export(format="onnx")

if __name__ == "__main__":
    data_yaml = r"c:\Users\hp\OneDrive\projects\Coffee-prediction\Coffee Fruit Maturity ---.v1i.yolov8\data.yaml"
    output_dir = r"c:\Users\hp\OneDrive\projects\Coffee-prediction\models\yolo"
    train_yolo(data_yaml, output_dir, epochs=50)

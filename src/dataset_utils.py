import os
import cv2
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def create_effnet_dataset(yolo_dataset_path, output_path):
    """
    Creates a classification dataset for EfficientNet by cropping bounding boxes from YOLO dataset.
    """
    data_yaml = load_yaml(os.path.join(yolo_dataset_path, 'data.yaml'))
    classes = data_yaml['names']
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_path = os.path.join(yolo_dataset_path, split)
        img_dir = os.path.join(split_path, 'images')
        lbl_dir = os.path.join(split_path, 'labels')
        
        output_split_path = os.path.join(output_path, split)
        os.makedirs(output_split_path, exist_ok=True)
        
        for cls_name in classes:
            os.makedirs(os.path.join(output_split_path, cls_name), exist_ok=True)
            
        if not os.path.exists(img_dir):
            continue
            
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {split} split...")
        for img_file in tqdm(img_files):
            img_path = os.path.join(img_dir, img_file)
            lbl_file = os.path.splitext(img_file)[0] + '.txt'
            lbl_path = os.path.join(lbl_dir, lbl_file)
            
            if not os.path.exists(lbl_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls_idx = int(parts[0])
                cls_name = classes[cls_idx]
                
                # YOLO format: cls x_center y_center width height (normalized)
                x_center, y_center, bw, bh = map(float, parts[1:5])
                
                x1 = int((x_center - bw/2) * w)
                y1 = int((y_center - bh/2) * h)
                x2 = int((x_center + bw/2) * w)
                y2 = int((y_center + bh/2) * h)
                
                # Boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                crop_name = f"{os.path.splitext(img_file)[0]}_crop_{i}.jpg"
                cv2.imwrite(os.path.join(output_split_path, cls_name, crop_name), crop)

if __name__ == "__main__":
    yolo_path = r"c:\Users\hp\OneDrive\projects\Coffee-prediction\Coffee Fruit Maturity ---.v1i.yolov8"
    output_path = r"c:\Users\hp\OneDrive\projects\Coffee-prediction\classification_dataset"
    create_effnet_dataset(yolo_path, output_path)

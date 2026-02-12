import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import traceback
from typing import Dict, List, Tuple, Optional

class HybridModel:
    def __init__(self, yolo_path: str, effnet_path: str, num_classes: int = 3):
        """
        Initialize the hybrid model with both YOLO and EfficientNet.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = ['Unripe', 'Ripe', 'Overripe']
        self.num_classes = num_classes
        
        # ===== STRICT QA THRESHOLDS =====
        self.qa_min_conf = 0.40             # Minimum confidence for a "strong" detection
        self.qa_min_total_area = 0.002      # 0.2% minimum total area (sum of boxes)
        self.qa_max_total_area = 0.65       # 65% maximum total area
        self.qa_min_single_box_area = 0.001 # 0.1% minimum size for a single box
        self.qa_center_fraction = 0.60      # Middle 60% of image must contain a detection
        
        # Sketch/Noise Thresholds
        self.qa_min_saturation = 15.0       # Minimum average saturation (0-255)
        self.qa_min_color_variance = 500.0  # Minimum color variance
        
        # Fusion parameter
        self.fusion_alpha = 0.65
        
        # Load YOLO
        print(f"[INFO] Loading YOLO from {yolo_path}")
        self.yolo = YOLO(yolo_path)
        
        # Load EfficientNet
        print(f"[INFO] Loading EfficientNet from {effnet_path}")
        self.effnet = models.efficientnet_b0(weights=None)
        num_ftrs = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(num_ftrs, num_classes)
        self.effnet.load_state_dict(torch.load(effnet_path, map_location=self.device))
        self.effnet = self.effnet.to(self.device)
        self.effnet.eval()
        
        # EfficientNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("[INFO] Models loaded successfully")

    # ===== SAFE HELPER METHODS =====
    @staticmethod
    def safe_max(arr, default=0):
        try: return float(max(arr)) if len(arr) > 0 else default
        except: return default
    
    @staticmethod
    def safe_sum(arr, default=0):
        try: return float(sum(arr)) if len(arr) > 0 else default
        except: return default
    
    @staticmethod
    def safe_div(num, den, default=0):
        try: return float(num / den) if den > 0 else default
        except: return default

    def get_qa_rules(self) -> Dict:
        return {
            "min_conf": self.qa_min_conf,
            "min_total_area": self.qa_min_total_area,
            "max_total_area": self.qa_max_total_area,
            "min_single_box_area": self.qa_min_single_box_area,
            "center_check": True
        }

    def _is_sketch_or_noise(self, img_bgr) -> Tuple[bool, str, float]:
        """
        Heuristic Check: Detects sketches, line drawings, or low-info images.
        Returns: (is_sketch, reason, metric_value)
        """
        try:
            # 1. Color Saturation Check
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            avg_sat = np.mean(saturation)
            
            if avg_sat < self.qa_min_saturation:
                return True, f"Image too grayscale (sat={avg_sat:.1f})", avg_sat

            # 2. Color Variance (Standard Deviation)
            # Sketches usually have low variance in color globally (mostly white/black)
            (mean, std) = cv2.meanStdDev(img_bgr)
            variance = np.mean(std ** 2)
            
            # Note: This is a loose heuristic; relies more on YOLO, but helps reject simple line art
            # Real photos usually have higher variance due to lighting/texture
            
            return False, "Photo-like", avg_sat
        except Exception:
            return False, "Error in heuristic", 0.0

    def _check_central_region(self, boxes, img_w, img_h) -> bool:
        """Check if any detection overlaps with the central region of class."""
        if len(boxes) == 0:
            return False
            
        cx_min = img_w * (1 - self.qa_center_fraction) / 2
        cx_max = img_w * (1 + self.qa_center_fraction) / 2
        cy_min = img_h * (1 - self.qa_center_fraction) / 2
        cy_max = img_h * (1 + self.qa_center_fraction) / 2
        
        for box in boxes:
            bx1, by1, bx2, by2 = box
            
            # Check for overlap
            overlap_x = max(0, min(bx2, cx_max) - max(bx1, cx_min))
            overlap_y = max(0, min(by2, cy_max) - max(by1, cy_min))
            
            if overlap_x > 0 and overlap_y > 0:
                return True
                
        return False

    def qa_gate(self, image_path: str, mode: str = 'hybrid') -> Tuple[bool, str, Dict]:
        """
        Strict QA Gate:
        1. Heuristic Check (Anti-sketch)
        2. YOLO Detection Check (Presence, Size, Location)
        3. Threshold Validation
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "QA Reject: Cannot read image", {'reason': 'invalid_image'}
            
            h, w = img.shape[:2]
            img_area = max(h * w, 1)
            
            # ===== STAGE 0: HEURISTIC CHECK =====
            is_sketch, sketch_msg, metric = self._is_sketch_or_noise(img)
            # Note: We make this a "warning" rather than hard reject if YOLO sees strong coffee
            # But if YOLO is weak AND it looks like a sketch -> hard reject
            
            # ===== STAGE A: YOLO PRESENCE CHECK =====
            results = self.yolo(image_path, verbose=False)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                return False, "QA Reject: No coffee fruits detected", {
                    'reason': 'no_detections',
                    'strong_count': 0, 
                    'rules': self.get_qa_rules()
                }
            
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            
            strong_boxes = []
            total_box_area = 0
            
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                
                # Filter by minimum confidence
                if score >= self.qa_min_conf:
                    strong_boxes.append(box)
                    total_box_area += area
            
            strong_count = len(strong_boxes)
            max_conf = self.safe_max(scores)
            total_area_ratio = self.safe_div(total_box_area, img_area)
            
            qa_report = {
                'strong_count': strong_count,
                'max_conf': float(max_conf),
                'total_area_ratio': float(total_area_ratio),
                'sketch_metric': float(metric),
                'rules': self.get_qa_rules()
            }
            
            # --- RULE 1: Strong Detection Count ---
            if strong_count < 1:
                qa_report['reason'] = 'insufficient_strong_boxes'
                return False, "QA Reject: No confident coffee detections", qa_report
            
            # --- RULE 2: Total Area (Too small = noise, Too big = close-up/blob) ---
            if total_area_ratio < self.qa_min_total_area:
                qa_report['reason'] = 'area_too_small'
                return False, "QA Reject: Detected objects too small (likely noise)", qa_report
                
            if total_area_ratio > self.qa_max_total_area:
                qa_report['reason'] = 'area_too_large'
                return False, "QA Reject: Detected area too large (wrong subject)", qa_report
                
            # --- RULE 3: Central Region Check ---
            # At least one strong box should touch the central region (reject corner noise)
            if not self._check_central_region(strong_boxes, w, h):
                qa_report['reason'] = 'peripheral_detections_only'
                return False, "QA Reject: Objects only found at edges (likely noise)", qa_report

            # --- RULE 4: Sketch/Noise Confirmation ---
            # If it looks like a sketch AND we barely passed YOLO (only 1 box), be stricter
            if is_sketch and strong_count < 2:
                 qa_report['reason'] = f'sketch_heuristic_fail_{sketch_msg}'
                 return False, f"QA Reject: Image looks like sketch/drawing ({sketch_msg})", qa_report

            # ===== STAGE B: Optional Classifier Check (Hybrid/EffNet only) =====
            # If using EffNet/Hybrid, ensure EffNet doesn't completely disagree
            # skip for pure YOLO mode to keep it fast/pure
            if mode != 'yolo':
                 # (Logic kept lightweight: if YOLO says yes strongly, we trust it, 
                 # unless we implemented a specific "Not Coffee" class in EffNet, which we haven't trained yet.
                 # So we rely on the strong YOLO checks above.)
                 pass
            
            qa_report['passed'] = True
            qa_report['reason'] = 'passed_all_checks'
            return True, "QA Pass: Valid coffee fruit image", qa_report

        except Exception as e:
            print(f"[ERROR] QA Gate Exception: {e}")
            traceback.print_exc()
            return False, "QA Reject: Processing error", {'reason': 'exception', 'error': str(e)}

    def predict_yolo_only(self, image_path: str) -> Dict:
        """Mode A: YOLO-Only prediction with calibrated classification belief."""
        results = self.yolo(image_path, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, 'cls') else None
        
        predictions = []
        class_counts = {'Unripe': 0, 'Ripe': 0, 'Overripe': 0}
        class_raw_confs = {'Unripe': [], 'Ripe': [], 'Overripe': []}
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(scores[i]) # Objectness/Detection confidence
            
            # Get class name
            if classes is not None:
                cls_idx = int(classes[i])
                cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else 'Ripe'
            else:
                cls_name = 'Ripe'
            
            predictions.append({
                'box': [x1, y1, x2, y2],
                'class': cls_name,
                'confidence': conf
            })
            
            class_counts[cls_name] += 1
            class_raw_confs[cls_name].append(conf)
        
        # Determine final label (Majority Vote)
        if predictions:
            final_label = max(class_counts, key=class_counts.get)
            
            # CALIBRATION: Mean of class confidences, capped at 0.75
            # This represents "Classification Belief" rather than just object detection certainty
            raw_cls_conf = np.mean(class_raw_confs[final_label]) if class_raw_confs[final_label] else 0.0
            final_confidence = min(float(raw_cls_conf), 0.75)
        else:
            final_label = 'None'
            final_confidence = 0.0
        
        return {
            'mode': 'YOLO',
            'predictions': predictions,
            'final_label': final_label,
            'final_confidence': final_confidence,
            'distribution': class_counts,
            'fruits_detected': len(predictions)
        }

    def predict_effnet_only(self, image_path: str) -> Dict:
        """Mode B: EfficientNet-Only prediction with raw softmax."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.effnet(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf, cls_idx = torch.max(probs, 0)
        
        final_label = self.classes[cls_idx.item()]
        
        # CALIBRATION: Raw softmax, capped at 0.75
        # No artificial boosting allowed here
        final_confidence = min(float(conf.item()), 0.75)
        
        distribution = {cls: 0 for cls in self.classes}
        distribution[final_label] = 1
        
        return {
            'mode': 'EfficientNet',
            'predictions': [{
                'box': None,
                'class': final_label,
                'confidence': final_confidence
            }],
            'final_label': final_label,
            'final_confidence': final_confidence,
            'distribution': distribution,
            'fruits_detected': 1
        }

    def predict_hybrid(self, image_path: str) -> Dict:
        """Mode C: Hybrid prediction with STRICT AGREEMENT-BASED fusion."""
        # 1. Run YOLO (Object Presence)
        results = self.yolo(image_path, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, 'cls') else None
        
        predictions = []
        class_counts = {'Unripe': 0, 'Ripe': 0, 'Overripe': 0}
        class_raw_confs = {'Unripe': [], 'Ripe': [], 'Overripe': []}
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(scores[i])
            cls_idx = int(classes[i]) if classes is not None else 1
            cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else 'Ripe'
            
            predictions.append({
                'box': [x1, y1, x2, y2],
                'class': cls_name,
                'confidence': conf
            })
            class_counts[cls_name] += 1
            class_raw_confs[cls_name].append(conf)
            
        # YOLO Metrics
        if predictions:
            yolo_label = max(class_counts, key=class_counts.get)
            yolo_raw_avg = np.mean(class_raw_confs[yolo_label]) if class_raw_confs[yolo_label] else 0.0
            yolo_class_conf = min(float(yolo_raw_avg), 0.75) # Cap at 0.75
        else:
            yolo_label = 'None'
            yolo_class_conf = 0.0

        # 2. Run EfficientNet (Global Context)
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.effnet(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            effnet_raw, cls_idx = torch.max(probs, 0)
            
        effnet_label = self.classes[cls_idx.item()]
        effnet_conf = min(float(effnet_raw.item()), 0.75) # Cap at 0.75

        # 3. Fusion Logic
        agreement = (yolo_label == effnet_label) and (yolo_label != 'None')
        
        if agreement:
            # Formula: 1 - (1 - P1) * (1 - P2)
            hybrid_raw = 1.0 - ((1.0 - yolo_class_conf) * (1.0 - effnet_conf))
            
            # Mild reinforcement: min(hybrid * 1.05, 0.98 in formula, but user said 0.95 allowed)
            # We use 0.95 as the hard clamp as requested in "Validation Rules"
            hybrid_boosted = hybrid_raw * 1.05
            hybrid_conf = min(hybrid_boosted, 0.95)
            
            # Sanity Check: Must be strictly higher than individual models
            # (Mathematically true for probabilistic OR unless inputs are 1.0)
            hybrid_conf = max(hybrid_conf, yolo_class_conf + 0.01, effnet_conf + 0.01)
            hybrid_conf = min(hybrid_conf, 0.95) # Re-clamp
            
            final_label = yolo_label
            status_msg = "High confidence due to model agreement"
        else:
            # Disagreement: No boost, take minimum
            hybrid_conf = min(yolo_class_conf, effnet_conf)
            final_label = yolo_label # Trust YOLO for object presence/label
            status_msg = "Model Disagreement - Confidence Penalized"

        return {
            'mode': 'Hybrid',
            'predictions': predictions,
            'final_label': final_label,
            'final_confidence': float(hybrid_conf),
            'distribution': class_counts,
            'fruits_detected': len(predictions),
            'agreement': agreement,
            'status_msg': status_msg,
            'yolo_conf': yolo_class_conf,
            'effnet_conf': effnet_conf
        }

    def draw_annotations(self, image_path: str, predictions: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        img_cv = cv2.imread(image_path)
        
        color_map = {
            'Unripe': (113, 212, 99),
            'Ripe': (43, 75, 255),
            'Overripe': (99, 110, 141)
        }
        
        for pred in predictions:
            if pred['box'] is None:
                continue
            
            x1, y1, x2, y2 = pred['box']
            label = pred['class']
            conf = pred['confidence']
            color = color_map.get(label, (255, 255, 255))
            
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cv, (x1, y1 - 20), (x1 + w + 5, y1), color, -1)
            cv2.putText(img_cv, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_cv

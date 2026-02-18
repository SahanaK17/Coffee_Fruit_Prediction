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
        self.qa_min_conf = 0.35             # Minimum confidence for a "strong" detection
        self.qa_min_fruits = 3              # MANDATORY: At least 3 fruits to be a valid cluster
        self.qa_min_total_area = 0.005      # 0.5% min area (increased to block noise)
        self.qa_max_total_area = 0.75       # 75% max area
        self.qa_center_fraction = 0.60      # Middle 60% of image must contain a detection
        
        # Domain validation
        self.qa_effnet_min_conf = 0.60      # Minimum EfficientNet confidence for domain validity
        self.qa_aspect_ratio_min = 0.4      # Range for coffee fruit shape (approx 1:1 or 2:3)
        self.qa_aspect_ratio_max = 2.5
        
        # Sketch/Noise Thresholds
        self.qa_min_saturation = 15.0       # Minimum average saturation (0-255)
        self.qa_min_color_variance = 500.0  # Minimum color variance
        
        # Fusion parameter
        self.fusion_alpha = 0.60
        
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
    def safe_max(arr, default=0.0):
        try: return float(max(arr)) if len(arr) > 0 else default
        except: return default
    
    @staticmethod
    def safe_div(num, den, default=0.0):
        try: return float(num / den) if den > 0 else default
        except: return default

    def get_qa_rules(self) -> Dict:
        return {
            "min_conf": self.qa_min_conf,
            "min_fruits": self.qa_min_fruits,
            "min_total_area": self.qa_min_total_area,
            "domain_conf": self.qa_effnet_min_conf,
            "aspect_ratio": f"{self.qa_aspect_ratio_min}-{self.qa_aspect_ratio_max}"
        }

    def _is_sketch_or_noise(self, img_bgr) -> Tuple[bool, str, float]:
        """
        Heuristic Check: Detects sketches, line drawings, or low-info images.
        """
        try:
            # 1. Color Saturation Check
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            avg_sat = np.mean(saturation)
            
            if avg_sat < self.qa_min_saturation:
                return True, f"Image too grayscale (sat={avg_sat:.1f})", avg_sat

            # 2. Color Variance (Standard Deviation)
            (mean, std) = cv2.meanStdDev(img_bgr)
            variance = np.mean(std ** 2)
            
            if variance < self.qa_min_color_variance:
                return True, f"Low color variance (var={variance:.1f})", float(variance)
            
            return False, "Photo-like", avg_sat
        except Exception:
            return False, "Error in heuristic", 0.0

    def _check_central_region(self, boxes, img_w, img_h) -> bool:
        """Check if any detection overlaps with the central region of class."""
        if len(boxes) == 0: return False
        cx_min = img_w * (1 - self.qa_center_fraction) / 2
        cx_max = img_w * (1 + self.qa_center_fraction) / 2
        cy_min = img_h * (1 - self.qa_center_fraction) / 2
        cy_max = img_h * (1 + self.qa_center_fraction) / 2
        
        for box in boxes:
            bx1, by1, bx2, by2 = box
            overlap_x = max(0, min(bx2, cx_max) - max(bx1, cx_min))
            overlap_y = max(0, min(by2, cy_max) - max(by1, cy_min))
            if overlap_x > 0 and overlap_y > 0: return True
        return False

    def _check_aspect_ratios(self, boxes) -> Tuple[bool, float]:
        """Check if boxes are roughly circular/elliptical (rejects long bars)."""
        if len(boxes) == 0: return False, 0.0
        ratios = []
        valid_count = 0
        for box in boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if h > 0:
                ratio = w / h
                ratios.append(ratio)
                if self.qa_aspect_ratio_min <= ratio <= self.qa_aspect_ratio_max:
                    valid_count += 1
        
        avg_ratio = float(np.mean(ratios)) if ratios else 0.0
        
        # Require at least 50% of detections to be valid coffee shapes
        if len(boxes) > 0 and (valid_count / len(boxes)) < 0.5:
             return False, avg_ratio
        return True, avg_ratio

    def qa_gate(self, image_path: str, mode: str = 'hybrid') -> Tuple[bool, str, Dict]:
        """
        Strict QA Gate with Domain Validation (Coffee vs Non-Coffee)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "QA Reject: Cannot read image", {'reason': 'invalid_image'}
            
            h, w = img.shape[:2]
            img_area = max(h * w, 1)
            
            # ===== STAGE 1: Visual Heuristics & Sketch Check =====
            is_sketch, sketch_msg, metric = self._is_sketch_or_noise(img)
            if is_sketch:
                return False, f"QA Reject: Artificial/Sketch detected ({sketch_msg})", {
                    'reason': 'sketch_detected', 'sketch_metric': metric
                }

            # ===== STAGE 2: YOLO Object Presence & Geometry =====
            results = self.yolo(image_path, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            
            strong_boxes = []
            total_box_area = 0
            
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                # Filter by confidence
                if score >= self.qa_min_conf:
                    strong_boxes.append(box)
                    total_box_area += area
            
            strong_count = len(strong_boxes)
            max_conf = self.safe_max(scores)
            total_area_ratio = self.safe_div(total_box_area, img_area)
            
            qa_report = {
                'reason': 'checking',
                'strong_count': strong_count,
                'max_conf': float(max_conf),
                'total_area_ratio': float(total_area_ratio),
                'rules': self.get_qa_rules()
            }
            
            # [CRITICAL] Rule: Minimum Fruits (Domain Requirement)
            if strong_count < self.qa_min_fruits:
                qa_report['reason'] = 'insufficient_fruits'
                return False, f"QA Reject: Found only {strong_count} fruits (Min {self.qa_min_fruits} required)", qa_report

            # [CRITICAL] Rule: Area Constraints (Too small=noise, Too big=close-up/blob)
            if total_area_ratio < self.qa_min_total_area:
                 qa_report['reason'] = 'area_too_small'
                 return False, "QA Reject: Objects too small/distant", qa_report
            
            if total_area_ratio > self.qa_max_total_area:
                 qa_report['reason'] = 'area_too_large'
                 return False, "QA Reject: Objects too large/close-up", qa_report

            # [CRITICAL] Rule: Aspect Ratio (Valid shapes)
            valid_shapes, avg_ratio = self._check_aspect_ratios(strong_boxes)
            qa_report['avg_aspect_ratio'] = avg_ratio
            if not valid_shapes:
                qa_report['reason'] = 'invalid_object_geometry'
                return False, f"QA Reject: Invalid object shapes (Avg ratio {avg_ratio:.2f})", qa_report

            # [CRITICAL] Rule: Central Focus
            if not self._check_central_region(strong_boxes, w, h):
                qa_report['reason'] = 'peripheral_only'
                return False, "QA Reject: No central detections", qa_report

            # ===== STAGE 3: EfficientNet Domain Validation (Semantic Check) =====
            # Run this for ALL modes to ensure domain correctness as requested
            try:
                img_pil = Image.open(image_path).convert('RGB')
                img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.effnet(img_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    effnet_max, _ = torch.max(probs, 0)
                    effnet_conf = float(effnet_max.item())
                
                qa_report['effnet_domain_conf'] = effnet_conf
                
                # [CRITICAL] Rule: Semantic Confidence
                if effnet_conf < self.qa_effnet_min_conf:
                    qa_report['reason'] = 'low_domain_confidence'
                    return False, f"QA Reject: Classification confidence too low ({effnet_conf:.2f}). Not coffee.", qa_report
            except Exception as e:
                print(f"[WARNING] EffNet Domain Check Failed: {e}")
                # If checking fails, fallback to strict YOLO only? Or fail safe?
                # Fail safe: Reject if we can't verify domain
                return False, "QA Reject: Domain check error", qa_report

            # PASS
            qa_report['passed'] = True
            qa_report['reason'] = 'passed_all_checks'
            return True, "QA Pass: Valid coffee fruit image", qa_report

        except Exception as e:
            print(f"[ERROR] QA Gate Exception: {e}")
            traceback.print_exc()
            return False, "QA Reject: Processing error", {'reason': 'exception', 'error': str(e)}

    def predict_yolo_only(self, image_path: str) -> Dict:
        """Mode A: YOLO-Only prediction with dynamic confidence."""
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
        
        if predictions:
            final_label = max(class_counts, key=class_counts.get)
            raw_cls_conf = np.mean(class_raw_confs[final_label])
            final_confidence = float(raw_cls_conf)
        else:
            final_label = 'None'
            final_confidence = 0.0
        
        return {
            'mode': 'YOLO',
            'predictions': predictions,
            'final_label': final_label,
            'final_confidence': round(final_confidence, 4),
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
        final_confidence = float(conf.item())
        
        distribution = {
            cls_name: float(probs[i].item()) 
            for i, cls_name in enumerate(self.classes)
        }
        
        return {
            'mode': 'EfficientNet',
            'predictions': [{
                'box': None,
                'class': final_label,
                'confidence': final_confidence
            }],
            'final_label': final_label,
            'final_confidence': round(final_confidence, 4),
            'distribution': distribution,
            'fruits_detected': 1
        }

    def predict_hybrid(self, image_path: str) -> Dict:
        """Mode C: Hybrid prediction with Calibrated Agreement Fusion."""
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
            
        if predictions:
            yolo_label = max(class_counts, key=class_counts.get)
            yolo_conf = float(np.mean(class_raw_confs[yolo_label]))
        else:
            yolo_label = 'None'
            yolo_conf = 0.0

        # 2. Run EfficientNet (Global Context)
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.effnet(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            effnet_raw, cls_idx = torch.max(probs, 0)
            
        effnet_label = self.classes[cls_idx.item()]
        effnet_conf = float(effnet_raw.item())

        # 3. Fusion Logic (Semantic Agreement)
        agreement = (yolo_label == effnet_label) and (yolo_label != 'None')
        
        # Base confidences
        cy = yolo_conf
        ce = effnet_conf
        
        # Threshold for boosting
        boost_thresh = 0.65
        
        if agreement:
            if cy >= boost_thresh and ce >= boost_thresh:
                # Strong Agreement: Boost above individuals
                # Weighted toward the higher confidence + bonus
                base = max(cy, ce)
                hybrid_conf = min(base + 0.05, 0.99)
                status_msg = "Agreement (Boosted)"
            else:
                # Weak Agreement: Average
                hybrid_conf = (cy + ce) / 2.0
                status_msg = "Agreement (Weak)"
            final_label = yolo_label
        else:
            # Disagreement: Penalize
            # Hybrid confidence should NOT exceed the individual models
            # Logic: Trust the more confident one, but reduce confidence because of disagreement
            
            if cy > ce:
                final_label = yolo_label
                hybrid_conf = cy * 0.85 
                status_msg = "Disagreement (YOLO Favored)"
            else:
                final_label = effnet_label
                hybrid_conf = ce * 0.85
                status_msg = "Disagreement (EffNet Favored)"

        # Final Clamp (Realism)
        hybrid_conf = float(min(max(hybrid_conf, 0.0), 1.0))

        # Debug Logs
        print(f"\n[HYBRID FUSION]")
        print(f"  YOLO: {yolo_label} ({yolo_conf:.4f})")
        print(f"  EffNet: {effnet_label} ({effnet_conf:.4f})")
        print(f"  Agreement: {agreement}")
        print(f"  Hybrid: {hybrid_conf:.4f}")
        print(f"  Status: {status_msg}")

        return {
            'mode': 'Hybrid',
            'predictions': predictions,
            'final_label': final_label,
            'final_confidence': round(hybrid_conf, 4),
            'distribution': class_counts,
            'fruits_detected': len(predictions),
            'agreement': agreement,
            'status_msg': status_msg,
            'yolo_confidence': round(yolo_conf, 4),
            'eff_confidence': round(effnet_conf, 4),
            'yolo_label': yolo_label,
            'eff_label': effnet_label
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

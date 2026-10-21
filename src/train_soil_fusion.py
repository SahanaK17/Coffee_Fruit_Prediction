"""
Training Script for Soil-Fused Coffee Fruit Maturity Model
Extends the existing EfficientNet training pipeline with SoilGrids soil features.

Usage:
    python src/train_soil_fusion.py

This script:
1. Loads the existing classification dataset (image crops)
2. Assigns soil features using location metadata (lat, lon → SoilGrids API)
3. Creates the SoilFusionModel with the pretrained EfficientNet backbone
4. Trains with combined image + soil features
5. Compares performance: image-only vs image+soil
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
import sys
import time
import copy
import json
import numpy as np
import random
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.soil_fusion_model import SoilFusionModel
from src.soil_service import (
    get_soil_data,
    preprocess_soil_features,
    SOIL_COLUMNS,
)


# ===== CONFIGURATION =====
DATA_DIR = PROJECT_ROOT / "classification_dataset"
OUTPUT_DIR = PROJECT_ROOT / "models" / "soil_fusion"
EFFNET_WEIGHTS = PROJECT_ROOT / "models" / "effnet" / "effnet" / "best.pth"
LOCATION_META = PROJECT_ROOT / "data" / "location_metadata.csv"

# Training hyperparams
NUM_EPOCHS = 3
BATCH_SIZE = 32
LR_FROZEN = 0.001        # Learning rate for frozen-backbone phase
LR_FINETUNE = 0.0001     # Learning rate for full fine-tuning phase
FROZEN_EPOCHS = 1         # Train fusion layers only for first N epochs
NUM_CLASSES = 3
NUM_SOIL_FEATURES = 5


# ===== LOCATION METADATA =====

def generate_location_metadata():
    """
    Generate location metadata for the training dataset.
    
    Since the Roboflow dataset doesn't include GPS coordinates,
    we simulate realistic coordinates from major coffee-growing regions.
    
    In production, each image would have its own (lat, lon) from field data.
    """
    if LOCATION_META.exists():
        print(f"[TRAIN] Location metadata already exists at {LOCATION_META}")
        return pd.read_csv(LOCATION_META)
    
    # Major coffee-growing regions with realistic coordinates
    # These represent typical growing areas in Karnataka, Kerala, Tamil Nadu (India)
    # and other major coffee regions globally
    coffee_regions = [
        # Karnataka (Coorg/Chikmagalur/Hassan belt)
        {"name": "Coorg_1",        "lat": 12.4244, "lon": 75.7382},
        {"name": "Coorg_2",        "lat": 12.3375, "lon": 75.8069},
        {"name": "Chikmagalur_1",  "lat": 13.3161, "lon": 75.7720},
        {"name": "Chikmagalur_2",  "lat": 13.2257, "lon": 75.6761},
        {"name": "Hassan",         "lat": 13.0068, "lon": 76.1004},
        # Kerala (Wayanad/Idukki)
        {"name": "Wayanad",        "lat": 11.6854, "lon": 76.1320},
        {"name": "Idukki",         "lat": 9.8494,  "lon": 76.9720},
        # Tamil Nadu (Nilgiris)
        {"name": "Nilgiris",       "lat": 11.4916, "lon": 76.7337},
        # Brazil (major producer)
        {"name": "MinasGerais",    "lat": -21.1767, "lon": -44.9455},
        {"name": "SaoPaulo",       "lat": -22.2587, "lon": -48.2044},
        # Colombia
        {"name": "Huila",          "lat": 2.5359,  "lon": -75.5277},
        # Ethiopia
        {"name": "Sidamo",         "lat": 6.7138,  "lon": 38.4745},
    ]
    
    # Fetch soil data for each region (will be cached)
    records = []
    for region in coffee_regions:
        soil = get_soil_data(region["lat"], region["lon"])
        records.append({
            "region": region["name"],
            "lat": region["lat"],
            "lon": region["lon"],
            **{k: v if v is not None else -1 for k, v in soil.items()}
        })
        print(f"  [{region['name']}] pH={soil.get('ph')}, clay={soil.get('clay')}, "
              f"sand={soil.get('sand')}, silt={soil.get('silt')}, soc={soil.get('soc')}")
    
    df = pd.DataFrame(records)
    LOCATION_META.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCATION_META, index=False)
    print(f"[TRAIN] Saved location metadata with {len(records)} regions")
    return df


# ===== CUSTOM DATASET =====

class SoilFusionDataset(Dataset):
    """
    Dataset that pairs images with soil features.
    
    Each sample returns:
        image_tensor: (3, 224, 224)
        soil_tensor:  (5,) normalized
        label:        int class index
    """
    
    def __init__(self, image_dataset, soil_features_list: list, transform=None):
        """
        Args:
            image_dataset: torchvision ImageFolder dataset
            soil_features_list: list of numpy arrays, one per sample
            transform: additional transforms (already applied by ImageFolder)
        """
        self.image_dataset = image_dataset
        self.soil_features = soil_features_list
        self.transform = transform
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        soil = torch.tensor(self.soil_features[idx], dtype=torch.float32)
        return image, soil, label


def assign_soil_to_images(image_dataset, locations_df: pd.DataFrame) -> list:
    """
    Assign soil features to each image in the dataset.
    
    Strategy:
    - Randomly assign each image to one of the known coffee regions
    - This simulates a real-world scenario where images come from various locations
    - In production, each image would have its actual GPS metadata
    """
    n_samples = len(image_dataset)
    n_regions = len(locations_df)
    
    soil_features_list = []
    
    for i in range(n_samples):
        # Assign to a region (round-robin for balanced distribution)
        region_idx = i % n_regions
        row = locations_df.iloc[region_idx]
        
        soil_data = {
            "ph":   float(row["ph"]) if row["ph"] >= 0 else None,
            "clay": float(row["clay"]) if row["clay"] >= 0 else None,
            "sand": float(row["sand"]) if row["sand"] >= 0 else None,
            "silt": float(row["silt"]) if row["silt"] >= 0 else None,
            "soc":  float(row["soc"]) if row["soc"] >= 0 else None,
        }
        
        normalized = preprocess_soil_features(soil_data)
        soil_features_list.append(normalized)
    
    return soil_features_list


# ===== BASELINE: IMAGE-ONLY MODEL EVALUATION =====

def evaluate_baseline(model_path: str, dataloader, class_names, device) -> dict:
    """
    Evaluate the existing image-only EfficientNet model (baseline).
    Returns accuracy and per-class metrics.
    """
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (Image-Only EfficientNet)")
    print("=" * 60)
    
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    print(f"Baseline Accuracy: {acc:.4f}")
    print(report_str)
    
    return {
        "accuracy": acc,
        "report_str": report_str,
        "report_dict": report_dict,
    }


# ===== TRAINING =====

def train_soil_fusion():
    """Main training function for the soil-fused model."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"SOIL FUSION MODEL TRAINING")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")
    
    # ===== 1. Generate/Load Location Metadata =====
    print("[STEP 1] Loading location metadata and soil data...")
    locations_df = generate_location_metadata()
    print(f"  Regions loaded: {len(locations_df)}")
    
    # ===== 2. Prepare Image Datasets =====
    print("[STEP 2] Loading image datasets...")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    if not DATA_DIR.exists():
        print(f"[ERROR] Classification dataset not found at {DATA_DIR}")
        print("  Run src/dataset_utils.py first to create the dataset.")
        sys.exit(1)
    
    image_datasets = {}
    fusion_datasets = {}
    dataloaders = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"[WARNING] Split '{split}' not found, skipping")
            continue
        
        # Load image dataset
        img_ds = datasets.ImageFolder(str(split_dir), data_transforms[split])
        
        # Subsample for demonstration to 64 images max
        if len(img_ds) > 64:
            indices = list(range(len(img_ds)))
            random.seed(42)
            random.shuffle(indices)
            indices = indices[:64]
            img_ds = torch.utils.data.Subset(img_ds, indices)
            img_ds.classes = image_datasets.get('train', datasets.ImageFolder(str(DATA_DIR / 'train'))).classes

        image_datasets[split] = img_ds
        
        # Assign soil features to each image
        soil_list = assign_soil_to_images(img_ds, locations_df)
        
        # Create fusion dataset
        fusion_ds = SoilFusionDataset(img_ds, soil_list)
        fusion_datasets[split] = fusion_ds
        
        # Create dataloader
        dataloaders[split] = DataLoader(
            fusion_ds, batch_size=BATCH_SIZE, shuffle=(split == 'train'), num_workers=0
        )
        
        print(f"  {split}: {len(img_ds)} images")
    
    # We need to manually provide class names since Subset doesn't have it at the top level
    base_train_ds = datasets.ImageFolder(str(DATA_DIR / 'train'))
    class_names = base_train_ds.classes
    print(f"  Classes: {class_names}")
    
    # ===== 3. Evaluate Baseline (Image-Only) =====
    print("\n[STEP 3] Evaluating baseline model...")
    
    # Create a simple image-only dataloader for baseline
    baseline_test_ds = datasets.ImageFolder(
        str(DATA_DIR / 'test'), data_transforms['test']
    )
    if len(baseline_test_ds) > 64:
        indices = list(range(len(baseline_test_ds)))
        random.seed(42)
        random.shuffle(indices)
        indices = indices[:64]
        baseline_test_ds = torch.utils.data.Subset(baseline_test_ds, indices)
        
    baseline_test_loader = DataLoader(
        baseline_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    baseline_results = evaluate_baseline(
        str(EFFNET_WEIGHTS), baseline_test_loader, class_names, device
    )
    
    # ===== 4. Create Soil Fusion Model =====
    print("\n[STEP 4] Creating Soil Fusion Model...")
    
    model = SoilFusionModel(
        num_classes=NUM_CLASSES,
        num_soil_features=NUM_SOIL_FEATURES,
        pretrained_effnet_path=str(EFFNET_WEIGHTS),
    )
    model = model.to(device)
    
    print(f"  Total parameters: {model.get_total_params():,}")
    
    # ===== 5. Phase 1: Frozen Backbone Training =====
    print(f"\n[STEP 5] Phase 1: Training fusion layers (backbone frozen) for {FROZEN_EPOCHS} epochs...")
    model.freeze_backbone()
    print(f"  Trainable parameters: {model.get_trainable_params():,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FROZEN
    )
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'phase': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(FROZEN_EPOCHS):
        print(f"\nEpoch {epoch}/{FROZEN_EPOCHS - 1} (Phase 1: Frozen)")
        print("-" * 40)
        
        for phase in ['train', 'valid']:
            if phase not in dataloaders:
                continue
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            n_samples = 0
            
            for images, soil, labels in dataloaders[phase]:
                images = images.to(device)
                soil = soil.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, soil)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                n_samples += images.size(0)
            
            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects.double() / n_samples
            
            print(f"  {phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        history['phase'].append('frozen')
    
    # ===== 6. Phase 2: Full Fine-tuning =====
    finetune_epochs = NUM_EPOCHS - FROZEN_EPOCHS
    print(f"\n[STEP 6] Phase 2: Full fine-tuning (backbone unfrozen) for {finetune_epochs} epochs...")
    model.unfreeze_backbone()
    print(f"  Trainable parameters: {model.get_trainable_params():,}")
    
    optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    for epoch in range(finetune_epochs):
        epoch_num = FROZEN_EPOCHS + epoch
        print(f"\nEpoch {epoch_num}/{NUM_EPOCHS - 1} (Phase 2: Fine-tune)")
        print("-" * 40)
        
        for phase in ['train', 'valid']:
            if phase not in dataloaders:
                continue
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            n_samples = 0
            
            for images, soil, labels in dataloaders[phase]:
                images = images.to(device)
                soil = soil.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, soil)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                n_samples += images.size(0)
            
            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects.double() / n_samples
            
            print(f"  {phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        history['phase'].append('finetune')
        scheduler.step()
    
    # ===== 7. Save Best Model =====
    print(f"\n[STEP 7] Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    
    save_path = OUTPUT_DIR / "soil_fusion_best.pth"
    torch.save(model.state_dict(), str(save_path))
    print(f"  Model saved to {save_path}")
    
    # Save model config for inference
    config = {
        "num_classes": NUM_CLASSES,
        "num_soil_features": NUM_SOIL_FEATURES,
        "class_names": class_names,
        "image_feature_dim": model.image_feature_dim,
        "best_val_acc": float(best_acc),
        "frozen_epochs": FROZEN_EPOCHS,
        "total_epochs": NUM_EPOCHS,
    }
    config_path = OUTPUT_DIR / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {config_path}")
    
    # ===== 8. Evaluate Fusion Model on Test Set =====
    print("\n[STEP 8] Evaluating Soil Fusion Model on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    if 'test' in dataloaders:
        with torch.no_grad():
            for images, soil, labels in dataloaders['test']:
                images = images.to(device)
                soil = soil.to(device)
                labels = labels.to(device)
                outputs = model(images, soil)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        fusion_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        fusion_report_str = classification_report(all_labels, all_preds, target_names=class_names)
        fusion_report_dict = classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )
        
        print(f"\nFusion Model Accuracy: {fusion_acc:.4f}")
        print(fusion_report_str)
        
        # Save classification report
        with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
            f.write("SOIL FUSION MODEL - Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(fusion_report_str)
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Soil Fusion Model')
        plt.savefig(str(OUTPUT_DIR / 'confusion_matrix.png'))
        plt.close()
    
    # ===== 9. Comparison Report =====
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    baseline_acc = baseline_results["accuracy"]
    improvement = fusion_acc - baseline_acc if 'test' in dataloaders else 0
    improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
    
    comparison = {
        "baseline_accuracy": float(baseline_acc),
        "fusion_accuracy": float(fusion_acc) if 'test' in dataloaders else None,
        "improvement": float(improvement),
        "improvement_pct": float(improvement_pct),
        "baseline_report": baseline_results.get("report_dict", {}),
        "fusion_report": fusion_report_dict if 'test' in dataloaders else {},
    }
    
    print(f"  Image-Only (Baseline):  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"  Image + Soil (Fusion):  {fusion_acc:.4f} ({fusion_acc*100:.2f}%)")
    print(f"  Improvement:            {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print("=" * 60)
    
    # Save comparison
    with open(OUTPUT_DIR / "comparison_report.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # ===== 10. Plot Training Curves =====
    print("\n[STEP 10] Saving training curves...")
    plot_training_curves(history, OUTPUT_DIR)
    plot_comparison(baseline_acc, fusion_acc, class_names, 
                    baseline_results.get("report_dict", {}),
                    fusion_report_dict if 'test' in dataloaders else {},
                    OUTPUT_DIR)
    
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Best model: {save_path}")
    print(f"  Comparison: {OUTPUT_DIR / 'comparison_report.json'}")
    print(f"{'=' * 60}\n")
    
    return comparison


def plot_training_curves(history: dict, output_dir: Path):
    """Plot training/validation curves for the fusion model."""
    epochs = range(len(history['train_loss']))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss', markersize=3)
    axes[0].axvline(x=FROZEN_EPOCHS - 0.5, color='gray', linestyle='--', alpha=0.5, label='Unfreeze Backbone')
    axes[0].set_title('Soil Fusion Model - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=3)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Val Acc', markersize=3)
    axes[1].axvline(x=FROZEN_EPOCHS - 0.5, color='gray', linestyle='--', alpha=0.5, label='Unfreeze Backbone')
    axes[1].set_title('Soil Fusion Model - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Soil Fusion Model Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'training_curves.png'), dpi=150)
    plt.close()


def plot_comparison(baseline_acc, fusion_acc, class_names, baseline_report, fusion_report, output_dir):
    """Plot baseline vs fusion comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall accuracy bar chart
    models = ['Image Only\n(Baseline)', 'Image + Soil\n(Fusion)']
    accuracies = [baseline_acc * 100, fusion_acc * 100]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = axes[0].bar(models, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Overall Accuracy Comparison')
    axes[0].set_ylim(0, 105)
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Per-class F1 comparison
    if baseline_report and fusion_report:
        x = np.arange(len(class_names))
        width = 0.35
        
        baseline_f1 = [baseline_report.get(c, {}).get('f1-score', 0) * 100 for c in class_names]
        fusion_f1 = [fusion_report.get(c, {}).get('f1-score', 0) * 100 for c in class_names]
        
        axes[1].bar(x - width/2, baseline_f1, width, label='Image Only', color='#FF6B6B', edgecolor='black', linewidth=0.5)
        axes[1].bar(x + width/2, fusion_f1, width, label='Image + Soil', color='#4ECDC4', edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('F1-Score (%)')
        axes[1].set_title('Per-Class F1-Score Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names)
        axes[1].legend()
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Baseline vs Soil Fusion Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'comparison_chart.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    train_soil_fusion()

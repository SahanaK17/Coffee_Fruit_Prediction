import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = r"c:\Users\hp\OneDrive\projects\Coffee-prediction\models\effnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Training Curves (Placeholder)
plt.figure(figsize=(10, 5))
epochs = range(1, 11)
train_acc = [0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94]
val_acc = [0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.88, 0.89, 0.90, 0.91]
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy (Placeholder)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
plt.close()

# 2. Confusion Matrix (Placeholder)
import seaborn as sns
plt.figure(figsize=(8, 6))
cm = np.array([[50, 5, 2], [3, 45, 4], [1, 2, 48]]) # Dummy data
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unripe', 'Ripe', 'Overripe'], yticklabels=['Unripe', 'Ripe', 'Overripe'])
plt.title('Confusion Matrix (Placeholder)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# 3. Classification Report (Placeholder)
report = """              precision    recall  f1-score   support

      Unripe       0.92      0.88      0.90        57
        Ripe       0.86      0.87      0.86        52
    Overripe       0.94      0.96      0.95        51

    accuracy                           0.91       160
   macro avg       0.91      0.90      0.90       160
weighted avg       0.91      0.91      0.91       160
"""
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

print("Placeholder metrics created.")

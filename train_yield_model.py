import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import os
from pathlib import Path

def generate_synthetic_data(n_samples=300):
    """
    Generate a synthetic dataset for coffee yield prediction.
    Features:
    - fruit_count: 5 to 150
    - avg_fruit_size: 0.001 to 0.01 (normalized area)
    - fruit_size_variance: 0.0001 to 0.002
    - canopy_coverage: 0.1 to 0.9
    - deep_features: simulated as a 1280-dim vector (using PCA to reduce for training stability in this example)
    """
    np.random.seed(42)
    
    fruit_counts = np.random.randint(5, 150, n_samples)
    avg_fruit_sizes = np.random.uniform(0.001, 0.01, n_samples)
    fruit_size_variances = np.random.uniform(0.0001, 0.002, n_samples)
    canopy_coverages = np.random.uniform(0.1, 0.9, n_samples)
    
    # Deep features: represent plant structure, here we simulate 10 "latent" features
    deep_features = np.random.normal(0, 1, (n_samples, 10))
    
    # Yield logic: complex interaction
    # More fruits + bigger fruits -> more weight
    # Canopy coverage -> health indicator
    # Variance -> uniformity (negative impact if high?)
    
    yield_kg = (
        (fruit_counts * 0.12) * (avg_fruit_sizes / 0.004) * (canopy_coverages + 0.5) -
        (fruit_size_variances * 100) +
        np.sum(deep_features * 0.05, axis=1) +
        np.random.normal(0, 0.2, n_samples)
    )
    
    # Ensure yield is realistic (0.5 to 15 kg per branch/view)
    yield_kg = np.clip(yield_kg, 0.5, 15.0)
    
    # Construct DataFrame
    data = {
        'fruit_count': fruit_counts,
        'avg_fruit_size': avg_fruit_sizes,
        'fruit_size_variance': fruit_size_variances,
        'canopy_coverage': canopy_coverages
    }
    
    # Add deep feature columns
    for i in range(10):
        data[f'deep_feat_{i}'] = deep_features[:, i]
        
    data['yield_kg'] = yield_kg
    
    return pd.DataFrame(data)

def train_model():
    print("Generating comprehensive synthetic dataset (300 samples)...")
    df = generate_synthetic_data(300)
    
    # Features include tabular counts and deep features
    feature_cols = [c for c in df.columns if c != 'yield_kg']
    X = df[feature_cols]
    y = df['yield_kg']
    
    print(f"Training XGBoost Regressor with {len(feature_cols)} features...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    
    # Create models directory if not exists
    os.makedirs('models/yield', exist_ok=True)
    
    model_path = 'models/yield/yield_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names to ensure consistent inference
    joblib.dump(feature_cols, 'models/yield/feature_names.joblib')
    
    # Save a sample of the dataset for reference
    df.head(10).to_csv('data/yield_synthetic_sample.csv', index=False)
    print("Sample dataset saved to data/yield_synthetic_sample.csv")

if __name__ == "__main__":
    train_model()

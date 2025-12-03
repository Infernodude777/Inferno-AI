import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import os

# Generate synthetic dataset for demonstration
def generate_sample_data(n_samples=1000):
    """Generate a simple classification dataset"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def train_model():
    print("🔥 Inferno-AI Training Started...")
    
    # Load or generate data
    X, y = generate_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n📊 Confusion matrix saved to results/confusion_matrix.png")
    
    # Save model
    joblib.dump(model, 'models/inferno_model.pkl')
    print("\n💾 Model saved to models/inferno_model.pkl")
    
    return metrics

if __name__ == '__main__':
    metrics = train_model()
    print("\n🔥 Training Complete!")

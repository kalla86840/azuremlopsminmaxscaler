import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import os

def load_data():
    script_dir = os.path.dirname(__file__)
    X_path = os.path.join(script_dir, 'X_scaled.csv')
    y_path = os.path.join(script_dir, 'y.csv')
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    return X, y

def load_model():
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, 'model.joblib')
    return joblib.load(model_path)

def evaluate_model():
    X, y = load_data()
    model = load_model()
    predictions = model.predict(X)

    lines = []
    lines.append(f"R-squared: {r2_score(y, predictions):.4f}")
    lines.append(f"RMSE: {np.sqrt(mean_squared_error(y, predictions)):.4f}")

    try:
        acc = accuracy_score(y, np.round(predictions))
        report = classification_report(y, np.round(predictions))
        lines.append(f"Accuracy Score: {acc:.4f}")
        lines.append("Classification Report:")
        lines.append(report)
    except Exception as e:
        lines.append(f"Accuracy/Classification Report not applicable: {str(e)}")

    # Save to text file
    script_dir = os.path.dirname(__file__)
    metrics_path = os.path.join(script_dir, 'metrics_report.txt')
    with open(metrics_path, 'w') as f:
        f.write("\n".join(lines))

    print("\n".join(lines))

if __name__ == "__main__":
    evaluate_model()
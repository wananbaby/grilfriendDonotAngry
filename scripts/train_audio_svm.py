import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure project root is on path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.audio_features import load_dataset


def train(data_dir: str, output_path: str, test_size: float = 0.2, random_state: int = 42):
    X, y, label_map = load_dataset(data_dir)
    if X.shape[0] == 0:
        raise RuntimeError(f"No audio files found in {data_dir}. Expect structure: data_dir/class_name/*.wav")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print("Evaluation report:")
    print(report)

    out_dir = Path(os.path.dirname(output_path)) if os.path.dirname(output_path) else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    label_map_path = out_dir / (Path(output_path).stem + "_labels.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Saved model to {output_path}")
    print(f"Saved label map to {label_map_path}")


def main():
    parser = argparse.ArgumentParser(description="Train audio emotion classifier (SVM) on local dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root: subfolders per class with audio files")
    parser.add_argument("--output_path", type=str, default="models/audio_svm.pkl", help="Path to save trained model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation split size")
    args = parser.parse_args()
    train(args.data_dir, args.output_path, args.test_size)


if __name__ == "__main__":
    main()

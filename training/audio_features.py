import os
from typing import List, Tuple, Dict

import numpy as np
import librosa


def extract_features(file_path: str, sr: int = 16000) -> np.ndarray:
    y, orig_sr = librosa.load(file_path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(64, dtype=np.float32)
    hop_length = 512
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    mfcc_mean = mfcc.mean(axis=1).ravel()
    mfcc_std = mfcc.std(axis=1).ravel()
    # Spectral features
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).mean(axis=1).ravel()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length).mean(axis=1).ravel()
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length).mean(axis=1).ravel()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length).mean(axis=1).ravel()
    # Tempo (approximate prosody)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_arr = np.array([float(tempo)], dtype=np.float32)
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        zcr, spec_bw, spec_contrast, rolloff,
        tempo_arr
    ]).astype(np.float32)
    return features


def load_dataset(data_dir: str, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    X: List[np.ndarray] = []
    y: List[int] = []
    class_names: List[str] = []

    # class directories
    for entry in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, entry)
        if not os.path.isdir(cls_path):
            continue
        class_names.append(entry)

    label_map = {i: name for i, name in enumerate(class_names)}
    name_to_label = {name: i for i, name in label_map.items()}

    for class_name in class_names:
        cls_dir = os.path.join(data_dir, class_name)
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                    fp = os.path.join(root, f)
                    try:
                        feats = extract_features(fp, sr=sr)
                        X.append(feats)
                        y.append(name_to_label[class_name])
                    except Exception:
                        # Skip unreadable files
                        continue

    if len(X) == 0:
        return np.empty((0, 64), dtype=np.float32), np.empty((0,), dtype=np.int64), label_map
    X_arr = np.stack(X, axis=0)
    y_arr = np.array(y, dtype=np.int64)
    return X_arr, y_arr, label_map

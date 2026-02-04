import argparse
import os
from pathlib import Path
import numpy as np
import soundfile as sf


def gen_tone(duration_s=1.5, sr=16000, freq=220.0, amplitude=0.2, noise_level=0.02):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    noise = noise_level * np.random.randn(t.size)
    return np.clip(tone + noise, -1.0, 1.0)


def write_wav(path, y, sr=16000):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")


def create_dataset(root: str, samples_per_class: int = 40, sr: int = 16000):
    classes = {
        "neutral": {"freq_range": (150, 250), "amp_range": (0.15, 0.25), "noise": (0.01, 0.03)},
        "angry": {"freq_range": (300, 600), "amp_range": (0.25, 0.45), "noise": (0.02, 0.06)},
    }
    for cls, cfg in classes.items():
        cls_dir = os.path.join(root, cls)
        Path(cls_dir).mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            freq = np.random.uniform(*cfg["freq_range"])
            amp = np.random.uniform(*cfg["amp_range"])
            noise = np.random.uniform(*cfg["noise"])
            dur = np.random.uniform(1.0, 2.0)
            y = gen_tone(duration_s=dur, sr=sr, freq=freq, amplitude=amp, noise_level=noise)
            write_wav(os.path.join(cls_dir, f"{cls}_{i:03d}.wav"), y, sr=sr)
    print(f"Created dataset at {root}")


def main():
    parser = argparse.ArgumentParser(description="Generate a dummy audio dataset for quick training test.")
    parser.add_argument("--output_dir", type=str, default="data/demo_audio", help="Output dataset root")
    parser.add_argument("--samples_per_class", type=int, default=40, help="Number of samples per class")
    args = parser.parse_args()
    create_dataset(args.output_dir, args.samples_per_class)


if __name__ == "__main__":
    main()

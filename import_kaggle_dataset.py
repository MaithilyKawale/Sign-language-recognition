"""Import Kaggle sign datasets into the project's common landmark format.

Examples:
  python import_kaggle_dataset.py --source image-folders --input-dir dataset
  python import_kaggle_dataset.py --source asl-parquet --train-csv train.csv --landmarks-dir train_landmark_files --max-per-label 100

Image folder format: dataset/LABEL/image.jpg
ASL landmark format: Google Kaggle ASL Signs train.csv + train_landmark_files/.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import cv2
import numpy as np

from utils import FEATURE_SIZE, FACE_INDICES, extract_features


def clean_label(value: str, prefix: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").upper()
    return f"{prefix}_{value}" if prefix else value


def append_static(rows: list[tuple[str, np.ndarray]], output: Path) -> int:
    if not rows:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_SIZE)])
        for label, vector in rows:
            writer.writerow([label] + vector.astype(float).tolist())
    return len(rows)


def import_image_folders(root: Path, prefix: str, max_per_class: int) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("Install project requirements first: pip install -r requirements.txt") from exc
    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {root}")
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit("Expected folders like dataset/A/image.jpg and dataset/B/image.jpg.")
    output = Path("data/static.csv")
    pending: list[tuple[str, np.ndarray]] = []
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total_seen = total_detected = 0
    with mp.solutions.holistic.Holistic(
        static_image_mode=True, model_complexity=0,
        min_detection_confidence=0.45
    ) as model:
        for folder in classes:
            label = clean_label(folder.name, prefix)
            count = 0
            for image_path in sorted(folder.rglob("*")):
                if image_path.suffix.lower() not in supported:
                    continue
                if max_per_class and count >= max_per_class:
                    break
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                total_seen += 1
                result = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                vector, quality = extract_features(result)
                # Static image datasets often contain only hands, so don't require face/body.
                if quality < 0.30:
                    continue
                pending.append((label, vector))
                count += 1
                total_detected += 1
                if len(pending) >= 500:
                    append_static(pending, output)
                    pending.clear()
            print(f"{folder.name}: imported {count} images as {label}")
    append_static(pending, output)
    print(f"Imported {total_detected}/{total_seen} usable images into {output}.")
    print("These image samples are static examples; they do not train dynamic signs.")


def _landmark_frame_to_vector(frame_rows) -> np.ndarray:
    """Convert one Kaggle ASL Signs frame to the same feature layout as webcam data."""
    pose = np.zeros((33, 3), dtype=np.float32)
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    face = np.zeros((len(FACE_INDICES), 3), dtype=np.float32)
    face_slot = {idx: slot for slot, idx in enumerate(FACE_INDICES)}
    for row in frame_rows.itertuples(index=False):
        kind = str(row.type)
        idx = int(row.landmark_index)
        xyz = (float(row.x), float(row.y), float(row.z))
        if kind == "pose" and 0 <= idx < 33:
            pose[idx] = xyz
        elif kind == "left_hand" and 0 <= idx < 21:
            left[idx] = xyz
        elif kind == "right_hand" and 0 <= idx < 21:
            right[idx] = xyz
        elif kind == "face" and idx in face_slot:
            face[face_slot[idx]] = xyz
    return np.concatenate([pose.ravel(), left.ravel(), right.ravel(), face.ravel()]).astype(np.float32)


def import_asl_parquet(train_csv: Path, landmarks_dir: Path, max_per_label: int, sequence_length: int) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("ASL parquet import needs pandas and pyarrow. Install with: pip install pandas pyarrow") from exc
    if not train_csv.exists():
        raise SystemExit(f"Training CSV not found: {train_csv}")
    if not landmarks_dir.is_dir():
        raise SystemExit(f"Landmark directory not found: {landmarks_dir}")
    metadata = pd.read_csv(train_csv)
    required = {"path", "sign"}
    if not required.issubset(metadata.columns):
        raise SystemExit(f"{train_csv} must contain columns: path, sign")
    out_root = Path("data/dynamic")
    counts: dict[str, int] = {}
    imported = skipped = 0
    for row in metadata.itertuples(index=False):
        raw_label = str(getattr(row, "sign"))
        label = clean_label(raw_label, "ASL")
        if max_per_label and counts.get(label, 0) >= max_per_label:
            continue
        relative = Path(str(getattr(row, "path")))
        parquet_path = relative if relative.is_absolute() else Path(".") / relative
        if not parquet_path.exists():
            # Some downloads store only the landmark directory separately.
            matches = list(landmarks_dir.glob(f"**/{relative.name}"))
            if not matches:
                skipped += 1
                continue
            parquet_path = matches[0]
        try:
            frame_data = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
            vectors = [_landmark_frame_to_vector(group) for _, group in frame_data.groupby("frame", sort=True)]
            if not vectors:
                skipped += 1
                continue
            array = np.asarray(vectors, dtype=np.float32)
            # Uniformly sample across the clip, preserving motion information.
            indices = np.linspace(0, len(array) - 1, sequence_length).round().astype(int)
            array = array[indices]
            dest = out_root / label
            dest.mkdir(parents=True, exist_ok=True)
            index = len(list(dest.glob("*.npz")))
            np.savez_compressed(dest / f"kaggle_{index:06d}.npz", X=array, label=label)
            counts[label] = counts.get(label, 0) + 1
            imported += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"Skipping {parquet_path}: {exc}")
    print(f"Imported {imported} ASL sequences; skipped {skipped}. Output: {out_root}")
    print("Next run: python train_dynamic.py")
    print("For a first run, use --max-per-label 20 or 50; the full Kaggle dataset is very large.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["image-folders", "asl-parquet"], required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("dataset"),
                        help="Root folder of class-labeled images for image-folders mode.")
    parser.add_argument("--prefix", default="ISL",
                        help="Prefix added to image-folder labels to avoid mixing ISL and ASL labels.")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Maximum images per folder; 0 means all.")
    parser.add_argument("--train-csv", type=Path, default=Path("train.csv"))
    parser.add_argument("--landmarks-dir", type=Path, default=Path("train_landmark_files"))
    parser.add_argument("--max-per-label", type=int, default=50,
                        help="Maximum ASL parquet clips per label; 0 means all.")
    parser.add_argument("--sequence-length", type=int, default=30)
    args = parser.parse_args()
    if args.source == "image-folders":
        import_image_folders(args.input_dir, args.prefix.upper(), args.max_per_class)
    else:
        import_asl_parquet(args.train_csv, args.landmarks_dir, args.max_per_label, args.sequence_length)


if __name__ == "__main__":
    main()

"""Collect labelled static frames or dynamic landmark sequences."""
import argparse
import csv
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from utils import extract_features, FEATURE_SIZE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "dynamic"], required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--samples", type=int, default=150, help="Number of static frames or dynamic sequences")
    ap.add_argument("--sequence-length", type=int, default=30)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-quality", type=float, default=0.45)
    args = ap.parse_args()
    label = args.label.strip().upper().replace(" ", "_")
    if not label:
        raise SystemExit("Label cannot be empty.")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1.")
    holistic = mp.solutions.holistic
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "static.csv"
    saved = 0
    sequence = []
    try:
        with holistic.Holistic(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
            while saved < args.samples:
                ok, frame = cap.read()
                if not ok:
                    print("Webcam frame unavailable.")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = model.process(rgb)
                features, quality = extract_features(result)
                cv2.putText(frame, f"{args.mode.upper()} | {label} | saved {saved}/{args.samples}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30,220,30), 2)
                status = "SPACE: save" if quality >= args.min_quality else "Show hands/body clearly; low landmark quality"
                cv2.putText(frame, f"{status} (quality {quality:.2f}) | Q: quit", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,255), 2)
                cv2.imshow("Landmark collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 32:
                    if quality < args.min_quality:
                        print(f"Rejected frame: quality {quality:.2f} < {args.min_quality:.2f}")
                        continue
                    if args.mode == "static":
                        new_file = not csv_path.exists()
                        with csv_path.open("a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if new_file:
                                writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_SIZE)])
                            writer.writerow([label] + features.tolist())
                    else:
                        sequence.append(features)
                        # Collect a short burst per keypress; user presses Space to start a sequence.
                        for _ in range(args.sequence_length - 1):
                            ok2, fr2 = cap.read()
                            if not ok2:
                                break
                            fr2 = cv2.flip(fr2, 1)
                            rr = model.process(cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB))
                            ff, qq = extract_features(rr)
                            if qq < args.min_quality:
                                ff = np.zeros(FEATURE_SIZE, dtype=np.float32)
                            sequence.append(ff)
                        arr = np.asarray(sequence[-args.sequence_length:], dtype=np.float32)
                        if len(arr) < args.sequence_length:
                            arr = np.pad(arr, ((0, args.sequence_length-len(arr)), (0,0)))
                        dest = out_dir / "dynamic" / label
                        dest.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(dest / f"{saved:05d}.npz", X=arr, label=label)
                        sequence.clear()
                    saved += 1
                    print(f"Saved {saved}/{args.samples}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Collection finished. Saved {saved} samples.")

if __name__ == "__main__":
    main()

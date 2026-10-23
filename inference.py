"""Live webcam inference with static RF, optional sliding-window LSTM, TTS, STT."""
import argparse
from collections import deque
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import numpy as np
from utils import extract_features, SpeechOutput, helper_reply, listen_once, stable_prediction

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-quality", type=float, default=0.45)
    ap.add_argument("--static-threshold", type=float, default=0.60)
    ap.add_argument("--dynamic-threshold", type=float, default=0.70)
    ap.add_argument("--sequence-length", type=int, default=30)
    ap.add_argument("--no-tts", action="store_true")
    args = ap.parse_args()

    static = None
    if Path("models/static.joblib").exists():
        static = joblib.load("models/static.joblib")
    dynamic = None
    dynamic_labels = None
    if Path("models/dynamic.keras").exists() and Path("models/dynamic_labels.npy").exists():
        try:
            import tensorflow as tf
            dynamic = tf.keras.models.load_model("models/dynamic.keras")
            dynamic_labels = np.load("models/dynamic_labels.npy", allow_pickle=False)
        except Exception as exc:
            print(f"[model] Dynamic model unavailable: {exc}")

    if static is None and dynamic is None:
        raise SystemExit("No trained model found. Collect data and run train_static.py and/or train_dynamic.py.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Close other camera apps or try --camera 1.")
    # Lower resolution and request a modest frame rate to reduce CPU load.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    holistic = mp.solutions.holistic
    sequence = deque(maxlen=args.sequence_length)
    history = deque(maxlen=7)
    speech = SpeechOutput(enabled=not args.no_tts)
    current_text = ""
    dynamic_text = ""
    frame_no = 0
    try:
        with holistic.Holistic(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Could not read webcam frame.")
                    break
                frame = cv2.flip(frame, 1)
                result = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                features, quality = extract_features(result)
                frame_no += 1
                label_text = "Show hands/body clearly"
                confidence = 0.0

                if quality >= args.min_quality:
                    sequence.append(features)
                    if static is not None:
                        probabilities = static["model"].predict_proba(features.reshape(1, -1))[0]
                        best = int(np.argmax(probabilities))
                        confidence = float(probabilities[best])
                        label = str(static["encoder"].inverse_transform([best])[0])
                        if confidence >= args.static_threshold:
                            stable = stable_prediction(history, label, confidence)
                            if stable:
                                current_text = stable
                                label_text = f"Static: {stable} ({confidence:.2f})"
                            else:
                                label_text = f"Static: {label}? ({confidence:.2f})"
                        else:
                            label_text = f"Uncertain ({confidence:.2f})"
                            history.clear()
                    if dynamic is not None and len(sequence) == args.sequence_length and frame_no % 3 == 0:
                        seq = np.asarray(sequence, dtype=np.float32)[None, ...]
                        probs = dynamic.predict(seq, verbose=0)[0]
                        idx = int(np.argmax(probs))
                        dconf = float(probs[idx])
                        if dconf >= args.dynamic_threshold:
                            dynamic_text = str(dynamic_labels[idx])
                            # Dynamic model is shown separately so users can see both model outputs.
                else:
                    history.clear()
                    sequence.append(features)
                    label_text = f"Low landmark quality ({quality:.2f})"

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 112), (20, 20, 20), -1)
                cv2.putText(frame, label_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 240, 120), 2)
                cv2.putText(frame, f"Text: {current_text or '-'}", (12, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"Dynamic: {dynamic_text or '-'} | V: voice, C: chatbot, Q: quit", (12, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
                cv2.imshow("Sign Language Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("v"):
                    heard = listen_once()
                    if heard:
                        print(f"You said: {heard}")
                        reply = helper_reply(heard)
                        print(f"Helper: {reply}")
                        speech.say(reply)
                if key == ord("c"):
                    # OpenCV key events do not provide terminal text; input() is intentionally simple.
                    try:
                        message = input("Chatbot message: ")
                        reply = helper_reply(message)
                        print(f"Helper: {reply}")
                        speech.say(reply)
                    except (EOFError, KeyboardInterrupt):
                        pass
                if current_text:
                    speech.say(current_text)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

"""Shared landmark features, audio helpers, and chatbot."""
from __future__ import annotations
import threading
from collections import deque
import numpy as np

FACE_INDICES = sorted(set([10,152,234,454,1,4,5,6,9,168,197,195,33,133,160,159,158,144,145,153,362,263,387,386,385,373,374,380,70,63,105,66,107,336,296,334,293,300,61,146,91,181,84,17,314,405,321,375,291]))
POSE_COUNT, HAND_COUNT = 33, 21
FEATURE_SIZE = (POSE_COUNT + HAND_COUNT * 2 + len(FACE_INDICES)) * 3

def _array(landmarks, count):
    out = np.zeros((count, 3), dtype=np.float32)
    if landmarks is not None:
        for i, p in enumerate(landmarks.landmark[:count]):
            out[i] = (p.x, p.y, p.z)
    return out

def extract_features(results):
    """Fixed-length xyz vector plus a heuristic visibility score."""
    pose = _array(results.pose_landmarks, POSE_COUNT)
    left = _array(results.left_hand_landmarks, HAND_COUNT)
    right = _array(results.right_hand_landmarks, HAND_COUNT)
    face = np.zeros((len(FACE_INDICES), 3), dtype=np.float32)
    if results.face_landmarks is not None:
        pts = results.face_landmarks.landmark
        for i, idx in enumerate(FACE_INDICES):
            if idx < len(pts):
                p = pts[idx]
                face[i] = (p.x, p.y, p.z)
    hands = int(results.left_hand_landmarks is not None) + int(results.right_hand_landmarks is not None)
    quality = 0.65 * hands / 2 + 0.25 * float(results.pose_landmarks is not None) + 0.10 * float(results.face_landmarks is not None)
    vector = np.concatenate([pose.ravel(), left.ravel(), right.ravel(), face.ravel()]).astype(np.float32)
    if vector.size != FEATURE_SIZE:
        raise ValueError(f"Expected {FEATURE_SIZE} features, got {vector.size}")
    return vector, float(quality)

class SpeechOutput:
    def __init__(self, enabled=True):
        self.engine = None
        self.lock = threading.Lock()
        self.last_text = ""
        if enabled:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
            except Exception as exc:
                print(f"[audio] TTS disabled: {exc}")
    def say(self, text):
        if not self.engine or not text or text == self.last_text:
            return
        self.last_text = text
        def worker():
            try:
                with self.lock:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception as exc:
                print(f"[audio] TTS error: {exc}")
        threading.Thread(target=worker, daemon=True).start()

def helper_reply(message):
    """Offline rule-based helper; this is not a general-purpose LLM."""
    text = message.strip().lower()
    if not text:
        return "Please type or say a question."
    if any(x in text for x in ("hello", "hey", "namaste")) or text == "hi":
        return "Hello! I can help you use the sign recognition demo."
    if any(x in text for x in ("start", "run", "work")):
        return "Collect labelled examples, train a model, then run inference.py."
    if any(x in text for x in ("accuracy", "wrong prediction")):
        return "Collect varied balanced examples, improve lighting, and test on signers not used for training."
    if any(x in text for x in ("supported signs", "which signs", "labels")):
        return "It recognizes only the labels you collect and train, not a universal vocabulary."
    if "thank" in text:
        return "You're welcome!"
    return "I am the offline helper, not a general AI chatbot. Ask about setup, data collection, training, accuracy, or supported signs."

def listen_once(timeout=5, phrase_time_limit=7):
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("[voice] Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        return recognizer.recognize_google(audio)
    except Exception as exc:
        print(f"[voice] Speech input unavailable: {exc}")
        return None

def stable_prediction(history, label, confidence, window=7):
    history.append((label, confidence))
    while len(history) > window:
        history.popleft()
    labels = [x[0] for x in history if x[0]]
    if not labels:
        return None
    candidate = max(set(labels), key=labels.count)
    return candidate if labels.count(candidate) >= max(3, window // 2) else None

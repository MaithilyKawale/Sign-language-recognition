# Lightweight Sign Language Recognition

A modular webcam-based prototype using MediaPipe Holistic landmarks, Random Forest for static signs, and a small LSTM for dynamic signs. Optional offline text-to-speech, speech-to-text, and a rule-based helper chatbot are included.

IMPORTANT: This is a starter implementation, not a validated communication device. Accuracy depends on collected labels, signer diversity, camera conditions, and held-out testing. It does not recognize every sign language or arbitrary sentences out of the box.

## Setup (Windows PowerShell)
Use 64-bit Python 3.10 or 3.11 in a clean virtual environment:

    git clone https://github.com/MaithilyKawale/Sign-language-recognition.git
    cd Sign-language-recognition
    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Collect static examples
Collect each label separately and consistently. Press Space to save a valid frame; Q quits.

    python data_collection.py --mode static --label HELLO --samples 150
    python data_collection.py --mode static --label THANK_YOU --samples 150
    python train_static.py

## Collect dynamic sequences
Press Space to record one sequence. Each sequence is saved under data/dynamic/LABEL.

    python data_collection.py --mode dynamic --label GOOD_MORNING --samples 40 --sequence-length 30
    python data_collection.py --mode dynamic --label HOW_ARE_YOU --samples 40 --sequence-length 30
    python train_dynamic.py

## Run

    python inference.py

Keys: Q quits, V starts voice input, C lets you type a chatbot message in the terminal. Use --no-tts to disable speech output or --camera 1 to select another camera.

## Notes
- Models are created only after you collect data and train them.
- Collect multiple sessions and backgrounds; evaluate on a separate signer/session when possible.
- Face landmarks are subsampled for speed, which can reduce accuracy for signs where facial expression matters.
- SpeechRecognition may require internet depending on the recognizer. pyttsx3 TTS is generally offline.
- The project saves landmark arrays by default, not raw video. Keep consent and privacy in mind.

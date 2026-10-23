# Sign Language Recognition: ISL + ASL

A webcam-based sign recognition prototype with MediaPipe landmarks, a Random Forest for static signs, an LSTM for dynamic signs, on-screen predictions, optional text-to-speech/speech input, and a small rule-based helper.

It supports **two data sources**:
1. **Kaggle/external datasets** imported into the project format.
2. **Your own webcam samples** collected and labelled by you.

**Important:** Indian Sign Language (ISL) and American Sign Language (ASL) are different languages. A model does not automatically understand both just because both datasets are added. Train with language-specific labels, and only combine samples when they represent the same sign/label and task. Many ISL Kaggle datasets are static alphabet images, while Google ASL Signs contains dynamic isolated signs represented as landmark sequences. The project is a learning prototype, not a validated communication device or sentence translator.

## 1. Setup (Windows PowerShell)

Use 64-bit Python 3.10 or 3.11 in a clean virtual environment:

```powershell
git clone https://github.com/MaithilyKawale/Sign-language-recognition.git
cd Sign-language-recognition
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Kaggle parquet landmark datasets, install the optional import dependencies too:

```powershell
pip install -r requirements-kaggle.txt
```

## 2. Get datasets from Kaggle

Download datasets using your Kaggle account and check their licence/terms before use. The importer does not automatically download data.

### Option A — ISL alphabet image dataset

One possible starting point is the [Indian Sign Language Alphabet Dataset](https://www.kaggle.com/datasets/rushilverma07/indian-sign-language-alphabet-dataset). Extract it so the folders are arranged like:

```text
dataset/
  A/
    image1.jpg
    image2.jpg
  B/
    image1.jpg
  ...
```

If Kaggle's zip has extra nested folders, move the class folders directly under `dataset/`. Then convert the images to MediaPipe features:

```powershell
python import_kaggle_dataset.py --source image-folders --input-dir dataset --prefix ISL --max-per-class 500
python train_static.py
```

The script adds labels such as `ISL_A` and `ISL_B` to `data/static.csv`. Image import requires MediaPipe to detect a hand/body in the image; some images may be skipped. If you want webcam examples to add to the same class, collect them using the exact matching label, e.g. `ISL_A`.

### Option B — ASL dynamic signs landmark dataset

Kaggle's [Google Isolated Sign Language Recognition dataset](https://www.kaggle.com/competitions/asl-signs/data) contains `train.csv` plus `train_landmark_files/` parquet files. It is large (tens of GB), so download only if you have enough disk space. Agree to Kaggle competition rules where required.

Place the files so the importer can access `train.csv` and the `train_landmark_files` folder. From the project root run, for a small first test:

```powershell
python import_kaggle_dataset.py --source asl-parquet --train-csv train.csv --landmarks-dir train_landmark_files --max-per-label 20 --sequence-length 30
python train_dynamic.py
```

Increase `--max-per-label` after confirming the import works. Set it to `0` to import all available clips (not recommended for the first run). The importer resamples each clip to a fixed sequence length. Kaggle's ASL labels are prefixed with `ASL_` to keep them separate from ISL labels.

## 3. Collect your own webcam dataset

You can add your own examples after importing a dataset. Match the exact label if you want to add samples to an existing class. Use the same language and same meaning/sign for that label.

### Static signs (e.g. alphabet handshapes)

```powershell
python data_collection.py --mode static --label ISL_A --samples 150
python data_collection.py --mode static --label ISL_B --samples 150
python train_static.py
```

The webcam window opens. Show the sign clearly and press **Space** to save a frame; press **Q** to quit. Repeat each class across different sessions, backgrounds, distances, and lighting. The sample count is a starting point, not a guarantee of accuracy.

If you did not import a Kaggle dataset, you can choose your own labels, for example `HELLO` and `THANK_YOU`.

### Dynamic signs (movement over time)

```powershell
python data_collection.py --mode dynamic --label ASL_HELLO --samples 40 --sequence-length 30
python data_collection.py --mode dynamic --label ASL_THANK_YOU --samples 40 --sequence-length 30
python train_dynamic.py
```

Press **Space** to record one short sequence. For Kaggle ASL classes, use the exact imported class label (e.g. `ASL_<dataset-sign-name>`) if adding webcam examples to that class. Dynamic signs need examples that show the movement; a single image cannot provide that movement information.

## 4. Run the webcam recognizer

```powershell
python inference.py
```

Keys: **Q** quits, **V** starts voice input, and **C** lets you type a helper-chatbot message in the terminal. Use `--no-tts` to disable speech output or `--camera 1` to try another camera if supported by the script.

## 5. Suggested workflow

1. Choose a task: ISL static alphabet, ISL words/phrases, or ASL isolated signs. Keep the vocabularies separate.
2. Download a compatible Kaggle dataset and read its licence.
3. Import it with `import_kaggle_dataset.py`.
4. Capture more webcam samples with the exact same labels where appropriate.
5. Train the matching model: `train_static.py` or `train_dynamic.py`.
6. Test using new sessions/signers not used during training. Review per-class precision/recall, not only overall accuracy.

## Limitations and notes

- Kaggle image folders are imported as static examples; they cannot train the LSTM dynamic-sign model.
- The Google ASL Signs landmark dataset is an isolated-sign dataset, not general ASL sentence translation.
- ISL and ASL labels/vocabularies should not be treated as interchangeable. A label such as `HELLO` must be verified to represent the same language-specific sign before combining data.
- The webcam collector stores landmarks/features rather than raw video by default.
- Face landmarks are subsampled for speed, which may reduce accuracy for signs where facial expression matters.
- SpeechRecognition may need internet depending on the recognizer. `pyttsx3` text-to-speech is generally offline.
- The helper chatbot is rule-based, not a general-purpose AI chatbot.
- Model accuracy depends on dataset quality, balanced labels, signer diversity, camera conditions, and honest held-out evaluation.

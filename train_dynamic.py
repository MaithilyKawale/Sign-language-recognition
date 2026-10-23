"""Train a compact LSTM from data/dynamic/<LABEL>/*.npz."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

def main():
    files = sorted(Path("data/dynamic").glob("*/*.npz"))
    if len(files) < 4:
        raise SystemExit("Collect dynamic sequences first with data_collection.py --mode dynamic.")
    xs, labels = [], []
    for file in files:
        with np.load(file) as d:
            xs.append(d["X"].astype(np.float32))
            labels.append(str(d["label"]))
    X = np.asarray(xs, dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    if len(encoder.classes_) < 2 or min(np.bincount(y)) < 2:
        raise SystemExit("Need at least two dynamic labels and two sequences per label.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = models.Sequential([
        layers.Input(shape=(X.shape[1], X.shape[2])),
        layers.LSTM(48, dropout=0.15),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(encoder.classes_), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")]
    model.fit(X_train, y_train, validation_split=0.2, epochs=60, batch_size=16, callbacks=callbacks, verbose=1)
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, pred, target_names=encoder.classes_, zero_division=0))
    Path("models").mkdir(exist_ok=True)
    model.save("models/dynamic.keras")
    np.save("models/dynamic_labels.npy", encoder.classes_)
    print("Saved models/dynamic.keras and models/dynamic_labels.npy")
if __name__ == "__main__":
    main()

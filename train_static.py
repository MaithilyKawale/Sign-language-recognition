"""Train a Random Forest on data/static.csv."""
import csv
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def main():
    path = Path("data/static.csv")
    if not path.exists():
        raise SystemExit("No data/static.csv. Run data_collection.py --mode static first.")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise SystemExit("Collect at least two samples per class; more is strongly recommended.")
    labels = np.asarray([r[0] for r in rows[1:]])
    X = np.asarray([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    counts = np.bincount(y)
    if len(encoder.classes_) < 2 or counts.min() < 2:
        raise SystemExit("Need at least two classes and at least two examples per class.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=250, max_depth=24, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"Random split accuracy (not a guarantee of real-world performance): {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=encoder.classes_, zero_division=0))
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": clf, "encoder": encoder}, "models/static.joblib", compress=3)
    print("Saved models/static.joblib")
if __name__ == "__main__":
    main()

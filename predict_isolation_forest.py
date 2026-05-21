import numpy as np
import pandas as pd
import joblib
import os

THRESHOLD = -0.45  # seuil retenu dans l'analyse (70% coverage)

FEATURES = [
    "depth", "nst", "gap", "dmin", "rms",
    "horizontalError", "depthError",
    "lon_sin", "lon_cos", "lat_sin", "lat_cos",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_models(data_dir: str = DATA_DIR):
    iso_path = os.path.join(data_dir, "iso_forest.pkl")
    if not os.path.exists(iso_path):
        raise FileNotFoundError(
            f"{iso_path} introuvable.\n"
            "Relancez la cellule de sauvegarde dans 03_training.ipynb "
            "(joblib.dump(iso_forest, '../data/iso_forest.pkl'))."
        )
    return (
        joblib.load(iso_path),
        joblib.load(os.path.join(data_dir, "imputer.pkl")),
        joblib.load(os.path.join(data_dir, "scaler.pkl")),
    )


def predict(entry: dict, iso_forest, imputer, scaler) -> dict:
    """
    entry : dict avec les clés de FEATURES.

    Retourne :
        score      : score IsolationForest
        label      : "normal" ou "anomalie"
        confidence : 0–100 %
    """
    df = pd.DataFrame([[entry.get(f, np.nan) for f in FEATURES]], columns=FEATURES)

    X = pd.DataFrame(scaler.transform(imputer.transform(df)), columns=FEATURES)
    score = float(iso_forest.score_samples(X)[0])

    # Mapping linéaire :  score=0 → 100 %,  score=threshold → 50 %,  score=2×threshold → 0 %
    confidence = float(np.clip((score - 2 * THRESHOLD) / (-2 * THRESHOLD) * 100, 0, 100))
    label = "normal" if score >= THRESHOLD else "anomalie"

    return {
        "score": round(score, 4),
        "label": label,
        "confidence": round(confidence, 1),
    }


if __name__ == "__main__":
    iso_forest, imputer, scaler = load_models()

    entry = {
        "depth":           35.0,
        "nst":             80.0,
        "gap":             45.0,
        "dmin":             1.5,
        "rms":              0.9,
        "horizontalError":  7.0,
        "depthError":       2.5,
        "lon_sin":          0.80,
        "lon_cos":         -0.60,
        "lat_sin":          0.13,
        "lat_cos":          0.99,
    }

    result = predict(entry, iso_forest, imputer, scaler)

    print(f"Score d'isolement : {result['score']}")
    print(f"Label             : {result['label']}")
    print(f"Confiance         : {result['confidence']} %")

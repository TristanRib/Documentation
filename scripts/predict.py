import sys
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_models(data_dir: str = DATA_DIR):
    iso_path = os.path.join(data_dir, "iso_forest.pkl")

    return (
        joblib.load(iso_path),
        joblib.load(os.path.join(data_dir, "xgb_model.pkl")),
        joblib.load(os.path.join(data_dir, "imputer.pkl")),
        joblib.load(os.path.join(data_dir, "scaler.pkl")),
    )


def predict_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    df : DataFrame avec les colonnes de FEATURES.

    Retourne le DataFrame d'origine enrichi de quatre colonnes :
        mag_prediction : magnitude prédite par XGBoost
        iso_score      : score brut IsolationForest (plus proche de 0 = plus normal)
        iso_label      : "normal" ou "anomalie"
        iso_confidence : 0–100 %
    """
    iso_forest, xgb_model, imputer, scaler = load_models()

    X_scaled = pd.DataFrame(
        scaler.transform(imputer.transform(df.reindex(columns=FEATURES))),
        columns=FEATURES,
        index=df.index,
    )

    scores = iso_forest.score_samples(X_scaled)

    result = df.copy()
    result["mag_prediction"] = np.round(xgb_model.predict(X_scaled), 4)
    result["iso_score"]      = np.round(scores, 4)
    result["iso_label"]      = np.where(scores >= THRESHOLD, "normal", "anomalie")
    result["iso_confidence"] = np.round(
        np.clip((scores - 2 * THRESHOLD) / (-2 * THRESHOLD) * 100, 0, 100), 1
    )

    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "transformed_data.csv")
    df = pd.read_csv(path)

    result = predict_dataset(df)

    print(result[["mag_prediction", "iso_score", "iso_label", "iso_confidence"]].to_string())
    print(f"\n{(result['iso_label'] == 'normal').sum()} normaux / "
          f"{(result['iso_label'] == 'anomalie').sum()} anomalies sur {len(result)} observations")

"""Tests du modèle de prédiction de magnitude — cf. PLAN_DE_TEST.md.

Lancer : pytest -v
Filtrer : pytest -m slice | -m robustness | -m perf
"""

import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from scripts.predict import predict_dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "transformed_data.csv")
RANDOM_STATE = 42


@pytest.fixture(scope="module")
def df_full():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def df_test(df_full):
    _, test = train_test_split(df_full, test_size=0.2, random_state=RANDOM_STATE)
    return test.reset_index(drop=True)


@pytest.fixture(scope="module", params=["xgb", "rf"])
def model_name(request):
    return request.param


@pytest.fixture(scope="module")
def preds(df_test, model_name):
    return predict_dataset(df_test, model_name=model_name)


def _mae(y_true, y_pred, mask=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return float(np.mean(np.abs(y_true - y_pred)))


def _latlon_degrees(df):
    lat = np.degrees(np.arctan2(df["lat_sin"].values, df["lat_cos"].values))
    lon = np.degrees(np.arctan2(df["lon_sin"].values, df["lon_cos"].values))
    return lat, lon


# ============================================================
# UC01 — Prédiction des gros séismes (mag >= 7)
# ============================================================
@pytest.mark.slice
def test_uc01_gros_seismes(df_test, preds):
    """Cas critique (alerte, évacuation) : MAE doit rester sous 0.60 sur mag>=7."""
    mask = (df_test["mag"] >= 7).values
    assert mask.sum() > 0, "Aucun séisme mag>=7 dans le test set"
    mae = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask)
    assert mae < 0.60, f"MAE gros séismes : {mae:.4f} (attendu < 0.60)"


# ============================================================
# UC02 — Prédiction des séismes modérés (5 <= mag < 6)
# ============================================================
@pytest.mark.slice
def test_uc02_seismes_moderes(df_test, preds):
    """Classe majoritaire : MAE doit être proche de la MAE globale (< 0.50)."""
    mask = ((df_test["mag"] >= 5) & (df_test["mag"] < 6)).values
    assert mask.sum() > 0, "Aucun séisme 5<=mag<6 dans le test set"
    mae = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask)
    assert mae < 0.50, f"MAE séismes modérés : {mae:.4f} (attendu < 0.50)"


# ============================================================
# UC03 — Zones très documentées (Pacifique / Japon)
# ============================================================
@pytest.mark.slice
def test_uc03_zone_documentee(df_test, preds):
    """Zone dense en données : MAE doit être meilleure que la globale (< 0.45)."""
    lat, lon = _latlon_degrees(df_test)
    mask = (lat >= -40) & (lat <= 50) & (lon >= 120) & (lon <= 180)
    assert mask.sum() > 0, "Aucun séisme dans la zone Pacifique/Japon"
    mae = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask)
    assert mae < 0.45, f"MAE zone Pacifique : {mae:.4f} (attendu < 0.45)"


# ============================================================
# UC04 — Zones peu documentées (Afrique de l'Est)
# ============================================================
@pytest.mark.slice
def test_uc04_zone_peu_documentee(df_test, preds):
    """Biais géographique attendu : MAE peut se dégrader mais doit rester < 0.70."""
    lat, lon = _latlon_degrees(df_test)
    mask = (lat >= -30) & (lat <= 20) & (lon >= 25) & (lon <= 50)
    if mask.sum() < 30:
        pytest.skip(f"Pas assez d'échantillons dans la zone ({mask.sum()})")
    mae = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask)
    assert mae < 0.70, f"MAE zone peu documentée : {mae:.4f} (attendu < 0.70)"


# ============================================================
# UC05 — Données récentes (>= 2000) vs anciennes (< 1950)
# ============================================================
@pytest.mark.slice
def test_uc05_dates(df_test, preds):
    """Récent doit être bien prédit ; ancien peut se dégrader mais doit rester pire que récent."""
    mask_recent = (df_test["year"] >= 2000).values
    mask_old = (df_test["year"] < 1950).values
    assert mask_recent.sum() > 0 and mask_old.sum() > 0, "Échantillons insuffisants"

    mae_recent = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask_recent)
    mae_old = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask_old)

    assert mae_recent < 0.50, f"MAE post-2000 trop élevée : {mae_recent:.4f} (attendu < 0.50)"
    assert mae_old > mae_recent, (
        f"Les séismes anciens ne sont pas moins bien prédits que les récents "
        f"(MAE old={mae_old:.4f}, MAE recent={mae_recent:.4f})"
    )


# ============================================================
# UC06 — Prédiction avec peu de stations (nst < 10)
# ============================================================
@pytest.mark.robustness
def test_uc06_peu_de_stations(df_test, preds):
    """Input dégradé : la MAE doit rester acceptable (< 0.55)."""
    mask = (df_test["nst"] < 10).values
    if mask.sum() < 30:
        pytest.skip(f"Pas assez d'échantillons nst<10 ({mask.sum()})")
    mae = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask)
    assert mae < 0.55, f"MAE nst<10 : {mae:.4f} (attendu < 0.55)"


# ============================================================
# UC07 — Score d'anomalie (Isolation Forest)
# ============================================================
@pytest.mark.robustness
def test_uc07_score_anomalie(df_test, preds):
    """Les entrées flaguées 'anomalie' doivent être moins bien prédites que les 'normal'."""
    mask_anom = (preds["iso_label"] == "anomalie").values
    mask_norm = (preds["iso_label"] == "normal").values
    assert mask_anom.sum() > 0, "Aucune anomalie détectée"
    mae_anom = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask_anom)
    mae_norm = _mae(df_test["mag"].values, preds["mag_prediction"].values, mask_norm)
    assert mae_anom > mae_norm, (
        f"L'Isolation Forest ne discrimine pas : "
        f"MAE anomalies={mae_anom:.4f}, MAE normales={mae_norm:.4f}"
    )


# ============================================================
# UC08 — Résistance au bruit sur lat / lon
# ============================================================
@pytest.mark.robustness
def test_uc08_bruit_latlon(df_test, preds, model_name):
    """Un bruit de ~1 km sur lat/lon ne doit pas dégrader la MAE de plus de 0.05."""
    rng = np.random.default_rng(RANDOM_STATE)
    lat, lon = _latlon_degrees(df_test)
    lat_noisy = lat + rng.normal(0, 0.01, len(lat))
    lon_noisy = lon + rng.normal(0, 0.01, len(lon))

    df_noisy = df_test.copy()
    df_noisy["lat_sin"] = np.sin(np.radians(lat_noisy))
    df_noisy["lat_cos"] = np.cos(np.radians(lat_noisy))
    df_noisy["lon_sin"] = np.sin(np.radians(lon_noisy))
    df_noisy["lon_cos"] = np.cos(np.radians(lon_noisy))

    preds_noisy = predict_dataset(df_noisy, model_name=model_name)
    mae_clean = _mae(df_test["mag"].values, preds["mag_prediction"].values)
    mae_noisy = _mae(df_test["mag"].values, preds_noisy["mag_prediction"].values)
    delta = mae_noisy - mae_clean
    assert delta < 0.05, (
        f"Dégradation excessive sous bruit : MAE clean={mae_clean:.4f}, "
        f"MAE bruitée={mae_noisy:.4f}, delta={delta:.4f}"
    )


# ============================================================
# UC09 — Imputation : masquer 20 % de dmin et rms
# ============================================================
@pytest.mark.robustness
def test_uc09_imputation(df_test, model_name):
    """Avec 20 % de masquage sur dmin/rms, la MAE doit rester sous 0.55."""
    rng = np.random.default_rng(RANDOM_STATE)
    df_masked = df_test.copy()
    for col in ("dmin", "rms"):
        idx = rng.choice(len(df_masked), size=int(0.2 * len(df_masked)), replace=False)
        df_masked.loc[df_masked.index[idx], col] = np.nan

    preds_masked = predict_dataset(df_masked, model_name=model_name)
    mae = _mae(df_test["mag"].values, preds_masked["mag_prediction"].values)
    assert mae < 0.55, f"MAE après masquage 20 % : {mae:.4f} (attendu < 0.55)"


# ============================================================
# UC10 — Tenue en charge
# ============================================================
@pytest.mark.perf
def test_uc10_tenue_en_charge(df_full, model_name):
    """100k observations doivent être prédites en < 30 s et < 500 MB de pic RAM."""
    tracemalloc.start()
    start = time.time()
    _ = predict_dataset(df_full, model_name=model_name)
    duration = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024**2

    assert duration < 30, f"Temps de prédiction trop long : {duration:.2f}s"
    assert peak_mb < 500, f"Pic RAM trop élevé : {peak_mb:.1f} MB"

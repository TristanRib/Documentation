"""Analyse de robustesse multi-pays (slide 56 du cours).

Pipeline India (artefacts produits par train_india.py) :
  - imputer_india.pkl          : SimpleImputer(median), fit train India
  - scaler_india.pkl           : StandardScaler, fit train India
  - isolation_forest_india.pkl : IsolationForest, fit X_train India normalise
  - model_india.keras          : MLP dense 64-32-16-1

Pour chaque pays (India test, USA, UK, Canada, Australia) on calcule :
  1. Le score d'anomalie via l'IF India (RMSE vs Coverage, methode du coude)
  2. La resistance a l'imputation par mediane India (degradation RMSE %)
  3. La resistance au bruit gaussien (variation RMSE %)

Tout est execute sans reentrainer aucun composant.
"""

import os
os.environ["KERAS_BACKEND"]      = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

SEED = 42
np.random.seed(SEED)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONTINUOUS_COLS = ["serving_size_g", "protein_g", "total_fat_g",
                   "total_carbs_g", "sodium_mg", "sugars_g", "avg_rating"]
NOISE_LEVELS    = np.array([1, 3, 5, 10, 15, 20])
N_REPEATS       = 3                   # 3 tirages suffisent pour stabiliser
COUNTRIES       = ["USA", "UK", "Canada", "Australia"]
COVERAGE_TARGET = 70
IMPUTATION_BUCKETS = 20               # ~20 points sur l'axe x au lieu de 100


def fast_predict(X):
    """Inference rapide sans l'overhead de model.predict()."""
    import tensorflow as tf
    return model(tf.convert_to_tensor(X, dtype=tf.float32),
                 training=False).numpy().flatten()


# =============================================================================
# 1. Chargement des artefacts India
# =============================================================================
print("Chargement des artefacts India...")
imputer    = joblib.load(os.path.join(BASE_DIR, "imputer_india.pkl"))
scaler     = joblib.load(os.path.join(BASE_DIR, "scaler_india.pkl"))
iso_forest = joblib.load(os.path.join(BASE_DIR, "isolation_forest_india.pkl"))
model      = keras.models.load_model(os.path.join(BASE_DIR, "model_india.keras"))
print("  imputer, scaler, iso_forest, model_india.keras OK")

# India : on reproduit exactement le meme split que train_india.py
data_india = pd.read_csv(os.path.join(DATA_DIR, "data_india.csv"))
X_india    = data_india.drop(columns=["price_usd_normalized", "calories"])
y_india    = data_india["price_usd_normalized"].values
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_india, y_india, test_size=0.2, random_state=SEED
)

X_test_i_scaled = scaler.transform(imputer.transform(X_test_i))
y_pred_i        = fast_predict(X_test_i_scaled)
rmse_india      = float(np.sqrt(mean_squared_error(y_test_i, y_pred_i)))
print(f"RMSE India (test) : {rmse_india:.4f}")


# =============================================================================
# 2. Fonctions helper
# =============================================================================

def isolation_test(X_scaled, y_true, y_pred):
    """Methode du coude (slide 25) : RMSE vs Coverage en balayant le seuil IF.

    Retourne (thresholds, coverages%, rmses, best_thr, best_rmse).
    best_thr = seuil le plus strict avec coverage >= COVERAGE_TARGET et RMSE min.
    """
    scores     = iso_forest.score_samples(X_scaled)
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    coverages, rmses = [], []
    n = len(y_true)
    for t in thresholds:
        mask = scores >= t
        coverages.append(100 * mask.sum() / n)
        if mask.sum() > 0:
            rmses.append(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        else:
            rmses.append(np.nan)

    best_thr, best_rmse = None, np.inf
    for t, cov, r in zip(thresholds, coverages, rmses):
        if cov >= COVERAGE_TARGET and not np.isnan(r) and r < best_rmse:
            best_rmse = r
            best_thr  = t

    return (np.asarray(thresholds), np.asarray(coverages),
            np.asarray(rmses), best_thr, best_rmse)


def imputation_test(X_raw, y_true, rmse_base, seed=SEED):
    """Slide 29 : degradation RMSE en fonction du nombre de NaN imputes.

    Pour chaque feature continue, on retire de 1% a 100% des valeurs (MCAR),
    on les remplit avec l'imputer India (median), et on mesure la degradation
    relative en RMSE. Moyenne sur N_REPEATS tirages.
    """
    rng         = np.random.default_rng(seed)
    n_total     = X_raw.shape[0]
    range_scale = max(1, n_total // IMPUTATION_BUCKETS)
    n_range     = list(range(range_scale, n_total, range_scale))
    degradation = {f: [] for f in CONTINUOUS_COLS}

    for feat in CONTINUOUS_COLS:
        feat_idx = list(X_raw.columns).index(feat)
        for n_missing in n_range:
            runs = []
            for _ in range(N_REPEATS):
                X_c = X_raw.copy()
                idx = rng.choice(n_total, n_missing, replace=False)
                X_c.iloc[idx, feat_idx] = np.nan
                X_imp = scaler.transform(imputer.transform(X_c))
                y_imp = fast_predict(X_imp)
                runs.append(np.sqrt(mean_squared_error(y_true, y_imp)))
            degradation[feat].append(
                (np.mean(runs) - rmse_base) / rmse_base * 100
            )
    return n_range, degradation


def noise_test(X_raw, y_true, rmse_base, seed=SEED):
    """Slides 14-15 : bruit gaussien proportionnel a l'ecart-type de la feature.

    Niveaux : 1, 3, 5, 10, 15, 20 % de std. Variation relative de RMSE (%).
    """
    X_base    = imputer.transform(X_raw)
    variation = {f: [] for f in CONTINUOUS_COLS}

    for feat in CONTINUOUS_COLS:
        feat_idx = list(X_raw.columns).index(feat)
        std_dev  = X_base[:, feat_idx].std()
        for lvl in NOISE_LEVELS:
            rng  = np.random.default_rng(seed + int(lvl))
            runs = []
            for _ in range(N_REPEATS):
                X_n = X_base.copy()
                X_n[:, feat_idx] += rng.normal(0, std_dev * lvl / 100, len(X_n))
                X_n_scaled = scaler.transform(X_n)
                y_n = fast_predict(X_n_scaled)
                runs.append(np.sqrt(mean_squared_error(y_true, y_n)))
            variation[feat].append(100 * (np.mean(runs) - rmse_base) / rmse_base)
    return variation


# =============================================================================
# 3. Analyse India (sur X_test)
# =============================================================================
print("\n--- India (test set, N=%d) ---" % len(y_test_i))
thr_i, cov_i, rmse_i, best_thr_i, best_rmse_i = isolation_test(
    X_test_i_scaled, y_test_i, y_pred_i
)
print(f"  Seuil IF retenu (cov >= {COVERAGE_TARGET}%) : "
      f"{best_thr_i:.4f}  RMSE={best_rmse_i:.4f}")

n_range_i, degradation_i = imputation_test(X_test_i, y_test_i, rmse_india)
variation_i              = noise_test(X_test_i, y_test_i, rmse_india)

results = {
    "India": {
        "n": len(y_test_i), "rmse_base": rmse_india,
        "thr": thr_i, "cov": cov_i, "rmses": rmse_i,
        "best_thr": best_thr_i, "best_rmse": best_rmse_i,
        "n_range": n_range_i, "degradation": degradation_i,
        "variation": variation_i,
    }
}


# =============================================================================
# 4. Analyse autres pays (dataset complet, sans reentrainement)
# =============================================================================
for country in COUNTRIES:
    print(f"\n--- {country} (dataset complet) ---")
    data_c = pd.read_csv(os.path.join(DATA_DIR, f"data_{country.lower()}.csv"))
    X_c    = data_c.drop(columns=["price_usd_normalized", "calories"])
    y_c    = data_c["price_usd_normalized"].values

    X_c_scaled = scaler.transform(imputer.transform(X_c))
    y_pred_c   = fast_predict(X_c_scaled)
    rmse_c     = float(np.sqrt(mean_squared_error(y_c, y_pred_c)))
    print(f"  N={len(y_c)}  RMSE base = {rmse_c:.4f}")

    thr_c, cov_c, rmse_cv, best_thr_c, best_rmse_c = isolation_test(
        X_c_scaled, y_c, y_pred_c
    )
    print(f"  Seuil IF retenu (cov >= {COVERAGE_TARGET}%) : "
          f"{best_thr_c:.4f}  RMSE={best_rmse_c:.4f}")

    n_range_c, degradation_c = imputation_test(X_c, y_c, rmse_c)
    variation_c              = noise_test(X_c, y_c, rmse_c)

    results[country] = {
        "n": len(y_c), "rmse_base": rmse_c,
        "thr": thr_c, "cov": cov_c, "rmses": rmse_cv,
        "best_thr": best_thr_c, "best_rmse": best_rmse_c,
        "n_range": n_range_c, "degradation": degradation_c,
        "variation": variation_c,
    }


# =============================================================================
# 5. Figures
# =============================================================================
print("\nGeneration des figures...")
ALL_COUNTRIES = ["India"] + COUNTRIES

# Figure 1 : RMSE vs Coverage (methode du coude)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, country in enumerate(ALL_COUNTRIES):
    r   = results[country]
    ax1 = axes[i]
    ax2 = ax1.twinx()
    ax1.plot(r["thr"], r["rmses"], color="tab:blue", lw=2, label="RMSE")
    ax2.plot(r["thr"], r["cov"],   color="tab:red", ls="--", lw=2, label="Coverage %")
    if r["best_thr"] is not None:
        ax1.axvline(r["best_thr"], color="green", lw=1.5, ls=":",
                    label=f"Seuil retenu : {r['best_thr']:.3f}")
    ax1.set_xlabel("Seuil de score d'isolement (plus eleve = plus strict)")
    ax1.set_ylabel("RMSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylabel("Coverage (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.axhline(COVERAGE_TARGET, color="gray", ls=":", lw=1)
    ax1.set_title(f"{country}  (N={r['n']}, RMSE base={r['rmse_base']:.4f})")
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
axes[-1].set_visible(False)
fig.suptitle("Analyse de robustesse : RMSE vs Coverage (Isolation Forest India)", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "isolation_rmse_coverage.png"), dpi=150)
plt.close(fig)
print("  isolation_rmse_coverage.png")

# Figure 2 : Impact imputation
fig, axes = plt.subplots(1, len(ALL_COUNTRIES), figsize=(20, 5), sharey=True)
for ax, country in zip(axes, ALL_COUNTRIES):
    r = results[country]
    x_pct = [100 * n / r["n"] for n in r["n_range"]]
    for feat in CONTINUOUS_COLS:
        ax.plot(x_pct, r["degradation"][feat], label=feat, lw=1.2)
    ax.set_xlabel("% de valeurs manquantes")
    ax.set_title(country)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Degradation RMSE (%)")
fig.suptitle("Impact de l'imputation par mediane India sur la degradation RMSE", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "imputation_degradation.png"), dpi=150)
plt.close(fig)
print("  imputation_degradation.png")

# Figure 3 : Test de bruit
fig, axes = plt.subplots(1, len(ALL_COUNTRIES), figsize=(20, 5), sharey=True)
for ax, country in zip(axes, ALL_COUNTRIES):
    r = results[country]
    for feat in CONTINUOUS_COLS:
        ax.plot(NOISE_LEVELS, r["variation"][feat], "o-", label=feat, lw=1.2)
    ax.set_xlabel("Niveau de bruit (% de l'ecart-type)")
    ax.set_title(country)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Variation RMSE (%)")
fig.suptitle("Test de robustesse au bruit gaussien (variation RMSE)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "noise_variation.png"), dpi=150)
plt.close(fig)
print("  noise_variation.png")


# =============================================================================
# 6. Tableau de synthese
# =============================================================================
print(f"\n{'='*70}")
print("SYNTHESE DES ZONES DE ROBUSTESSE")
print(f"{'='*70}")
print(f"{'Pays':<12} {'N':>5} {'RMSE base':>10} {'Seuil IF':>10} "
      f"{'RMSE @70%':>10} {'Gain RMSE':>10}")
print("-" * 70)
for country in ALL_COUNTRIES:
    r    = results[country]
    thr  = f"{r['best_thr']:.4f}" if r["best_thr"] is not None else "N/A"
    gain = (r["rmse_base"] - r["best_rmse"]) / r["rmse_base"] * 100
    print(f"{country:<12} {r['n']:>5} {r['rmse_base']:>10.4f} "
          f"{thr:>10} {r['best_rmse']:>10.4f} {gain:>9.1f}%")



# =============================================================================
# 7. Sensibilites extremes (a 20% bruit / 100% NaN) imprimees a l'ecran
# =============================================================================
print("\n" + "=" * 80)
print("FEATURES LES PLUS SENSIBLES")
print("=" * 80)
print(f"{'Pays':<12} {'Bruit max @20%':<35} {'Imputation max @100%':<35}")
print("-" * 80)
for country in ALL_COUNTRIES:
    v = results[country]["variation"]
    d = results[country]["degradation"]
    nf = max(v, key=lambda f: v[f][-1])
    df = max(d, key=lambda f: d[f][-1])
    print(f"{country:<12} "
          f"{nf + f' ({v[nf][-1]:+.1f}%)':<35} "
          f"{df + f' ({d[df][-1]:+.1f}%)':<35}")

print(f"\nFigures dans : {RESULTS_DIR}")
print("Termine.")

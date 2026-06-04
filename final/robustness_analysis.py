import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import joblib
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model      = keras.models.load_model('model_india.keras')
imputer    = joblib.load('imputer_india.pkl')
scaler     = joblib.load('scaler_india.pkl')
iso_forest = joblib.load('isolation_forest_india.pkl')

os.makedirs('results', exist_ok=True)

CONTINUOUS   = ['serving_size_g', 'protein_g', 'total_fat_g',
                'total_carbs_g', 'sodium_mg', 'sugars_g', 'avg_rating']
NOISE_LEVELS = [1, 3, 5, 10, 15, 20]


def predict(X_scaled):
    return model.predict(X_scaled, verbose=0).flatten()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

data_india = pd.read_csv('data/data_india.csv')
X_india    = data_india.drop(columns=['price_usd_normalized', 'calories'])
y_india    = data_india['price_usd_normalized'].values
_, X_india_test, _, y_india_test = train_test_split(X_india, y_india, test_size=0.2, random_state=42)

datasets = {
    'India':     (X_india_test, y_india_test),
    'USA':       (lambda d: (d.drop(columns=['price_usd_normalized','calories']), d['price_usd_normalized'].values))(pd.read_csv('data/data_usa.csv')),
    'UK':        (lambda d: (d.drop(columns=['price_usd_normalized','calories']), d['price_usd_normalized'].values))(pd.read_csv('data/data_uk.csv')),
    'Canada':    (lambda d: (d.drop(columns=['price_usd_normalized','calories']), d['price_usd_normalized'].values))(pd.read_csv('data/data_canada.csv')),
    'Australia': (lambda d: (d.drop(columns=['price_usd_normalized','calories']), d['price_usd_normalized'].values))(pd.read_csv('data/data_australia.csv')),
}

# Prédictions avec les artefacts India
baseline = {}
for name, (X_raw, y_true) in datasets.items():
    X_scaled      = scaler.transform(imputer.transform(X_raw))
    y_pred        = predict(X_scaled)
    rmse_base     = rmse(y_true, y_pred)
    baseline[name] = {'X_raw': X_raw, 'y': y_true,
                      'X_scaled': X_scaled, 'y_pred': y_pred, 'rmse_base': rmse_base}
    print(f"{name:10s}  N={len(y_true):5d}  RMSE={rmse_base:.4f}")


# Scores d'anomalie : RMSE vs Coverage
fig, axes = plt.subplots(1, len(baseline), figsize=(20, 4))

for ax, (name, r) in zip(axes, baseline.items()):
    scores     = iso_forest.score_samples(r['X_scaled'])
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    coverages, rmses = [], []
    for t in thresholds:
        mask = scores >= t
        coverages.append(100 * mask.sum() / len(r['y']))
        rmses.append(rmse(r['y'][mask], r['y_pred'][mask]) if mask.sum() > 0 else np.nan)

    ax2 = ax.twinx()
    ax.plot(thresholds, rmses, color='tab:blue', lw=2, label='RMSE')
    ax2.plot(thresholds, coverages, color='tab:red', ls='--', lw=2, label='Coverage %')
    ax.set_title(f'{name}  (RMSE={r["rmse_base"]:.3f})')
    ax.set_xlabel("Score d'isolement")
    ax.set_ylabel('RMSE', color='tab:blue')
    ax2.set_ylabel('Coverage (%)', color='tab:red')
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)

fig.suptitle("Scores d'anomalie : RMSE vs Coverage (IF entraîné sur India)")
plt.tight_layout()
fig.savefig('results/anomaly_scores.png', dpi=150)
plt.close(fig)
print("-> results/anomaly_scores.png")


# Robustesse à l'imputation
fig, axes = plt.subplots(1, len(baseline), figsize=(20, 4), sharey=True)
rng = np.random.default_rng(42)

for ax, (name, r) in zip(axes, baseline.items()):
    X_raw  = r['X_raw']
    y_true = r['y']
    n      = X_raw.shape[0]
    step   = max(1, n // 20)
    n_range = list(range(step, n, step))
    x_pct   = [100 * k / n for k in n_range]

    for feat in CONTINUOUS:
        if feat not in X_raw.columns:
            continue
        feat_idx     = list(X_raw.columns).index(feat)
        degradations = []
        for n_missing in n_range:
            runs = []
            for _ in range(3):
                X_c = X_raw.copy()
                idx = rng.choice(n, n_missing, replace=False)
                X_c.iloc[idx, feat_idx] = np.nan
                y_imp = predict(scaler.transform(imputer.transform(X_c)))
                runs.append(rmse(y_true, y_imp))
            degradations.append((np.mean(runs) - r['rmse_base']) / r['rmse_base'] * 100)
        ax.plot(x_pct, degradations, label=feat, lw=1.2)

    ax.set_title(name)
    ax.set_xlabel('% valeurs manquantes')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Dégradation RMSE (%)')
fig.suptitle("Robustesse à l'imputation par médiane (India)")
plt.tight_layout()
fig.savefig('results/imputation.png', dpi=150)
plt.close(fig)
print("-> results/imputation.png")


# Robustesse au bruit 
fig, axes = plt.subplots(1, len(baseline), figsize=(20, 4), sharey=True)

for ax, (name, r) in zip(axes, baseline.items()):
    X_imp  = imputer.transform(r['X_raw'])  # espace impute, avant scale
    y_true = r['y']

    for feat in CONTINUOUS:
        if feat not in r['X_raw'].columns:
            continue
        feat_idx = list(r['X_raw'].columns).index(feat)
        std_dev  = X_imp[:, feat_idx].std()
        variations = []
        for lvl in NOISE_LEVELS:
            runs = []
            for seed in range(3):
                rng2 = np.random.default_rng(42 + seed + int(lvl * 100))
                X_n = X_imp.copy()
                X_n[:, feat_idx] += rng2.normal(0, std_dev * lvl / 100, len(X_n))
                y_n = predict(scaler.transform(X_n))
                runs.append(rmse(y_true, y_n))
            variations.append(100 * (np.mean(runs) - r['rmse_base']) / r['rmse_base'])
        ax.plot(NOISE_LEVELS, variations, 'o-', label=feat, lw=1.2)

    ax.set_title(name)
    ax.set_xlabel("Niveau de bruit (% écart-type)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Variation RMSE (%)')
fig.suptitle("Robustesse au bruit gaussien (IF et artefacts India)")
plt.tight_layout()
fig.savefig('results/noise.png', dpi=150)
plt.close(fig)
print("-> results/noise.png")
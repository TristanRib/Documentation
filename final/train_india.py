"""Entrainement du modele Keras + Isolation Forest sur le dataset India.

Genere les artefacts charges par robustness_analysis.py :
  - imputer_india.pkl       (SimpleImputer, strategy=median)
  - scaler_india.pkl        (StandardScaler, fit sur le train uniquement)
  - model_india.keras       (MLP dense 64-32-16-1)
  - isolation_forest_india.pkl
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Chargement et split
data = pd.read_csv(os.path.join(DATA_DIR, "data_india.csv"))
X = data.drop(columns=["price_usd_normalized", "calories"])
y = data["price_usd_normalized"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# 2. Imputer (fit sur train uniquement) puis scaler (fit sur train)
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

# 3. Modele Keras
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=30, batch_size=32, verbose=2,
)

# 4. Isolation Forest fit sur les donnees train normalisees (slide 37)
iso_forest = IsolationForest(random_state=SEED)
iso_forest.fit(X_train_scaled)

# 5. Export des artefacts
joblib.dump(imputer,    os.path.join(BASE_DIR, "imputer_india.pkl"))
joblib.dump(scaler,     os.path.join(BASE_DIR, "scaler_india.pkl"))
joblib.dump(iso_forest, os.path.join(BASE_DIR, "isolation_forest_india.pkl"))
model.save(os.path.join(BASE_DIR, "model_india.keras"))
print("Artefacts exportes dans", BASE_DIR)

# 6. Figures de diagnostic
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"],     label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_title("Loss (MSE)"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history["mae"],     label="Train")
axes[1].plot(history.history["val_mae"], label="Val")
axes[1].set_title("MAE"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle("Convergence du modele Keras (India)")
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "training_convergence.png"), dpi=120)
plt.close(fig)

corr = data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Matrice de correlation (India)")
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "correlation_india.png"), dpi=120)
plt.close(fig)

y_pred = model.predict(X_test_scaled, verbose=0).flatten()
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        "k--", lw=1, label="y=x")
ax.set_title("Predictions vs True Values (India - test)")
ax.set_xlabel("True"); ax.set_ylabel("Predicted")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "pred_vs_true_india.png"), dpi=120)
plt.close(fig)

print("Figures: results/training_convergence.png, correlation_india.png, pred_vs_true_india.png")

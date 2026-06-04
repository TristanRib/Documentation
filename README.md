# Analyse de robustesse — Prédiction de prix alimentaires

| Busin Thomas | Riboulet-Depret Tristan | Van-Duysen Nicolas |
|--------------|-------------------------|--------------------|

## Vue d'ensemble

On prédit le prix normalisé de plats de restaurant (`price_usd_normalized`) à partir de leurs valeurs nutritionnelles, en partant du dataset Inde. L'objectif du sujet final est de tester la robustesse de ce pipeline sur des données d'autres pays, sans jamais réentraîner quoi que ce soit.

Concrètement : on entraîne tout sur l'Inde, on exporte le modèle + les artefacts, et on les applique tels quels sur USA, UK, Canada et Australie pour voir ce qui tient et ce qui casse.

---

## Dataset

5 fichiers CSV, un par pays : `data/data_india.csv`, `data_usa.csv`, `data_uk.csv`, `data_canada.csv`, `data_australia.csv`.

Les features utilisées sont les valeurs nutritionnelles (`serving_size_g`, `protein_g`, `total_fat_g`, `total_carbs_g`, `sodium_mg`, `sugars_g`, `avg_rating`) ainsi que des indicateurs catégoriels (type de plat, gamme de prix). La cible est `price_usd_normalized`.

| Pays      | N observations |
|-----------|---------------|
| India     | 2 196 (train) |
| USA       | 2 721         |
| UK        | 1 053         |
| Canada    | 901           |
| Australia | 629           |

---

## Démarche

### 1. Entraînement sur l'Inde (`train_india.py`)

On entraîne sur l'Inde uniquement :
- `SimpleImputer(strategy='median')` pour les valeurs manquantes
- `StandardScaler` pour la normalisation
- Un MLP Keras (64 → 32 → 16 → 1) sur 30 epochs
- Une `IsolationForest` pour détecter les anomalies

Les 4 artefacts sont exportés : `model_india.keras`, `scaler_india.pkl`, `imputer_india.pkl`, `isolation_forest_india.pkl`.

### 2. Analyse de robustesse (`robustness_analysis.py`)

Pour chaque pays, on applique le même pipeline (sans rien refitter) et on mesure 3 choses :

**Scores d'anomalie** — on balaie le seuil de l'Isolation Forest et on trace la RMSE vs Coverage. La zone de robustesse se lit sur le graphe : c'est le plateau stable avant que la RMSE ne monte.

**Résistance à l'imputation** — on retire aléatoirement entre 0 et 100 % des valeurs d'une feature, on laisse l'imputer médiane India les remplacer, et on mesure la dégradation de RMSE. Répété 3 fois pour stabiliser.

**Résistance au bruit** — on ajoute un bruit gaussien proportionnel à l'écart-type de chaque feature (de 1 % à 20 %), puis on scale et on prédit. Même logique.

---

## Résultats & Analyse

### RMSE de base par pays

| Pays      | RMSE  |
|-----------|-------|
| India     | 1.81  |
| USA       | 8.69  |
| UK        | 10.12 |
| Canada    | 7.16  |
| Australia | 8.07  |

La RMSE explose dès qu'on sort de l'Inde. C'est attendu : le modèle a appris la relation macronutriments → prix *indien*. Les prix américains ou britanniques répondent à une logique économique complètement différente (coût de la vie, marges, TVA). Les macros d'un burger restent comparables partout, mais son prix en dollars ne suit pas la même distribution selon le pays.

### Scores d'anomalie

Les courbes RMSE/Coverage ont une forme similaire pour tous les pays. La zone de robustesse (le plateau) se situe autour de -0.52 sur l'axe du score d'isolement pour tous les pays. Ça veut dire que l'IF formée sur l'Inde détecte de la même manière les points atypiques dans les autres pays : elle repère les observations qui s'écartent le plus de la distribution indienne, qui sont précisément celles où le modèle extrapole le plus et fait le plus d'erreurs.

Filtrer les 28-30 % d'observations les plus "anomales" réduit la RMSE d'environ 30 % pour tous les pays. Le mécanisme est en partie mécanique (on retire les pires cas), mais il valide l'approche : l'IF sert bien de garde-fou utile en production.

### Résistance au bruit

India montre une variation jusqu'à ~1.35 % sur `protein_g` à 20 % de bruit. Les autres pays restent sous 0.1 %. Ce n'est pas qu'ils sont plus robustes — c'est que leur RMSE de base est déjà énorme (~8-10), donc le bruit ajouté représente une variation absolue similaire mais relative négligeable. La seule mesure fiable est celle sur l'Inde : le modèle est très robuste au bruit, moins de 1.5 % de dégradation même à 20 % de bruit sur les features les plus sensibles.

### Résistance à l'imputation

Sur l'Inde, `sodium_mg` est la feature la plus critique : +8 % de dégradation quand 100 % des valeurs sont remplacées par la médiane. C'est logique — le sodium varie énormément selon le type de plat (dessert vs curry vs fast food), donc remplacer par la médiane efface une information très discriminante pour le prix.

Sur les autres pays, on observe des dégradations négatives (la RMSE *baisse* avec l'imputation). Même explication que pour le bruit : la RMSE de base est dominée par le distribution shift, et la médiane indienne "recentre" légèrement les features vers la distribution d'entraînement, ce qui peut améliorer mécaniquement la prédiction.

### Conclusion opérationnelle

Le pipeline India ne se transfère pas tel quel pour prédire des prix dans d'autres pays. RMSE × 4 à 5 dès qu'on sort du marché d'entraînement. Pour un déploiement multi-marché il faudrait soit un fine-tuning par pays, soit ajouter le pays comme feature explicite, soit post-scaler les sorties du modèle par pays.

En revanche, l'Isolation Forest reste utile partout comme filtre de confiance : on peut s'en servir pour signaler les entrées trop éloignées de la distribution d'entraînement avant de renvoyer une prédiction.

---

## Structure du projet

```
final/
├── train_india.py            # Entraînement complet sur l'Inde + export des artefacts
├── robustness_analysis.py    # Analyse de robustesse sur tous les pays
├── data/
│   ├── data_india.csv
│   ├── data_usa.csv
│   ├── data_uk.csv
│   ├── data_canada.csv
│   └── data_australia.csv
├── results/
│   ├── anomaly_scores.png    # RMSE vs Coverage par pays
│   ├── imputation.png        # Dégradation RMSE par feature corrompue
│   └── noise.png             # Variation RMSE par niveau de bruit
├── model_india.keras
├── scaler_india.pkl
├── imputer_india.pkl
└── isolation_forest_india.pkl
```

---

## Usage

```bash
cd final/

# 1. Entraîner le modèle et exporter les artefacts
python train_india.py

# 2. Lancer l'analyse de robustesse (génère les 3 figures dans results/)
python robustness_analysis.py
```

**Dépendances** : `tensorflow`, `keras`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib`

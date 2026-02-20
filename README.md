# Seismic Activities
| Busin Thomas | Riboulet-Depret Tristan | Van-Duysen Nicolas |
|--------------|-------------------------|--------------------|

## Overview

Ce projet utilise les données du dataset **200 Years of Global Major Earthquakes (1826–2026)** provenant de **Kaggle**, qui compile des informations sur des séismes majeurs enregistrés à travers le monde sur deux siècles.

L’objectif est d’explorer, analyser et construire un modèle prédictif à partir de ces données historiques, afin de mieux comprendre les tendances sismiques globales.

---

## Dataset

**Source** : https://www.kaggle.com/datasets/dhrubangtalukdar/200-years-of-global-major-earthquakes-18262026

**Description** : Ce dataset répertorie des séismes majeurs (généralement d’un seuil significatif de magnitude ou impact) sur une période de 1826 à 2026. Les variables typiques incluent des informations temporelles, géographiques et sismologiques pour chaque événement.

Voir la [datacard](./data/raw_data.yaml) pour la description des données.

---

## Project Dependencies

| Catégorie              | Librairie         | Utilisation                                           |
|------------------------|-------------------|-------------------------------------------------------|
| Manipulation de données | pandas            | Chargement, nettoyage et transformation des données   |
| Calcul numérique        | numpy             | Opérations numériques et manipulation de tableaux     |
| Visualisation           | matplotlib        | Graphiques et visualisations                          |
| Visualisation           | seaborn           | Visualisation statistique                             |
| Machine Learning        | scikit-learn      | Prétraitement, training & évaluation de modèles       |
| Machine Learning        | xgboost           | Modèle de gradient boosting                           |
| Accès données           | kagglehub         | Téléchargement depuis Kaggle                          |
| Configuration           | yaml (PyYAML)     | Chargement de configurations                          |
| Standard Library        | datetime          | Gestion des dates et heures                           |

---

## Project Structure

### [`/notebooks`](./notebooks)

Notebooks Jupyter organisés par phase du pipeline :
- **Exploration** : analyse exploratoire des séismes, distributions, patterns temporels et spatiaux.
- **Transformation** : prétraitement des variables, traitement des valeurs manquantes, feature engineering.
- **Training** : entraînement de modèles, validation croisée, comparaisons de performance.

### [`/models`](./models)

Contient des *model cards* documentant chaque modèle :
- Type et architecture du modèle
- Hyperparamètres
- Performance (mae, mse, r2, etc.)
- Conclusions et recommandations

### [`/data`](./data)

Contient :
- *Data cards* décrivant les datasets utilisés
- [transformed_data.csv](./data/transformed_data.csv) : version transformée prête à l’usage pour entraînement

---

## Methodology

1. Import et analyse exploratoire des données (EDA)
2. Feature engineering (extraction de tendances temporelles, clustering géographique)
3. Sélection et entraînement de modèles de prédiction
4. Évaluation des métriques (MAE, RMSE, R²)

---

## Usage

1. Cloner le dépôt
2. Télécharger le dataset via Kaggle
3. Exécuter les notebooks dans l’ordre :
   - [01_exploration.ipynb](./notebooks/01_exploration.ipynb)
   - [02_transformation.ipynb](./notebooks/02_transformation.ipynb)
   - [03_training.ipynb](./notebooks/03_training.ipynb)

---

## Reproducibility

- Version Python recommandée : `>=3.9`
- Seed fixée pour la reproductibilité des splits

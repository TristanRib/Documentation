# Seismic Activities – Kaggle

| Busin Thomas | Riboulet-Depret Tristan | Van-Duysen Nicolas |
|--------------|-------------------------|--------------------|

## Description

Ce projet vise à analyser et modéliser des données d’activités sismiques afin de construire un modèle prédictif performant.
Le workflow couvre l’exploration des données, leur transformation, puis l’entraînement et l’évaluation de modèles.

---

## Structure du projet

### [/notebook](/notebooks)

Contient les notebooks Jupyter organisés par étape du pipeline :

- **Exploration** : analyse exploratoire des données (EDA), visualisations, compréhension des variables.
- **Transformation** : nettoyage, feature engineering, encodage, normalisation.
- **Training** : entraînement des modèles, validation, évaluation des performances.

---

### [/models](/models)

Contient les **model cards** décrivant :

- Le type de modèle utilisé  
- Les hyperparamètres principaux  
- Les métriques de performance  
- Les hypothèses et limites  

---

### [/data](/data)

Contient :

- **Data cards** : description des datasets utilisés (source, variables, format...).
- `transformed_data.csv` : dataset après preprocessing, utilisé pour l’entraînement.
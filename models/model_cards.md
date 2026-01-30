# Model Card Magnitude - XGBRegressor

## 1 - Détails du modèle

- Ce modèle a été développé par Thomas Busin, Riboulet--Depret Tristan et Van-Duysen Nicolas, étudiants en Mastère en Intelligence Artificielle à EPSI Lille.
- La dernière version de ce modèle a été publiée en janvier 2026.
- Le modèle a été intialisé depuis la librairie xgboost avec un random_state à 42.
- L'architecture du modèle est donc celle proposée de base par la librairie xgboost.

## 2 - Usages prévus

- Déterminer la magnitude d'un séisme depuis les informations obtenues par une station de sismologie (date, latitude, longitude, profondeur...).

## 3 - Données

- Les colonnes numériques ont toutes étés normalisées grâce au StandardScaler de la librairie scikit-learn.
  ![la liste des colonnes numériques](ressources/colonnes_numeriques.png)
- Toutes les données nulles se sont vues assigner des valeurs.
- La catégorie "magType" a été encodée pour être interprétable par le modèle.

## 4 - Entraînement

- Le dataset d'entraînement a été généré grâce à la fonction train_test_split avec une proportion de dataset d'entraînement de 0.8 et un random_state à 42.
- Le dataset d'entraînement contient donc 40 features et 84861 observations
- Les hyperparamètres n'ayant que très peu d'impact sur les métriques, les hyperparamètres de base seront utilisés pour l'entraînement du modèle

## 5 - Métriques

- Les métriques observées sont la Mean Absolute Error, Mean Squared Error et la R² (coefficient de détermination)
  ![MAE, RMSE, R²](ressources/metriques_mag.png)

# Model Card Depth - XGBRegressor

## 1 - Détails du modèle

- Ce modèle a été développé par Thomas Busin, Riboulet--Depret Tristan et Van-Duysen Nicolas, étudiants en Mastère en Intelligence Artificielle à EPSI Lille.
- La dernière version de ce modèle a été publiée en janvier 2026.
- Le modèle a été intialisé depuis la librairie xgboost avec un random_state à 42.
- L'architecture du modèle est donc celle proposée de base par la librairie xgboost.

## 2 - Usages prévus

- Déterminer la profondeur d'un séisme depuis les informations obtenues par une station de sismologie (date, latitude, longitude, profondeur...).

## 3 - Données

- Les colonnes numériques ont toutes étés normalisées grâce au StandardScaler de la librairie scikit-learn.
  ![la liste des colonnes numériques](ressources/colonnes_numeriques.png)
- Toutes les données nulles se sont vues assigner des valeurs.
- La catégorie "magType" a été encodée pour être interprétable par le modèle.

## 4 - Entraînement

- Le dataset d'entraînement a été généré grâce à la fonction train_test_split avec une proportion de dataset d'entraînement de 0.8 et un random_state à 42.
- Le dataset d'entraînement contient donc 40 features et 84861 observations
- Les hyperparamètres n'ayant que très peu d'impact sur les métriques, les hyperparamètres de base seront utilisés pour l'entraînement du modèle

## 5 - Métriques

- Les métriques observées sont la Mean Absolute Error, Mean Squared Error et la R² (coefficient de détermination)
  ![MAE, RMSE, R²](ressources/metriques_depth.png)


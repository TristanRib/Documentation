# Model Card Depth - XGBRegressor

## 1 - Détails du modèle

- Ce modèle a été développé par Thomas Busin, Riboulet--Depret Tristan et Van-Duysen Nicolas, étudiants en Mastère en Intelligence Artificielle à EPSI Lille.
- La dernière version de ce modèle a été publiée en janvier 2026.
- Le modèle a été intialisé depuis la librairie xgboost avec un random_state à 42.
- L'architecture du modèle est donc celle proposée de base par la librairie xgboost.

## 2 - Usages prévus

- Déterminer la profondeur d'un séisme depuis les informations obtenues par une station de sismologie (date, latitude, longitude, profondeur...).

## 3 - Biais, risques et limitations

### Biais

- Biais d'historicité : Les instruments de mesure d'il y a 200 étaient bien moins performants que ceux de nos jours, des séismes de même magnitude peuvent avoir été observés différemment d'une période à l'autre
- Biais humain : Les mesures dépendent d'humains qui pour des raisons politiques, idéologiques... ont pu taire ou communiquer des séismes avec des données erronnées
- Biais géographique : Certaines zones géographiques sismiques peuvent être mal évaluées par manque de moyens
- Biais statistique : L'auteur du dataset a pu limiter la quantité de séismes lors de la création du dataset et donc en omettre certains

### Risques

- En manquant de données dans certaines zones, on peut sous-estimer la profondeur d'un séisme dans une zone jusqu'alors peu risquée
- Se fier aveuglément à ce modèle de prédiction peut engendrer soit des mesures coûteuses et inutiles, soit ignorer certaines catastrophes qui pourraient coûter des vies

### Limites

- Limitation du dataset : Ce dataset ne comptabilise que les séismes d'une magnitude supérieure ou égale à 5, ce qui fait que le modèle ne peut prédire efficacement seulement les profondeurs de séismes de magnitudes comprises entre 5 et 10

## 4 - Données

- Les colonnes numériques ont toutes étés normalisées grâce au [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) de la librairie scikit-learn. Cette méthode de normalisation permet à toutes les variables d'être évaluées sur la même échelle et de réduire les écarts entre les features tout en respectant celui-ci
  Features numériques utilisées pour le modèle : ['latitude', 'longitude', 'depth', 'mag', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst', 'date_ts_ms'
- Toutes les données nulles de la seule catégorie en contenant ('place') se sont vues assigner des valeurs ('Inconnu').
- La catégorie "magType" a été encodée pour être interprétable par le modèle via la méthode [get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) de pandas.

## 5 - Entraînement

- Le dataset d'entraînement a été généré grâce à la fonction train_test_split avec une proportion de dataset d'entraînement de 0.8 et un random_state à 42.
- Le dataset d'entraînement contient donc 40 features et 84861 observations
- Les hyperparamètres n'ayant que très peu d'impact sur les métriques, les hyperparamètres de base seront utilisés pour l'entraînement du modèle

## 6 - Métriques

- Les métriques observées sont la Mean Absolute Error, Mean Squared Error et la R² (coefficient de détermination)
- MAE : 0.2139
- RMSE : 0.4051
- R² : 0.8384
- Avec ces métriques, on remarque que le coefficient de détermination est très élevé et la MAE très faible. On peut donc en déduire que le modèle est très performant mais que malheureusement il contient certaines erreures importantes comme l'indique la différence entre la MAE et la RMSE.

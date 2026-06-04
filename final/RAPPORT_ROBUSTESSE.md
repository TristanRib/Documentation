# Rapport de robustesse - Sujet final (slide 56)

Run : `robustness_analysis.py` | Niveaux de bruit : [1, 3, 5, 10, 15, 20]% | Répétitions : 3 | Cible coverage : 70%

## 0. Démarche

On a :

1. Entraîné une Isolation Forest sur le dataset India (`isolation_forest_india.pkl`), avec le `SimpleImputer` (médiane) et le `StandardScaler` aussi fittés sur India uniquement.
2. Choisi une zone de robustesse par la méthode du coude : on balaie le seuil de score IsolationForest et on retient le seuil le plus strict qui conserve au moins 70 % des observations, en réduisant la RMSE sur ce sous-ensemble.
3. Importé les datasets des quatre autres pays (USA, UK, Canada, Australie) et fait passer dans le même pipeline (imputer + scaler + isolation forest + modèle MLP), sans réentraîner.
4. Pour chaque pays, mesuré :
   - La résistance au bruit : bruit gaussien proportionnel à l'écart-type de chaque feature, niveaux de 1 à 20 %.
   - La résistance à l'imputation : retrait MCAR de 1 à 100 % des valeurs, ré-imputation par la médiane de l'Inde.
   - L'analyse des scores d'anomalie : la distribution des scores et aussi la zone de robustesse via méthode du coude.

Toutes les figures correspondantes sont dans le dossier `results/`.

---

## 1. Synthèse par pays

| Pays      |    N | RMSE base | Seuil IF | RMSE @70% |  Gain |
| --------- | ---: | --------: | -------: | --------: | ----: |
| India     |  440 |    1.8003 |  -0.5256 |    1.2552 | 30.3% |
| USA       | 2721 |    8.5264 |  -0.5243 |    5.9197 | 30.6% |
| UK        | 1053 |    9.9653 |  -0.5287 |    6.8789 | 31.0% |
| Canada    |  901 |    7.0113 |  -0.5263 |    5.1212 | 27.0% |
| Australia |  629 |    7.9274 |  -0.5227 |    5.6325 | 28.9% |

### Pourquoi le seuil IF retenu est-il quasi-identique entre les pays ?

L'Isolation Forest a été calibrée sur l'Inde : elle produit donc des scores plus extrêmes sur les autres pays (plus d'"anomalies"), mais la forme générale de la distribution des scores reste plutôt égale. Le seuil défini par coverage >= 70 % tombe donc à peu près au même endroit sur l'axe des scores pour tous les pays. Cela signifie aussi que l'IF agit comme un détecteur indirect de distribution shift : sur USA/UK/Canada/Australie, les points classés "anormaux" sont ceux qui s'écartent le plus de la distribution indienne qui sont effectivement ceux où le MLP extrapole le plus.

### Pourquoi le Gain en RMSE est-il constantentre les pays ?

On a fixé un coverage de 70 %, donc on coupe systématiquement les 30 % d'observations les plus atypiques. Le gain de environ 30 % est en grande partie mécanique.

## 2. Sensibilités

| Pays      | Bruit max (20%)          | Imputation max (100% NaN) |
| --------- | ------------------------ | ------------------------- |
| India     | `protein_g` (+1.3%)      | `sodium_mg` (+8.0%)       |
| USA       | `serving_size_g` (+0.0%) | `avg_rating` (+0.9%)      |
| UK        | `protein_g` (+0.1%)      | `sugars_g` (+0.7%)        |
| Canada    | `protein_g` (+0.1%)      | `sugars_g` (+0.6%)        |
| Australia | `serving_size_g` (+0.0%) | `sugars_g` (+0.4%)        |

## 3. Variation RMSE (%) par niveau de bruit

| Pays / Feature               |    1% |    3% |    5% |   10% |   15% |   20% |
| ---------------------------- | ----: | ----: | ----: | ----: | ----: | ----: |
| India / `serving_size_g`     | +0.01 | -0.04 | -0.02 | +0.04 | -0.13 | +0.19 |
| India / `protein_g`          | -0.01 | -0.07 | +0.01 | +0.27 | +0.06 | +1.35 |
| India / `total_fat_g`        | +0.00 | +0.03 | -0.01 | +0.02 | +0.05 | +0.17 |
| India / `total_carbs_g`      | -0.01 | -0.01 | +0.03 | -0.05 | -0.04 | +0.14 |
| India / `sodium_mg`          | -0.01 | -0.02 | +0.02 | +0.22 | +0.07 | +1.08 |
| India / `sugars_g`           | -0.01 | -0.03 | +0.02 | -0.12 | -0.04 | +0.18 |
| India / `avg_rating`         | -0.01 | -0.05 | -0.00 | +0.13 | -0.21 | +0.34 |
| USA / `serving_size_g`       | -0.00 | +0.00 | +0.00 | +0.01 | -0.00 | +0.02 |
| USA / `protein_g`            | -0.00 | -0.00 | +0.01 | +0.00 | +0.01 | -0.00 |
| USA / `total_fat_g`          | -0.00 | -0.00 | -0.01 | +0.01 | +0.00 | -0.03 |
| USA / `total_carbs_g`        | -0.00 | +0.00 | +0.00 | -0.01 | -0.02 | +0.00 |
| USA / `sodium_mg`            | -0.00 | -0.00 | +0.01 | +0.01 | +0.01 | +0.01 |
| USA / `sugars_g`             | -0.00 | +0.00 | -0.00 | -0.02 | -0.00 | -0.01 |
| USA / `avg_rating`           | -0.00 | +0.00 | +0.01 | +0.00 | -0.01 | -0.02 |
| UK / `serving_size_g`        | +0.00 | -0.00 | +0.01 | +0.02 | +0.01 | +0.06 |
| UK / `protein_g`             | -0.00 | -0.01 | +0.01 | +0.03 | +0.07 | +0.06 |
| UK / `total_fat_g`           | -0.00 | +0.00 | +0.00 | -0.00 | -0.02 | -0.01 |
| UK / `total_carbs_g`         | -0.00 | +0.00 | -0.00 | +0.00 | +0.01 | -0.00 |
| UK / `sodium_mg`             | -0.00 | -0.00 | +0.02 | +0.01 | +0.05 | +0.04 |
| UK / `sugars_g`              | -0.00 | -0.00 | +0.00 | +0.01 | +0.02 | -0.01 |
| UK / `avg_rating`            | +0.00 | +0.00 | -0.00 | -0.00 | -0.01 | +0.02 |
| Canada / `serving_size_g`    | +0.00 | -0.00 | -0.02 | +0.03 | +0.00 | +0.05 |
| Canada / `protein_g`         | +0.00 | +0.01 | -0.01 | +0.01 | +0.04 | +0.14 |
| Canada / `total_fat_g`       | -0.00 | +0.00 | -0.01 | +0.01 | -0.00 | +0.01 |
| Canada / `total_carbs_g`     | -0.00 | +0.01 | +0.00 | -0.00 | -0.01 | +0.02 |
| Canada / `sodium_mg`         | +0.00 | +0.01 | -0.02 | +0.01 | +0.00 | +0.11 |
| Canada / `sugars_g`          | -0.00 | +0.00 | +0.00 | -0.01 | +0.01 | -0.00 |
| Canada / `avg_rating`        | +0.00 | +0.00 | -0.01 | +0.00 | -0.02 | +0.04 |
| Australia / `serving_size_g` | -0.00 | +0.00 | -0.01 | +0.02 | +0.01 | +0.03 |
| Australia / `protein_g`      | -0.00 | -0.00 | -0.01 | +0.04 | +0.12 | +0.01 |
| Australia / `total_fat_g`    | +0.00 | -0.00 | -0.01 | +0.00 | +0.03 | -0.05 |
| Australia / `total_carbs_g`  | +0.00 | -0.00 | +0.01 | -0.00 | -0.04 | +0.01 |
| Australia / `sodium_mg`      | +0.00 | +0.00 | -0.01 | +0.03 | +0.09 | +0.01 |
| Australia / `sugars_g`       | -0.00 | -0.01 | -0.01 | -0.00 | +0.02 | -0.00 |
| Australia / `avg_rating`     | -0.00 | -0.00 | -0.01 | -0.01 | -0.01 | -0.02 |

### Pourquoi un tel écart de RMSE entre India et les autres pays ?

La RMSE de base passe de 1.80 pour l'Inde à entre 7 et 10 pour les autres pays. Cet écart traduit un distribution shift sur la cible : les cinq pays sont extraits du même CSV mondial, donc leurs macronutriments restent comparables (un steak fait globalement les mêmes calories partout), mais `price_usd_normalized` reflète des réalités économiques très différentes (coût de la vie, marges des chaînes, TVA). Le MLP a appris la relation macros => prix sur le marché indien et il n'a aucune raison de produire le bon prix américain ou britannique.

## 4. Dégradation RMSE (%) par taux d'imputation

| Pays / Feature               |   10% |   25% |   50% |   75% |  100% |
| ---------------------------- | ----: | ----: | ----: | ----: | ----: |
| India / `serving_size_g`     | +0.40 | +1.13 | +2.01 | +2.80 | +3.94 |
| India / `protein_g`          | +0.54 | +0.57 | +0.82 | +1.00 | +1.71 |
| India / `total_fat_g`        | +0.20 | -0.10 | -0.09 | +0.03 | +0.11 |
| India / `total_carbs_g`      | -0.42 | +0.31 | -0.54 | -0.51 | -0.67 |
| India / `sodium_mg`          | +0.82 | +1.65 | +5.34 | +7.78 | +7.98 |
| India / `sugars_g`           | -0.44 | +0.31 | +0.61 | +0.47 | +0.99 |
| India / `avg_rating`         | -0.22 | -0.50 | +0.20 | +1.21 | +0.71 |
| USA / `serving_size_g`       | -0.03 | -0.12 | -0.15 | -0.35 | -0.46 |
| USA / `protein_g`            | -0.19 | -0.40 | -0.73 | -1.21 | -1.63 |
| USA / `total_fat_g`          | -0.02 | -0.07 | -0.15 | -0.23 | -0.28 |
| USA / `total_carbs_g`        | +0.02 | -0.02 | +0.04 | +0.10 | +0.06 |
| USA / `sodium_mg`            | -0.21 | -0.44 | -0.90 | -1.51 | -1.92 |
| USA / `sugars_g`             | +0.02 | +0.11 | +0.16 | +0.27 | +0.41 |
| USA / `avg_rating`           | +0.10 | +0.20 | +0.42 | +0.70 | +0.91 |
| UK / `serving_size_g`        | -0.05 | -0.07 | -0.15 | -0.15 | -0.29 |
| UK / `protein_g`             | -0.23 | -0.43 | -0.72 | -1.08 | -1.65 |
| UK / `total_fat_g`           | +0.01 | -0.05 | -0.09 | -0.15 | -0.25 |
| UK / `total_carbs_g`         | -0.07 | -0.07 | -0.06 | -0.06 | -0.16 |
| UK / `sodium_mg`             | -0.15 | -0.39 | -0.83 | -1.34 | -1.68 |
| UK / `sugars_g`              | +0.07 | +0.22 | +0.37 | +0.63 | +0.73 |
| UK / `avg_rating`            | +0.06 | +0.09 | +0.20 | +0.31 | +0.42 |
| Canada / `serving_size_g`    | -0.04 | -0.30 | -0.39 | -0.43 | -0.76 |
| Canada / `protein_g`         | -0.21 | -0.49 | -1.04 | -1.72 | -2.26 |
| Canada / `total_fat_g`       | -0.06 | -0.20 | -0.35 | -0.62 | -0.70 |
| Canada / `total_carbs_g`     | +0.06 | +0.04 | +0.20 | +0.27 | +0.30 |
| Canada / `sodium_mg`         | -0.10 | -0.38 | -1.13 | -1.61 | -2.09 |
| Canada / `sugars_g`          | +0.04 | +0.19 | +0.17 | +0.54 | +0.59 |
| Canada / `avg_rating`        | +0.14 | +0.08 | +0.19 | +0.28 | +0.48 |
| Australia / `serving_size_g` | -0.06 | -0.07 | -0.27 | -0.11 | -0.34 |
| Australia / `protein_g`      | -0.32 | -0.15 | -0.91 | -1.39 | -1.81 |
| Australia / `total_fat_g`    | -0.04 | -0.18 | -0.31 | -0.26 | -0.44 |
| Australia / `total_carbs_g`  | -0.03 | -0.08 | -0.15 | -0.09 | -0.27 |
| Australia / `sodium_mg`      | -0.22 | -0.45 | -0.82 | -1.74 | -2.03 |
| Australia / `sugars_g`       | -0.00 | +0.16 | +0.23 | +0.32 | +0.43 |
| Australia / `avg_rating`     | -0.04 | +0.08 | +0.04 | +0.11 | +0.12 |

## 5. Figures

- `results/isolation_rmse_coverage.png` - RMSE vs Coverage par pays (méthode du coude)
- `results/imputation_degradation.png` - dégradation RMSE vs % NaN imputés
- `results/noise_variation.png` - variation RMSE vs niveau de bruit

---

## 6. Analyse

### Pourquoi les % de sensibilité au bruit / imputation sont-ils quasi nuls sur les pays non-India ?

Ces pourcentages sont relatifs à la RMSE de base. Sur USA/UK/Canada/Australie, la RMSE de base est déjà énorme à cause du shift. La variation absolue de RMSE causée par le bruit reste comparable à India, mais divisée par 9 elle devient invisible. La seule vraie mesure de sensibilité est celle sur India (RMSE_base saine de 1.80), où on voit que :

- Le modèle est très robuste au bruit (=< 1.35 % max à 20 % de bruit).
- Le modèle est plus sensible à l'imputation : `sodium_mg` dégrade de +8.0 % quand 100 % des valeurs sont remplacées par la médiane.

### Pourquoi les courbes d'Australie sont les plus bruitées ?

Avec seulement 629 observations, l'estimation de la RMSE est moins stable.

---

## 7. Conclusion

Pipeline India non transférable telle quelle car une RMSE x 4-5 sur les autres pays. Un fine-tuning par pays (ou un encodage explicite du pays en feature, ou un post-scaling du prix) est nécessaire pour un déploiement

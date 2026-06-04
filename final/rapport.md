# Rapport de robustesse

Pour chaque pays on réalise trois analyses :
- **Scores d'anomalie** : on balaie le seuil de l'IF et on trace RMSE vs Coverage pour lire la zone de robustesse.
- **Résistance à l'imputation** : on corrompt aléatoirement les features continues (0 à 100% de valeurs manquantes), on les réimpute par la médiane India, et on mesure la dégradation de RMSE.
- **Résistance au bruit** : on ajoute un bruit gaussien proportionnel à l'écart-type de chaque feature (1% à 20%) et on mesure la variation de RMSE.

---

## Scores d'anomalie

*Voir `results/anomaly_scores.png`*

**India** — La courbe descend régulièrement de 1.80 vers ~1.0 puis remonte à droite. Le plateau stable se situe entre -0.55 et -0.50. On retient **-0.52** comme seuil de robustesse : en dessous, les observations sont trop atypiques pour que le modèle soit fiable. À ce seuil : coverage ~72%, RMSE ~1.26.

**USA** — Forme similaire à India mais sur une RMSE de base à 8.69. Le plateau est visible entre -0.55 et -0.50, zone de robustesse retenue : **-0.52**. Coverage ~72%, RMSE ~5.92.

**UK** — RMSE de base la plus élevée (10.12). La courbe a la même silhouette. Zone de robustesse : **-0.53**. Coverage ~69%, RMSE ~6.88.

**Canada** — RMSE de base 7.16, courbe la plus "propre" visuellement. Zone de robustesse : **-0.53**. Coverage ~70%, RMSE ~5.12.

**Australia** — Seulement 629 observations, les courbes sont plus bruitées (variance d'échantillonnage). La tendance reste la même. Zone de robustesse : **-0.52**. Coverage ~72%, RMSE ~5.63.

---

## Résistance à l'imputation

*Voir `results/imputation.png`*

**India** — C'est le seul pays où la dégradation est lisible et significative. `sodium_mg` est la feature critique : +8% de RMSE quand 100% des valeurs sont remplacées par la médiane. `serving_size_g` suit avec +3.9%. Les autres features sont peu ou pas impactées.

**USA, UK, Canada, Australia** — La dégradation relative est quasi nulle voire négative. La RMSE est autour de 8-10 à cause du distribution shift (voir section Analyse), donc une variation absolue de quelques dixièmes devient invisible en relatif. Certaines features montrent même une dégradation négative car la médiane India "recentre" les valeurs vers la distribution d'entraînement, ce qui améliore  la prédiction dans certains cas.

---

## Résistance au bruit

*Voir `results/noise.png`*

**India** — Le modèle résiste bien au bruit faible. En dessous de 10% d'écart-type, la variation de RMSE reste sous 0.3% pour toutes les features. À 20%, `protein_g` monte à +1.35% et `sodium_mg` à +1.08% — les features les plus sensibles sont aussi les plus importantes pour la prédiction du prix.

**USA, UK, Canada, Australia** — Même raisonnement que pour l'imputation. Les variations relatives sont proches de 0 (<0.1%) sur tous les pays. La mesure n'est pas significative sur une base RMSE dégradée par le shift.

---

## Analyse des raisons

### Pourquoi la RMSE explose sur les autres pays ?

Le modèle a appris la relation macronutriments → prix indien. Les macros d'un plat restent comparables d'un pays à l'autre (un burger fait globalement les mêmes calories), mais `price_usd_normalized` suit des réalités économiques locales très différentes : coût de la vie, marges, TVA. Le modèle ne peut pas produire le bon prix américain ou britannique, il n'a jamais été exposé à ces distributions.

### Pourquoi le seuil IF est le même partout ?

L'IF est calibrée sur India. Sur les autres pays elle voit globalement plus d'"anomalies", mais la forme de la distribution des scores reste proche. Le "coude" de la courbe RMSE/Coverage tombe au même endroit pour tous les pays. Ce qui est utile : l'IF joue le rôle de détecteur de distribution shift, les points qu'elle écarte en dessous du seuil sont précisément ceux qui s'écartent le plus de la distribution indienne, et donc ceux où le MLP extrapole le plus.

### Pourquoi sodium_mg est la feature la plus fragile à l'imputation sur India ?

Le sodium varie énormément selon le type de plat : un dessert n'a pas le même sodium qu'un plat frit. Remplacer par la médiane efface cette information discriminante, et comme le prix est lié au type de plat, la prédiction se dégrade.

### Pourquoi les courbes Australia sont-elles plus bruitées ?

629 observations seulement, la RMSE estimée sur un sous-ensemble de ce dataset est naturellement moins stable. Les oscillations visibles sur les graphes sont de la variance d'échantillonnage, pas du comportement du modèle.

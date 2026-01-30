# Datacard

Nom du dataset : 200 Years of Global Major Earthquakes

Version : 1

Domaine : Science (sismologie)

Provenance : Kaggle Dataset qui a tiré ses données du U.S. Geological Survey (USGS) Earthquake Catalog

Date de création : 08 janvier 2026

Usage prévu : Prédiction des séismes (date, lieu, magnitude)

Méthode de collecte : Données historiques & données

Période couverte : 1826-2026

## Dataset snapshot

| **Catégorie** | **Donnée** |
| --- | --- |
| Taille du dataset | 17.86 MB |
| Nombre d'instance | 106077 |
| Nombre de feature | 19  |

Notes

- Les données dupliquées ont été supprimées
- La donnée est triée par ordre chronologique
- Les premières archives historiques peuvent présenter une incertitude plus élevée en raison d'instruments de mesure limités.

## Lien dataset

<https://www.kaggle.com/datasets/dhrubangtalukdar/200-years-of-global-major-earthquakes-18262026/code>

## Dictionnaire des données

| **Nom**         | **Type** | **Unité**  | **Intervalle**                                   | **Description**                                                                   | **Valeurs manquantes** | **Doublons** | **Moyenne** | **Min** | **Max** |
|-----------------|----------|------------|--------------------------------------------------|-----------------------------------------------------------------------------------|------------------------|--------------|-------------|---------|---------|
| time            | date     |            | 24 septembre 1827 04h00 => 07 janvier 2026 22h00 | Date et heure du séisme.                                                          | 0                      | 1            |             |         |         |
| latitude        | float    | degrés     | \-77.080 => 87.386                               | Latitude du séisme.                                                               | 0                      | 30653        | 3.79        | -77.1   | 87.4    |
| longitude       | float    | degrés     | \-179.997 => 180.000                             | Longitude du séisme.                                                              | 0                      | 22695        | 40.3        | -180    | 180     |
| depth           | float    | kilomètres | \-4 => 700                                       | Profondeur en kilomètre du tremblement.                                           | 539                    | 94441        | 61.6        | -4      | 700     |
| mag             | float    | magnitude  | 5 => 9.5                                         | Magnitude du séisme.                                                              | 0                      | 105763       | 5.45        | 5       | 9.5     |
| magType         | object   |            |                                                  | Échelle de magnitude utilisée.                                                    | 0                      | 106049       |             |         |         |
| nst             | float    | unités     | 0 => 934                                         | Nombre de stations sismiques utilisées.                                           | 74700                  | 105339       | 158         | 0       | 360     |
| gap             | float    | degrés     | 6.5 => 360                                       | Écart azimutal indiquant la couverture de la station.                             | 64100                  | 104216       | 63.1        | 6.5     | 360     |
| dmin            | float    | degrés     | 0 => 41.046                                      | Distance minimale de la station sismique la plus proche.                          | 84700                  | 97083        | 4.24        | 0       | 41      |
| rms             | float    | unités     | \-1 => 69.320                                    | Racine carré moyenne des résidus de temps de trajet.                              | 31500                  | 105805       | 0.96        | -1      | 69.3    |
| id              | object   |            |                                                  | Identifiant unique.                                                               | 0                      | 0            |             |         |         |
| updated         | object   |            |                                                  | Date et heure de la mise à jour de la donnée.                                     | 0                      | 4274         |             |         |         |
| place           | object   |            |                                                  | Localisation (ville)                                                              | 222                    | 41814        |             |         |         |
| type            | object   |            |                                                  | Cause du séisme (tremblement de terre, explosion nucléaire, éruption volcanique…) | 0                      | 106070       |             |         |         |
| horizontalError | float    | kilomètres | 0 => 778                                         | La distance d'incertitude                                                         | 86100                  | 104721       | 7.83        | 0       | 778     |
| depthError      | float    | kilomètres | \-1 => 1091.9                                    | La profondeur d'incertitude                                                       | 51600                  | 102969       | 7.98        | -1      | 1090    |
| magError        | float    | magnitude  | 0 => 1.84                                        | La magnitude d'incertitude                                                        | 69500                  | 105737       | 0.17        | 0       | 1.84    |
| magNst          | float    | unités     | 0 => 1027                                        | Le nombre de stations sismiques qui ont évalué la magnitude                       | 63700                  | 105411       | 54.9        | 0       | 1030    |
| status          | object   |            |                                                  | Savoir si l'information a été vérifiée ou saisie automatiquement.                 | 0                      | 106075       |             |         |         |

## Matrice de corrélation

|                 | latitude | longitude | depth   | mag     | nst     | gap     | dmin    | rms     | horizontalError | depthError | magError | magNst  |
|-----------------|----------|-----------|---------|---------|---------|---------|---------|---------|-----------------|------------|----------|---------|
| latitude        | 1.000    | 0.198     | \-0.117 | 0.055   | 0.333   | 0.021   | \-0.370 | \-0.077 | \-0.069         | 0.044      | 0.172    | 0.210   |
| longitude       | 0.198    | 1.000     | \-0.083 | \-0.005 | 0.012   | \-0.185 | \-0.147 | \-0.019 | \-0.056         | 0.014      | 0.029    | 0.010   |
| depth           | \-0.117  | \-0.083   | 1.000   | \-0.016 | 0.172   | \-0.160 | \-0.077 | \-0.015 | 0.052           | \-0.024    | \-0.070  | 0.023   |
| mag             | 0.055    | \-0.005   | \-0.016 | 1.000   | 0.533   | \-0.312 | \-0.025 | 0.052   | 0.038           | 0.168      | 0.461    | \-0.009 |
| nst             | 0.333    | 0.012     | 0.172   | 0.533   | 1.000   | \-0.426 | \-0.097 | \-0.018 | \-0.117         | \-0.175    | \-0.331  | 0.412   |
| gap             | 0.021    | \-0.185   | \-0.160 | \-0.312 | \-0.426 | 1.000   | \-0.009 | 0.030   | 0.242           | 0.278      | 0.305    | \-0.091 |
| dmin            | \-0.370  | \-0.147   | \-0.077 | \-0.025 | \-0.097 | \-0.009 | 1.000   | \-0.036 | 0.302           | \-0.135    | 0.103    | \-0.052 |
| rms             | \-0.077  | \-0.019   | \-0.015 | 0.052   | \-0.018 | 0.030   | \-0.036 | 1.000   | 0.213           | 0.066      | \-0.044  | \-0.091 |
| horizontalError | \-0.069  | \-0.056   | 0.052   | 0.038   | \-0.117 | 0.242   | 0.302   | 0.213   | 1.000           | 0.277      | 0.204    | \-0.010 |
| depthError      | 0.044    | 0.014     | \-0.024 | 0.168   | \-0.175 | 0.278   | \-0.135 | 0.066   | 0.277           | 1.000      | 0.454    | \-0.143 |
| magError        | 0.172    | 0.029     | \-0.070 | 0.461   | \-0.331 | 0.305   | 0.103   | \-0.044 | 0.204           | 0.454      | 1.000    | \-0.499 |
| magNst          | 0.210    | 0.010     | 0.023   | \-0.009 | 0.412   | \-0.091 | \-0.052 | \-0.091 | \-0.010         | \-0.143    | \-0.499  | 1.000   |
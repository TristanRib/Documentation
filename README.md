200 Years of Global Major Earthquakes

Busin Thomas Riboulet-Depret Tristan Van-Duysen Nicolas

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

# Dataset Analyse

## Dictionnaire des données

| **Nom** | **Type** | **Unité** | **Intervalle** | **Description** |
| --- | --- | --- | --- | --- |
| time | date |     | 24 septembre 1827 04h00 => 07 janvier 2026 22h00 | Date et heure du séisme. |
| latitude | float | degrés | \-77.080 => 87.386 | Latitude du séisme. |
| longitude | float | degrés | \-179.997 => 180.000 | Longitude du séisme. |
| depth | float | kilomètres | \-4 => 700 | Profondeur en kilomètre du tremblement. |
| mag | float | magnitude | 5 => 9.5 | Magnitude du séisme. |
| magType | object |     |     | Échelle de magnitude utilisée. |
| nst | float | unités | 0 => 934 | Nombre de stations sismiques utilisées. |
| gap | float | degrés | 6.5 => 360 | Écart azimutal indiquant la couverture de la station. |
| dmin | float | degrés | 0 => 41.046 | Distance minimale de la station sismique la plus proche. |
| rms | float | unités | \-1 => 69.320 | Racine carré moyenne des résidus de temps de trajet. |
| id  | object |     |     | Identifiant unique. |
| updated | object |     |     | Date et heure de la mise à jour de la donnée. |
| place | object |     |     | Localisation (ville) |
| type | object |     |     | Cause du séisme (tremblement de terre, explosion nucléaire, éruption volcanique…) |
| horizontalError | float | kilomètres | 0 => 778 | La distance d'incertitude |
| depthError | float | kilomètres | \-1 => 1091.9 | La profondeur d'incertitude |
| magError | float | magnitude | 0 => 1.84 | La magnitude d'incertitude |
| magNst | float | unités | 0 => 1027 | Le nombre de stations sismiques qui ont évalué la magnitude |
| status | object |     |     | Savoir si l'information a été vérifiée ou saisie automatiquement. |

## Statistiques descriptives

## Corrélation

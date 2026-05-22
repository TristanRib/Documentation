## Plan de test

1. Prédiction des gros séismes (> 7) : C'est l'usage le plus important du modèle (prédiction pour évacuation, alertes).

2. Prédiction des séismes modérés (5 - 6) : Usage statistique pour cartographier les séismes, classe majoritaire des magnitudes de séismes.

3. Prédiction dans des zones très documentées (Pacifique - Japon) : Ce sont des zones dont l'usage sera le plus fréquent, avoir une pertinence sur la prédiction est crucial.

4. Prédiction dans des zones peu documentées (Océans isolés, Afrique) : Atteindre une fiabilité dans des régions avec moins de données montre la pertinence du modèle qui sait généraliser et rester performant dans toutes les régions.

5. Prédiction sur les données récentes (depuis les 2000) comparé aux anciennes (avant 1950) : On veut s'assurer que le modèle est performant sur les séismes récents tout en tolérant le manque de performance sur les séismes anciens (technologie moins performante, archivage moins rigoureux)

6. Prédiction avec peu de stations : Dégrader la qualité de l'input (manque de fiabilité des données par l'absence d'une quantité de stations pouvant mesurer les métriques) permet de tester la robustesse du modèle.

7. Score d'anomalie : Évaluer les métriques des anomalies nous permettra de comparer celles-ci avec les métriques des données considérées comme "normales" afin de confirmer l'intérêt de l'Isolation Forest dans le filtrage du dataset de test.

8. Résistance au bruit : Ajouter du bruit aux features de latitude & longitude constitue un bon test pour vérifier la variation de la MAE.

9. Imputation : Masquer aléatoirement une partie des données et conserver une MAE correcte jusqu'à 20% de masquage.

10. Tenue en charge : Augmenter progressivement la taille du dataset et vérifier la tenue du modèle et sa capacité à répondre rapidement malgré la quantité de données


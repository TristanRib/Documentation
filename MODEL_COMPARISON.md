# Comparaison des modèles

|                                           | XGBoost                 | RandomForest                                             |
| ----------------------------------------- | ----------------------- | -------------------------------------------------------- |
| MAE                                       | 0.2840                  | 0.1866                                                   |
| RMSE                                      | 0.4280                  | 0.3628                                                   |
| IsolationForest                           | Peu d'impact sur la MAE | Discrimination efficace sur les zones les plus complexes |
| Prédiction des gros séismes (magnitude 7) | 1.69                    | 1.39 (meilleur mais toujours moins bon qu'espéré)        |
| Test de charge                            | ~ 8 Mo                  | ~ 591 Mo (bien trop lourd)                               |
| Généralisation                            | Constance et précision  | Overfitting sur l'historicité des données                |

Recommandation : Conserver XGBoost. En effet, le modèle permet non seulement une meilleure portabilité ainsi qu'une résilience plus adaptée aux moments critiques (tenue en charge largement meilleure que RandomForest). De plus, XGBoost semble bien plus apte à prédire les plus récents séismes en évitant l'overfitting qui a été remarqué lors du test de généralisation sur RandomForest. Cependant, si l'utilisateur souhaite prioriser une précision moyennement supérieure en faisant abstraction des contraintes techniques, RandomForest pourrait potentiellement répondre à ce besoin, mais nous maintenons notre préférence pour XGBoost. Les autres tests n'ont pas été mentionnés car les résultats sont sensiblement similaires entre les deux modèles.

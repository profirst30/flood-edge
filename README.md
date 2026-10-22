# Crues : quand une station solaire doit-elle mesurer plus souvent ?

**Une station hydrométrique autonome, sur panneau solaire, mesure la hauteur d'eau une fois par heure pour économiser son énergie et ses envois. Pendant une crue, une mesure par heure arrive trop tard. Ce dépôt rejoue sept ans de vraies crues françaises pour comparer les règles qui décident, sur la station, quand passer en mode intensif.**

Projet personnel. Page de démo : https://profirst30.github.io/flood-edge/

## Le problème

Une station sans raccordement au réseau électrique vit de son panneau solaire et de sa batterie. Chaque mesure et chaque envoi en 4G coûtent de l'énergie. Une mesure par heure suffit en temps normal, mais pendant une crue soudaine, l'eau peut passer le seuil d'alerte juste après une mesure : l'alerte part alors avec près d'une heure de retard.

Mesurer tous les quarts d'heure en permanence règle le problème, mais multiplie les mesures par quatre, toute l'année, pour quelques jours de crue. La station doit donc décider elle-même quand passer au quart d'heure, et le faire **avant** le seuil. La question est : **quelle règle de décision donne des alertes à temps, avec le moins de mesures possible ?**

## Ce que fait le dépôt

1. **Données réelles** (`src/fetch.py`). Les hauteurs d'eau de six stations françaises, de 2019 à 2025, sur HydroPortail : la Garonne à Toulouse, et cinq rivières méditerranéennes à crues soudaines (Gard, Vidourle, Ardèche, Aude, Hérault).
2. **Rejeu** (`src/simulate.py`). La série source est remise sur une grille de 5 minutes : c'est la « vérité ». La station simulée mesure à chaque heure pile. Pendant une heure en mode intensif, elle mesure aussi à :15, :30 et :45. À chaque heure pile, une règle décide si l'heure qui commence est intensive, à partir des seules mesures horaires.
   - Le seuil d'alerte de chaque station est le maximum journalier dépassé 2 % des jours. Une crue commence quand la vérité atteint ce seuil.
   - Le retard de l'alerte est le temps entre ce moment et la première mesure de la station au-dessus du seuil. Une crue est manquée si aucune mesure ne la voit au-dessus du seuil.
3. **Cinq règles comparées.**
   - Fixe, une mesure par heure, et fixe, une mesure par quart d'heure : les deux bornes.
   - **Niveau** : intensif quand la dernière mesure dépasse une fraction du chemin entre le niveau médian et le seuil.
   - **Montée** : intensif au seuil, ou quand l'eau est montée de plus d'une certaine hauteur en une heure.
   - **Modèle** : intensif au seuil, ou quand un petit réseau de neurones prévoit le seuil dans les 3 heures, à partir des 6 dernières mesures horaires. 401 paramètres, exporté en ONNX puis quantifié en INT8 avec ONNX Runtime.
4. **Évaluation sans tricher.** Chaque station est testée avec un modèle entraîné sur les cinq autres. Son réglage est aussi choisi sur les cinq autres : le moins cher qui ne manque aucune crue et donne au moins 95 % des alertes en 15 minutes ou moins, comme une mesure par quart d'heure.
5. **Coût du calcul** (`src/run.py`). Le temps d'une décision du modèle, mesuré sur ce PC, sur un thread, en FP32 et en INT8.

## Résultats

<!-- results -->
| station | données | médiane (m) | seuil d'alerte (m) | crues | montée max en 1 h (m) |
|---|---|---:|---:|---:|---:|
| La Garonne à Toulouse | 2019-01-01 → 2025-12-31 (100,0 % valides) | 0,67 | 1,57 | 20 | 0,46 |
| Le Gard à Anduze | 2019-01-01 → 2025-12-31 (73,0 % valides) | 0,26 | 2,6 | 20 | 5,14 |
| Le Vidourle à Sommières | 2019-01-01 → 2025-12-31 (39,1 % valides) | 0,11 | 1,71 | 7 | 0,47 |
| L'Ardèche à Vallon-Pont-d'Arc | 2019-01-01 → 2025-12-31 (95,2 % valides) | -0,53 | 3,26 | 25 | 2,92 |
| L'Aude à Carcassonne | 2019-01-01 → 2025-12-31 (99,8 % valides) | 0,79 | 1,48 | 22 | 2,46 |
| L'Hérault à Laroque | 2019-01-02 → 2025-12-31 (90,2 % valides) | 0,45 | 3,54 | 23 | 4,58 |

Pour chaque station, le réglage est choisi sur les cinq autres : le moins cher qui ne manque aucune crue et donne au moins 95 % des alertes en 15 min ou moins.

| règle | mesures par an | × une par heure | crues manquées | retard moyen de l'alerte | 90 % des alertes en moins de | alertes en 15 min ou moins | pic manqué en moyenne | pic manqué au pire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixe, toutes les heures | 8 766 | 1,0 | 4 sur 117 | 26,9 min | 50,0 min | 37 % | 1,7 cm | 13,2 cm |
| fixe, tous les quarts d'heure | 35 064 | 4,0 | 0 sur 117 | 5,8 min | 10,0 min | 98 % | 0,3 cm | 7,0 cm |
| intensif au-dessus d'un niveau | 9 310 | 1,062 | 0 sur 117 | 6,4 min | 10,0 min | 95 % | 0,3 cm | 7,0 cm |
| intensif si l'eau monte vite | 10 163 | 1,159 | 0 sur 117 | 7,1 min | 10,0 min | 94 % | 0,3 cm | 7,0 cm |
| intensif si le modèle INT8 prévoit le seuil | 9 151 | 1,044 | 0 sur 117 | 6,3 min | 10,0 min | 96 % | 0,3 cm | 7,0 cm |

Sur AMD Ryzen 7 5825U with Radeon Graphics, un thread : une décision du modèle prend 26,6 µs en FP32 et 35,0 µs en INT8 (401 paramètres, 368 multiplications-additions ; fichier ONNX de 2 386 octets en FP32, 5 586 en INT8). INT8 et FP32 prennent la même décision sur 99,9 % des heures.
<!-- /results -->

Tous les chiffres ci-dessus sont écrits par `python -m src.run` à partir des données : aucun n'est tapé à la main.

## Ce que disent les chiffres

- **Une mesure par heure ne suffit pas pendant une crue.** L'alerte arrive en moyenne avec près d'une demi-heure de retard, et quelques crues brèves passent entre deux mesures sans être vues.
- **Passer en mode intensif avant le seuil rattrape presque tout le quart d'heure, pour quelques pour cent de mesures en plus.** La station garde une mesure par heure presque toute l'année et ne mesure au quart d'heure que quand l'eau approche du seuil.
- **La règle du niveau, la plus simple, fait presque aussi bien que le modèle.** Le modèle INT8 atteint l'objectif avec un peu moins de mesures, mais l'écart est faible et repose sur 117 crues : ce n'est pas une preuve qu'il est meilleur partout. Sur la station, la règle du niveau se règle et s'explique en une ligne.
- **La vitesse de montée seule est un mauvais signal.** Sur quatre des rivières méditerranéennes, l'eau est déjà montée de plusieurs mètres en une heure (dernière colonne du premier tableau) : quand une mesure horaire voit la montée, il est souvent trop tard. Pour tenir l'objectif, il faut un réglage si sensible qu'il dépense plus de mesures que les deux autres règles. Elle sauve pourtant la plus grosse crue du Gard (octobre 2024, sur la page de démo), où les deux autres règles réagissent trop tard : les deux signaux se complètent.
- **L'INT8 prend la même décision que le FP32 presque à chaque heure.** Mais sur ce processeur, il est plus lent et son fichier est plus gros : pour 401 paramètres, les nœuds de quantification pèsent plus que les poids. L'INT8 se justifie sur un microcontrôleur sans unité de calcul flottant ou avec un accélérateur, pas ici. Une décision par heure coûte de toute façon des dizaines de microsecondes : le calcul n'est pas le poste d'énergie, les mesures et les envois le sont.

## Limites

- **Aucune station ni aucun microcontrôleur testés.** Le temps de calcul est mesuré sur un PC portable. Je n'ai trouvé aucune source publique sur l'énergie d'une mesure et d'un envoi 4G, donc les règles sont comparées en nombre de mesures, pas en wattheures.
- **La vérité est reconstruite.** La série source ne garde une valeur que quand le niveau change : entre deux points, la vérité est une droite. Un pic plus court que l'écart entre deux points est invisible à toutes les règles. Les trous de plus de 12 heures sont comptés comme données manquantes : le Vidourle n'a que 39 % de données valides.
- **Le seuil d'alerte est fixé par une statistique**, pas par un vrai seuil de vigilance. Un exploitant choisirait le sien, et les règles devraient être réglées à nouveau.
- **Une mesure est supposée exacte et toujours envoyée.** Le bruit du capteur, les coupures réseau et les envois groupés ne sont pas modélisés.
- **Le réglage du modèle est choisi sur ses propres stations d'entraînement**, faute de stations en plus. Pour trois stations sur six, le réglage retenu est au bord de la grille testée.

## Reproduire

```sh
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate sous Linux
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m src.fetch          # 42 requêtes à HydroPortail, une par station et par an -> data/ (quelques minutes)
python -m src.run            # rejeu, modèles INT8, coût -> results/, docs/, ce README
python -m pytest tests
```

La page de démo (`docs/index.html`) lit `docs/results.json` et `docs/series.json`. Pour la voir en local : `python -m http.server -d docs`.

Données : hauteurs d'eau instantanées, « données les plus valides », [HydroPortail](https://www.hydro.eaufrance.fr/) (SC Vigicrues, ministère de la Transition écologique), téléchargées le 22/10/2026. Les codes des stations viennent du référentiel de [Hub'Eau](https://hubeau.eaufrance.fr/page/api-hydrometrie). `src/fetch.py` appelle l'adresse que le site utilise pour ses graphiques et son export, qui n'est pas une API documentée ; les mêmes séries s'exportent à la main depuis la page de chaque station. Ni HydroPortail ni Hub'Eau n'affichent de licence pour ces séries. Ce sont des informations publiques de l'État, réutilisables à condition de ne pas les altérer et d'en citer la source et la date (articles L321-1 et L322-1 du code des relations entre le public et l'administration) ; le catalogue Sandre d'Eaufrance indique la Licence Ouverte Etalab pour le référentiel des stations.

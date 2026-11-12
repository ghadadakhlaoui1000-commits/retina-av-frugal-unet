# Segmentation artères / veines de la rétine par U-Net 2D : modèles légers et mesure carbone

**Ghada Dakhlaoui** – Génie biomédical, INTeK (Tunisie) · PyTorch · Kaggle (GPU Tesla T4)

Projet personnel de segmentation multi-classes des vaisseaux rétiniens (fond, artère, veine) sur photos couleur du fond d'œil. Il compare quatre tailles de U-Net avec un protocole identique, mesure l'empreinte carbone de chaque entraînement et calcule des biomarqueurs vasculaires simplifiés à partir des masques prédits.

## Résumé des résultats (jeu de test : 31 images)

| Modèle | Paramètres | Dice artère | Dice veine | Dice vaisseau | CO₂ (g) | Entraînement | Inférence (ms / image 1024²) |
|---|---|---|---|---|---|---|---|
| U-Net 64 canaux (référence) | 31,04 M | 0,764 | 0,785 | 0,856 | 17,44 | 37,8 min | 147 |
| U-Net 32 canaux | 7,76 M | 0,769 | 0,787 | 0,857 | 7,77 | 16,9 min | 55 |
| U-Net 16 canaux | 1,94 M | 0,751 | 0,772 | 0,853 | 5,65 | 12,9 min | 31 |
| U-Net 8 canaux, 3 niveaux | 0,12 M | 0,678 | 0,707 | 0,845 | 5,26 | 12,9 min | 20 |

Dice par image (moyenne ± écart-type) pour la référence : artère 0,760 ± 0,125, veine 0,788 ± 0,121, vaisseau 0,857 ± 0,120.

**Lecture prudente.**
- Le modèle à 32 canaux a 4 fois moins de paramètres que la référence, pour un Dice équivalent (la différence est bien inférieure à l'écart-type entre images) et 2,2 fois moins de CO₂.
- Le modèle à 16 canaux perd environ 1,3 point de Dice artère avec 16 fois moins de paramètres.
- Le plus petit modèle (0,12 M) perd environ 8 points sur les artères et les veines, mais garde un Dice « vaisseau » de 0,845 : il détecte bien les vaisseaux, mais confond davantage artères et veines.
- Un seul entraînement par modèle (une graine) et 31 images de test : les écarts de 0,01 à 0,02 ne sont pas statistiquement établis.

## Données

- **Rotterdam RAV** (artery-vein dataset) : 206 photos couleur 1024 × 1024 avec masques artère/veine, hébergées sur DataverseNL (DOI 10.34894/9OIMWY).
- Citation : Vargas Quiros et al., arXiv:2512.17322. **Vérifiez la licence sur la page du dépôt** (les sources indiquent CC BY ou CC BY-NC) ; les données ne sont pas redistribuées dans ce dépôt.
- Codage des masques : rouge = artère, bleu = veine, magenta = croisement, vert = inconnu. Croisements et vaisseaux inconnus reçoivent l'étiquette 255, **ignorée par la perte**.
- Répartition moyenne des pixels : fond 93,09 %, artère 3,07 %, veine 3,54 %, croisement 0,15 %, inconnu 0,16 %. Ce fort déséquilibre justifie une perte combinant entropie croisée et Dice.
- Séparation **par image** : 144 entraînement / 31 validation / 31 test (graine 42), avec un contrôle automatique d'absence de fuite.

## Méthode

- **Prétraitement** : normalisation par canal dans le champ de vue, avec un plancher sur l'écart-type (évite des valeurs aberrantes sur des patches presque uniformes).
- **Patches** 256 × 256, centrés sur un vaisseau dans 70 % des cas ; augmentation : retournements, rotations de 90°, gamma, contraste/luminosité, gain par canal. Validation et test sur images entières, sans augmentation.
- **Modèle** : U-Net 2D classique (double convolution + BatchNorm + ReLU, convolutions transposées), 3 canaux d'entrée, 3 classes de sortie. Variantes : 64, 32 et 16 canaux de base (4 niveaux), et 8 canaux avec 3 niveaux.
- **Perte** : entropie croisée pondérée (poids 1/√fréquence) + (1 − Dice moyen artère/veine), pixels ignorés exclus.
- **Entraînement** : AdamW (lr 1e-3, décroissance cosinus, weight decay 1e-4), 60 époques, 1 152 patches par époque, batch 8, précision mixte, gradient écrêté à 5, protection contre les lots à perte non finie. Le meilleur modèle est choisi sur la **validation** ; le **test** n'est évalué qu'à la fin.
- **Carbone** : estimation CodeCarbon de chaque entraînement (valeurs comparables entre les modèles, mais approximatives et dépendantes du centre de données).

## Biomarqueurs vasculaires simplifiés (modèle de référence, 31 images de test)

| Biomarqueur | Référence (moy.) | Prédiction (moy.) | Corrélation r | Erreur absolue moy. | Biais (préd. − réf.) |
|---|---|---|---|---|---|
| Densité vasculaire (%) | 9,60 | 10,32 | 0,96 | 0,75 | +0,72 |
| Dimension fractale (box-counting) | 1,519 | 1,526 | 0,97 | 0,009 | +0,007 |
| Calibre moyen des artères (px) | 5,14 | 5,51 | 0,86 | 0,45 | +0,36 |
| Calibre moyen des veines (px) | 5,73 | 6,21 | 0,90 | 0,51 | +0,49 |
| Rapport calibre artères / veines | 0,901 | 0,893 | 0,73 | 0,042 | −0,009 |

Les masques prédits surestiment légèrement l'épaisseur des vaisseaux (calibre et densité un peu trop élevés). **Ce ne sont pas les CRAE, CRVE et AVR standard** : pas de détection du disque optique, pas de formule de Knudtson, calibre en pixels (images 1024 × 1024). Densité : vaisseaux de toutes classes dans le champ de vue. Calibre : 2 × distance au fond, échantillonnée sur le squelette.

## Limites

- Un seul entraînement par modèle (une graine) : la variabilité due au hasard n'est pas mesurée.
- 31 images de test ; pas de validation croisée.
- Photos couleur seulement : pas de multimodalité (angiographie FFA).
- Croisements et vaisseaux « inconnus » ignorés : leur segmentation n'est pas évaluée.
- Pas de comparaison avec une méthode de référence (nnU-Net) ni avec des architectures frugales spécialisées.
- Pas de métrique de topologie (clDice) ni de distance de surface : le Dice ne mesure pas bien la continuité des vaisseaux fins.
- Biomarqueurs simplifiés, non validés cliniquement.

## Pistes

Plusieurs graines et validation croisée ; métriques de topologie ; comparaison avec nnU-Net ; quantification, élagage ou distillation pour aller plus loin dans la frugalité ; jeu de données multimodal CFP + FFA ; CRAE/CRVE avec détection du disque optique.

## Reproduire

1. Kaggle (ou Colab) avec GPU T4, Internet activé.
2. Ouvrir `Retina_Projet_Complet_KAGGLE.ipynb` et exécuter toutes les cellules (environ 1 h 30) : téléchargement des données, pipeline, entraînement des quatre modèles, tableau comparatif, biomarqueurs.

Versions : PyTorch (environnement Kaggle), CodeCarbon. Graine 42.

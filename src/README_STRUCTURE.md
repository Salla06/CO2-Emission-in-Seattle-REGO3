# Structure du projet - Présentation Seattle CO2

## 📁 Organisation des fichiers

```
CO2-Emission-in-Seattle-REGO3/
├── src/
│   ├── img/                          # Dossier des images et photos
│   │   ├── team/                     # Photos des membres de l'équipe
│   │   │   ├── bognon.jpg
│   │   │   ├── daiferle.jpg
│   │   │   ├── dia.jpg
│   │   │   ├── gueye.jpg
│   │   │   └── toure.jpg
│   │   └── logos/                    # Logos et images
│   │
│   ├── styles/                       # Styles et scripts
│   │   ├── presentation.css          # Styles personnalisés
│   │   └── presentation.js           # Scripts de navigation
│   │
│   ├── *.py                          # Modules Python du projet
│   └── __init__.py
│
├── reports/
│   ├── presentation/
│   │   └── presentation_seattle_co2.html  # Présentation HTML
│   └── figures/
│
├── notebooks/
│   └── 04_modelisation_finale.ipynb  # Notebook d'analyse
│
├── models/
│   ├── pipeline_modele1_best.pkl
│   └── pipeline_modele2_best.pkl
│
├── results/
│   ├── figures/
│   ├── predictions_finales.csv
│   └── metrics_comparison.json
│
└── data/
    ├── raw_data/
    ├── processed_data/
    └── interim_data/
```

## 🎯 Équipe du projet

### Superviseure
- **Mme Fatou SALL** - ENSAE Dakar

### Membres
1. **Enagnon Justin BOGNON** - Data Scientist
2. **Mariane DAÏFERLE** - ML Engineer
3. **Mouhammdou DIA** - Data Analyst
4. **Aïssatou GUEYE** - Data Scientist
5. **Ndèye Salla TOURE** - Data Engineer

## 🎨 Présentation

### Fichiers clés
- `presentation_seattle_co2.html` : Présentation complète (13 slides)
- `presentation.css` : Styles personnalisés
- `presentation.js` : Navigation et contrôles

### Slides
1. Titre principal
2. Contexte et enjeux climatiques
3. Objectifs opérationnels
4. Méthodologie
5. Données utilisées
6. Performance des modèles
7. Facteurs d'émissions influents
8. Recommandations politiques
9. Pipeline MLOps
10. Conclusion et résultats
11. Revue de littérature
12. **Équipe du projet** (slide avec photos)
13. Remerciements

### Contrôles de la présentation
- **→ ou Espace** : Slide suivante
- **← Arrow** : Slide précédente
- **F** : Mode plein écran
- **P** : Autoplay on/off

## 📋 Ajout de photos des membres

Pour ajouter les photos réelles des membres :

1. Placez les photos dans `src/img/team/`
2. Nommez-les selon les prénoms :
   - `bognon.jpg`
   - `daiferle.jpg`
   - `dia.jpg`
   - `gueye.jpg`
   - `toure.jpg`

3. Modifiez le HTML du slide 12 pour remplacer les emojis par des balises `<img>` :

```html
<div style="width: 150px; height: 150px; margin: 0 auto 20px; border-radius: 50%; border: 3px solid #2E8B57; overflow: hidden;">
    <img src="../src/img/team/bognon.jpg" alt="Photo" style="width: 100%; height: 100%; object-fit: cover;">
</div>
```

## 📊 Résultats de la modélisation

**Modèle 1 (Sans ENERGY STAR)** - Random Forest
- R² = 1.00
- RMSE = 0.85 T CO₂
- MAPE = 0.30%

**Modèle 2 (Avec ENERGY STAR)** - Random Forest
- R² = 1.00
- RMSE = 0.82 T CO₂
- MAPE = 0.31%

**Conclusion** : Le Modèle 1 sans ENERGY STAR est préférable pour sa simplicité.

## 🚀 Utilisation

### Ouvrir la présentation
```bash
# Ouvrir directement dans un navigateur
reports/presentation/presentation_seattle_co2.html
```

### Exécuter le notebook
```bash
jupyter notebook notebooks/04_modelisation_finale.ipynb
```

## 📝 Notes importantes

- Les images placeholders utilisent des gradients CSS
- Les photos réelles remplacent les gradients une fois ajoutées
- Tous les styles et scripts sont centralisés dans `src/styles/`
- La structure est responsive et compatible mobile

---

**Mise à jour** : 1er février 2026
**Équipe** : ENSAE Dakar
**Projet** : Prédiction des émissions CO₂ - Bâtiments Seattle

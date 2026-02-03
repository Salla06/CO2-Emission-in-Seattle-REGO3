# 🏢 Projet Data Science : Prévision d'Émissions CO2 & Énergie - Seattle

## 📋 Contexte et Objectifs
La ville de Seattle s'est fixé pour objectif d'atteindre la **neutralité carbone d'ici 2050**. Ce projet vise à aider la ville et les gestionnaires immobiliers à mieux anticiper les émissions de gaz à effet de serre (GES) et la consommation d'énergie des bâtiments non résidentiels.

**Objectifs clés :**
1.  **Prédire** les émissions de CO2 (`TotalGHGEmissions`) et la consommation d'énergie (`SiteEnergyUse(kBtu)`).
2.  **Évaluer l'intérêt** du relevé "Energy Star Score" (coûteux) dans la qualité des prédictions.
3.  **Développer un outil de pilotage** (Dashboard) pour simuler des scénarios de rénovation.

---

## 🔬 Partie 1 : Pipeline Data Science (Analyse Approfondie)

### 1.1. Préparation et Nettoyage des Données
*   **Source** : Données "City of Seattle Building Energy Benchmarking" (2016).
*   **Nettoyage Rigoureux** :
    *   Filtrage des bâtiments résidentiels (hors périmètre).
    *   Traitement des valeurs aberrantes (Outliers) sur les consommations (suppression des anomalies physiques évidentes).
    *   Gestion des valeurs manquantes (Imputation ou suppression selon criticité).
*   **Feature Engineering** :
    *   Création de variables dérivées (ex: ratio surface/étage, âge du bâtiment, densité d'occupation).
    *   Encodage des variables catégorielles (One-Hot Encoding pour les types d'usage et quartiers).
    *   **Transformation Logarithmique** : Application de `Log(y+1)` sur les cibles (CO2 et Énergie) pour corriger l'asymétrie (skewness) des distributions et améliorer la performance des modèles.

### 1.2. Stratégie de Modélisation & Modèles Testés
Nous avons comparé systématiquement plusieurs familles d'algorithmes pour identifier la meilleure approche :

| Famille | Modèles Testés | Performance (R²) | Observation |
| :--- | :--- | :--- | :--- |
| **Baseline** | Dummy Regressor | ~0.00 | Seuil de référence (moyenne simple). |
| **Linéaire** | Ridge | ~0.52 | Performance modérée. Difficulté à capturer les non-linéarités complexes du parc immobilier. |
| **Ensemble (Bagging)** | **Random Forest** | ~0.60 - 0.63 | Performant et robuste aux outliers. |
| **Ensemble (Boosting)** | **Gradient Boosting** | **0.65 - 0.68** | **Vainqueur**. Meilleure généralisation et précision optimale après tuning. |

*L'optimisation des hyperparamètres a été réalisée via `GridSearchCV` (Validation Croisée 5-folds).*

### 1.3. Résultats Comparatifs : Avec vs Sans Energy Star Score
Un point crucial de l'étude était de déterminer si le "Energy Star Score" est indispensable.

| Scénario | Modèle Retenu | R² (Test) | RMSE (log) | MAE (log) | MAPE | Analyse |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Avec Energy Star** | **Gradient Boosting** | **0.6833** | **0.7246** | **0.5506** | **88.00%** | **Performance Optimale**. Le score apporte une information métier précieuse sur l'efficacité énergétique. |
| **Sans Energy Star** | **Gradient Boosting** | **0.6509** | **0.7608** | **0.5823** | **96.44%** | **Alternative Viable**. Le modèle reste performant en s'appuyant uniquement sur les caractéristiques structurelles (Surface, Usage, Année). |

**Gains de Performance (Modèle 2 vs Modèle 1) :**
*   **R² Score** : +0.0324 (+5.0%)
*   **RMSE (log)** : -0.0362 (-4.8% d'erreur)
*   **MAPE** : -8.44 points (amélioration de la précision relative)

**Conclusion** : L'Energy Star Score améliore significativement les prédictions (+5% de variance expliquée), justifiant son coût pour les bâtiments prioritaires. Le modèle "Sans Score" reste néanmoins suffisamment fiable (R²=0.65) pour être déployé sur l'ensemble du parc non audité.

---

## 📊 Partie 2 : Le Dashboard de Pilotage (Application Dash)

Pour rendre ces modèles accessibles, nous avons développé une application web interactive complète, bilingue et responsive.

### Architecture Technique
*   **Frontend** : Dash (Plotly), Dash Bootstrap Components.
*   **Backend** : Python, Flask (Core Dash), Scikit-Learn (Inférence modèles).
*   **Features** : Support multilingue (FR/EN), Thèmes (Clair/Sombre), Export PDF.

### Fonctionnalités Détaillées

#### 1. 🔮 Calculateur Prédictif (IA)
*   **Saisie Interactive** : Formulaire simple pour entrer les caractéristiques d'un bâtiment et obtenir une prédiction immédiate.
*   **Sélection Intelligente** : Choix automatique du modèle (Avec/Sans Score) selon les données saisies.
*   **Batch Processing** : Possibilité d'uploader un fichier CSV pour prédire les émissions de centaines de bâtiments simultanément.
*   **Visualisation** : Jauges de confiance XAI et explication des résultats.

#### 2. 🛠️ Simulateur de Rénovation ("What-If")
Outil d'aide à la décision pour simuler l'impact de travaux sur le score Energy Star et les émissions :
*   **Menu Travaux** : *Relampage LED (+8 pts)*, *Pompe à Chaleur (+15 pts)*, *Isolation (+10 pts)*, *Solaire (+12 pts)*.
*   **Graphiques** : Visualisation "Avant/Après" de la réduction carbone et des économies potentielles.
*   **Explicabilité** : Transparence sur les gains de points (basé sur les standards Portfolio Manager).

#### 3. ⭐ Analyse d'Impact Energy Star
*   **Podium de Performance** : Positionnement du bâtiment face à :
    *   La *Moyenne des Bâtiments Similaires*.
    *   L'*Objectif Zéro Carbone*.
    *   Le *Top Performance*.
*   **Jauge Interactive** : Visualisation claire de l'écart (GAP) et système de notation par étoiles.

#### 4. 📅 Benchmark 2050 (Trajectoire Climatique)
*   Projection temporelle des émissions face aux **Climate Targets** de Seattle.
*   Visualisation de la **"Zone d'Effort"** (l'écart à combler) et des jalons réglementaires (2030, 2040, Neutralité 2050).

---

## ▶️ Installation et Utilisation

Cloner le projet et installer les dépendances :
```bash
pip install -r requirements.txt
```

Lancer l'application :
```bash
python seattle_dashboard/app.py
```
*Accéder à l'interface via https://co2-emission-in-seattle-rego3-i479.onrender.com/*

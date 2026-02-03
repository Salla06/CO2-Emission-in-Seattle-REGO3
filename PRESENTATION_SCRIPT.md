# Script de Présentation du Dashboard Seattle CO2

*(Ce document est conçu pour vous servir de guide lors de votre présentation orale. Il est écrit à la première personne "Je".)*

---

## 1. Introduction & Contexte

Bonjour à tous,

Je vous présente aujourd'hui le **Dashboard Seattle CO2**, une application interactive conçue pour analyser et prédire les émissions de gaz à effet de serre des bâtiments non résidentiels de la ville de Seattle.

**La problématique :**
La ville de Seattle s'est fixée un objectif ambitieux de neutralité carbone d'ici 2050. Pour y parvenir, il est crucial de comprendre quels bâtiments polluent le plus et comment réduire leur impact. Cependant, les relevés réels d'émissions sont coûteux à obtenir.

**Notre solution :**
J'ai développé cet outil pour répondre à deux besoins :
1. **Comprendre** les données existantes via une exploration visuelle.
2. **Prédire** les émissions futures sans avoir besoin de relevés coûteux, grâce au Machine Learning.

---

## 2. Architecture Technique

Côté technique, j'ai choisi une stack robuste et adaptée à la Data Science :

*   **Front-end & Back-end :** J'utilise **Dash (Plotly)** en Python. C'est un choix stratégique qui permet de conserver tout le workflow en Python, du traitement des données jusqu'à l'interface utilisateur.
*   **Design :** J'ai intégré un design system personnalisé (Thème Sombre/Clair) avec **Dash Bootstrap Components** et du CSS sur-mesure pour une expérience utilisateur fluide et moderne.
*   **Machine Learning :** Le moteur prédictif repose sur **Scikit-Learn** et **XGBoost**.

---

## 3. Démonstration des Fonctionnalités

*(Navigation dans l'application)*

Laissez-moi vous guider à travers les fonctionnalités clés :

### A. Onglet "Insights" (Accueil)
C'est le tableau de bord exécutif. En un coup d'œil, on voit les KPIs de la ville : émissions totales, consommation d'énergie moyenne. La carte interactive permet de localiser géographiquement les bâtiments les plus polluants.

### B. Onglet "Analyse Exploratoire"
Ici, nous plongeons dans les données.
*(Montrer quelques graphiques)*
On observe par exemple la corrélation entre la surface du bâtiment et ses émissions, ou l'impact de l'année de construction. Ces visualisations interactives ont permis d'orienter nos choix de modélisation.

### C. Onglet "Modélisation" (Le Cœur du Projet)
C'est ici que nous évaluons la performance de nos algorithmes.
Nous avons testé plusieurs modèles : Régression Linéaire, Random Forest, et Gradient Boosting.
*(Montrer le tableau comparatif)*
Comme vous le voyez, le modèle **Gradient Boosting** offre les meilleures performances avec un R² de **0.68** sur le jeu de test, ce qui est très satisfaisant pour des données réelles aussi hétérogènes.

---

## 4. Zoom Technique : L'Intégration du Modèle

C'est une partie dont je suis particulièrement fier : **comment passer du notebook de recherche à une application web fonctionnelle ?**

Le processus se déroule en 3 étapes :

1.  **Entraînement & Sauvegarde (Côté Notebook) :**
    Dans mes notebooks Jupyter, j'ai entraîné le modèle final. J'utilise la librairie `joblib` pour "sérialiser" (sauvegarder) non seulement le modèle, mais tout le pipeline de traitement. C'est crucial : cela garantit que les nouvelles données subiront *exactement* les mêmes transformations (mise à l'échelle, encodage) que les données d'entraînement.

2.  **Chargement "Lazy" (Côté App) :**
    Dans l'application, j'ai codé une classe `Predictor`. Pour optimiser la vitesse de démarrage du site, le modèle (qui peut être lourd) n'est chargé en mémoire que lors de la *première* demande de prédiction. C'est ce qu'on appelle le "Lazy Loading".

3.  **Prédiction en Temps Réel :**
    Quand un utilisateur entre les caractéristiques d'un bâtiment :
    - L'application construit un DataFrame à la volée.
    - Elle applique le pipeline de pré-traitement sauvegardé.
    - Le modèle prédit le logarithme des émissions.
    - Nous appliquons la transformation inverse (exponentielle) pour afficher le résultat final en tonnes de CO2.

---

## 5. Onglet "Prédiction & Simulation"

Démonstration en direct :
*(Remplir le formulaire : Bureau, Downtown, 50 000 sqft...)*

Je rentre les caractéristiques d'un projet immobilier. En cliquant sur **"Prédire"**, le modèle interroge le moteur en temps réel et nous donne une estimation précise : **X tonnes de CO2/an**.

Plus fort encore, l'outil propose une **Simulation d'Impact Energy Star**.
*(Bouger le slider Energy Star)*
On peut voir instantanément comment l'amélioration de l'efficacité énergétique (le score Energy Star) réduirait l'empreinte carbone du bâtiment. Cela transforme un modèle "boîte noire" en un outil d'aide à la décision concret pour les urbanistes.

---

## 6. Conclusion 

En conclusion, ce dashboard n'est pas juste une visualisation de données. C'est un outil complet qui :
1.  **Valorise** les données open data de Seattle.
2.  **Industrialise** un modèle de Machine Learning performant.
3.  **Rend accessible** des prédictions complexes aux décideurs non-techniques.

Merci de votre attention, je suis prêt à répondre à vos questions sur le code ou la méthodologie.

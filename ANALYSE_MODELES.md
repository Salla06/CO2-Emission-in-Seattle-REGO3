# ============================================================================
# RAPPORT D'ANALYSE : Notebooks vs Dashboard
# ============================================================================
# Généré le : 2026-02-01
# Analyse des notebooks 04_modelisation.ipynb et 05_optimization.ipynb
# ============================================================================

## MODÈLES TESTÉS DANS LES NOTEBOOKS

### Notebook 4 - Modélisation de base
- Ridge (régularisation L2)
- Lasso (régularisation L1)  
- Random Forest (paramètres par défaut)
- Gradient Boosting (paramètres par défaut)
- SVR (Support Vector Regression)

### Notebook 5 - Optimisation

#### Modèle 1 (SANS ENERGYSTARScore) :
1. Random Forest + GridSearchCV
   - Grille complète de paramètres
   - Cross-validation 5-fold
   
2. Random Forest + RandomizedSearchCV
   - Recherche aléatoire sur espace plus large
   - N_ITER = 50 itérations
   
3. Gradient Boosting + RandomizedSearchCV
   - N_ITER = 50 itérations

#### Modèle 2 (AVEC ENERGYSTARScore) :
1. Random Forest + RandomizedSearchCV
   - N_ITER = 50 itérations
   - **MODÈLE FINAL SAUVEGARDÉ**
   
2. Gradient Boosting + RandomizedSearchCV
   - N_ITER = 50 itérations

## MODÈLE UTILISÉ DANS LE DASHBOARD

**Fichier chargé** : `models/pipeline_modele2_best.pkl`

**Type** : Pipeline scikit-learn contenant :
- Étape 1 : StandardScaler (normalisation)
- Étape 2 : RandomForestRegressor (modèle optimisé)

**Caractéristiques** :
- Inclut ENERGYSTARScore
- Optimisé via RandomizedSearchCV
- Target : TotalGHGEmissions_log (transformation log1p)

## HYPERPARAMÈTRES OPTIMAUX (À VÉRIFIER DANS LE NOTEBOOK)

Les meilleurs paramètres trouvés par RandomizedSearchCV pour le Modèle 2 RF :
- n_estimators : [à extraire du notebook]
- max_depth : [à extraire du notebook]
- min_samples_split : [à extraire du notebook]
- min_samples_leaf : [à extraire du notebook]
- max_features : [à extraire du notebook]

## PROBLÈMES IDENTIFIÉS

### 1. Feature Engineering manuel dans prediction_logic.py
❌ Le code actuel refait manuellement le feature engineering
✅ Le pipeline devrait gérer ça automatiquement

### 2. Incohérence potentielle
❌ Les features créées manuellement peuvent différer du pipeline
✅ Utiliser directement pipeline.predict(raw_data)

### 3. Modèles manquants
❌ Seul Modèle 2 (RF) est disponible
⚠️  Modèle 1 (sans ENERGY STAR) n'est pas exposé
⚠️  Comparaisons Ridge/Lasso/GB/SVR absentes

## RECOMMANDATIONS

### Priorité 1 : Corriger prediction_logic.py
- Utiliser le pipeline complet
- Passer des features RAW (non transformées)
- Laisser le pipeline gérer les transformations

### Priorité 2 : Vérifier model_features.json
- S'assurer qu'il liste les features attendues par le modèle
- Correspondance exacte avec les features du training

### Priorité 3 : Documentation
- Documenter les hyperparamètres optimaux
- Ajouter les métriques du modèle (R², RMSE, MAPE)

## MÉTRIQUES ATTENDUES (Modèle 2 optimisé)

D'après le notebook 5 :
- R² Cross-Validation : ~0.XX [à vérifier]
- R² Test : ~0.XX [à vérifier]
- RMSE Test (log) : ~0.XX [à vérifier]
- RMSE Test (original) : ~XX tonnes [à vérifier]
- MAPE Test : ~XX% [à vérifier]

## STRUCTURE ATTENDUE DU PIPELINE

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(
        n_estimators=...,
        max_depth=...,
        min_samples_split=...,
        min_samples_leaf=...,
        max_features=...,
        random_state=42
    ))
])
```

## FEATURES ATTENDUES PAR LE MODÈLE

Le modèle attend probablement ~465 features après :
- One-Hot Encoding des variables catégorielles :
  * BuildingType
  * PrimaryPropertyType
  * Neighborhood
  * LargestPropertyUseType
  * SecondLargestPropertyUseType

- Variables numériques :
  * Latitude, Longitude
  * PropertyGFATotal, PropertyGFAParking, PropertyGFABuilding(s)
  * NumberofFloors, YearBuilt, NumberofBuildings
  * LargestPropertyUseTypeGFA, SecondLargestPropertyUseTypeGFA
  * ENERGYSTARScore
  * SteamUse(kBtu), Electricity(kWh), NaturalGas(therms)

- Variables dérivées (Feature Engineering) :
  * GFA_per_floor = PropertyGFATotal / NumberofFloors
  * Parking_ratio = PropertyGFAParking / PropertyGFATotal
  * Building_age_squared = (2016 - YearBuilt)^2
  * Is_old_building = 1 if age > 30 else 0
  * Size_floors = PropertyGFATotal * NumberofFloors
  * Age_size = building_age * PropertyGFATotal
  * Age_floors = building_age * NumberofFloors
  * GFA_sqrt = sqrt(PropertyGFATotal)
  * Floors_squared = NumberofFloors^2
  * Neighborhood_mean, Neighborhood_std
  * PrimaryPropertyType_mean, PrimaryPropertyType_std

## PROCHAINES ÉTAPES

1. ✅ Exécuter inspect_model.py pour confirmer la structure
2. ⬜ Réécrire prediction_logic.py selon le format pipeline
3. ⬜ Extraire les hyperparamètres exacts du notebook 5
4. ⬜ Tester avec des données réelles du dashboard
5. ⬜ (Optionnel) Exposer d'autres modèles pour comparaison

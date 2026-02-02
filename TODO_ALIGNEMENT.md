# 🎯 RÉSUMÉ : Alignement Notebooks ↔ Dashboard

## ✅ CE QUI A ÉTÉ FAIT

### 1. Analyse Complète
- ✅ Analysé `04_modelisation.ipynb` : 5 modèles testés (Ridge, Lasso, RF, GB, SVR)
- ✅ Analysé `05_optimization.ipynb` : optimisations GridSearch et RandomizedSearch
- ✅ Identifié le modèle final : **RandomForest optimisé (Modèle 2 avec ENERGY STAR)**
- ✅ Vérifié que `pipeline_modele2_best.pkl` est bien chargé dans le dashboard

### 2. Problèmes Identifiés
- ❌ `prediction_logic.py` refait manuellement le feature engineering
- ❌ Risque d'incohérence entre features manuelles et features du pipeline
- ❌ Pas d'utilisation directe du pipeline scikit-learn

### 3. Fichiers Créés

#### `ANALYSE_MODELES.md`
Rapport détaillé listant :
- Tous les modèles testés dans les notebooks
- La structure attendue du pipeline
- Les 465 features après one-hot encoding
- Les hyperparamètres à extraire

#### `scripts/inspect_model.py`  
Script d'inspection pour :
- Afficher la structure du pipeline .pkl
- Lister les hyperparamètres optimaux

 Vérifier les features attendues
- Tester une prédiction simple

#### `utils/prediction_logic_v2.py`
Version CORRIGÉE qui :
- ✅ Utilise le pipeline complet
- ✅ Prépare les features brutes (pas de transformation manuelle)
- ✅ Laisse le StandardScaler et le modèle gérer les transformations
- ✅ Gère les erreurs avec fallback
- ✅ Extrait les feature importances du RF

---

## ⏳ CE QU'IL RESTE À FAIRE

### Étape 1 : Vérifier l'installation
```bash
python -m pip list | findstr "scikit joblib pandas numpy"
```

Si manquant :
```bash
python -m pip install joblib scikit-learn pandas numpy
```

### Étape 2 : Inspecter le Modèle
```bash
cd "c:\Users\HP\OneDrive\Bureau\Projet Machine Learning\seattle_dashboard"
python scripts\inspect_model.py
```

**Objectifs :**
- Confirmer que c'est un Pipeline
- Voir les étapes (StandardScaler + RandomForestRegressor)
- Extraire les hyperparamètres optimaux
- Vérifier combien de features attendues

### Étape 3 : Extraire les Hyperparamètres du Notebook 5

Ouvrir `notebooks/05_optimization.ipynb` et chercher :
```python
# Autour de la ligne 4610-4650
rf_random_m2.best_params_
# Devrait afficher :
# {
#     'n_estimators': XXX,
#     'max_depth': XXX,
#     'min_samples_split': XXX,
#     'min_samples_leaf': XXX,
#     'max_features': 'XXX'
# }
```

**Documenter ces valeurs dans** `ANALYSE_MODELES.md`

### Étape 4 : Vérifier model_features.json

Deux possibilités :

**A) Le fichier existe déjà**
```bash
# Vérifier son contenu
type utils\model_features.json
```

**B) Le fichier n'existe PAS**
Il faut le créer en extrayant les features du notebook 2 (preprocessing) :
```python
# Dans le notebook 02_processing.ipynb
# Après le one-hot encoding
X_train_m2.columns.tolist()
# Sauvegarder dans utils/model_features.json
```

### Étape 5 : Tester la Nouvelle Version

**Test 1 : Comparaison côte à côte**
```python
# Créer un script de test
from utils.prediction_logic import predict_co2 as predict_old
from utils.prediction_logic_v2 import predict_co2 as predict_new

test_data = {
    'gfa': 50000,
    'year_built': 2010,
    'number_of_floors': 5,
    'energy_star_score': 75,
    'location': {'lat': 47.6097, 'lon': -122.3338},
    'neighborhood': 'Downtown',
    'building_type': 'Office'
}

pred_old = predict_old(test_data)
pred_new = predict_new(test_data)

print(f"Ancienne version : {pred_old} T CO2")
print(f"Nouvelle version : {pred_new} T CO2")
print(f"Différence : {abs(pred_new - pred_old):.2f} T")
```

**Test 2 : Avec le dashboard**
- Remplacer `prediction_logic.py` par `prediction_logic_v2.py`
- Relancer l'application
- Tester plusieurs prédictions
- Vérifier qu'il n'y a pas d'erreurs

### Étape 6 : Mise en Production

Si **prediction_logic_v2.py** fonctionne correctement :

```bash
# Sauvegarder l'ancienne version
move utils\prediction_logic.py utils\prediction_logic_OLD.py

# Activer la nouvelle
move utils\prediction_logic_v2.py utils\prediction_logic.py

# Relancer le dashboard
python app.py
```

### Étape 7 : Documentation (Optionnel mais Recommandé)

Créer `docs/MODELE_DOCUMENTATION.md` avec :
- Nom du modèle : RandomForest Optimisé
- Hyperparamètres exacts
- Métriques de performance :
  * R² CV : [valeur]
  * R² Test : [valeur]
  * RMSE (log) : [valeur]
  * RMSE (tonnes) : [valeur]
  * MAPE : [valeur]%
- Date d'entraînement
- Nombre de features : 465
- Target transformée : log1p(TotalGHGEmissions)

---

## 🚨 POINTS D'ATTENTION

### 1. Features Catégorielles  
Le pipeline attend probablement les features **APRÈS** one-hot encoding. 

**Deux approches possibles :**

**A) Le pipeline fait le one-hot encoding**
→ Passer les colonnes brutes ('Neighborhood', 'BuildingType', etc.)

**B) Le pipeline attend les features déjà encodées**
→ Faire le one-hot encoding dans `_prepare_raw_features()`

**→ Le script `inspect_model.py` révélera la bonne approche**

### 2. Valeurs par Défaut
Certaines features ne sont pas demandées à l'utilisateur :
- `PropertyGFAParking` → mis à 0
- `SteamUse(kBtu)` → mis à 0
- `Electricity(kWh)` → estimé (gfa * 10)
- `NaturalGas(therms)` → estimé (gfa * 0.5)

Ces approximations peuvent affecter la précision !

### 3. Encodages Statistiques
- `Neighborhood_mean`, `Neighborhood_std`
- `PrimaryPropertyType_mean`, `PrimaryPropertyType_std`

Ces valeurs devraient venir du fichier de statistiques d'entraînement, pas être calculées à la volée !

**Solution :**  
Créer un fichier `utils/encoding_stats.json` avec les valeurs pré-calculées.

---

## 📊 RÉSULTAT ATTENDU

Après correction, la prédiction devrait utiliser :
1. ✅ Le modèle RandomForest optimisé avec les VRAIS hyperparamètres
2. ✅ Le StandardScaler du pipeline (pas de normalisation manuelle)
3. ✅ Les 465 features correctement construites
4. ✅ Les feature importances du modèle réel

**Bonus :** Si vous exposez le Modèle 1 (sans ENERGY STAR), les utilisateurs pourraient comparer :
- Prédiction avec ENERGY STAR connu
- Prédiction sans ENERGY STAR (phase de conception)

---

## 🎓 POUR ALLER PLUS LOIN

### Exposer Plus de Modèles
```python
# Dans constants.py
MODEL_1_PATH = os.path.join(MODELS_DIR, 'pipeline_modele1_best.pkl')
MODEL_2_PATH = os.path.join(MODELS_DIR, 'pipeline_modele2_best.pkl')
MODEL_GB_PATH = os.path.join(MODELS_DIR, 'pipeline_m2_gb_optimized.pkl')

# Dans app.py
# Ajouter un sélecteur pour choisir le modèle
```

### Ajouter les Intervalles de Confiance
Si le RandomForest a `n_estimators`, chaque arbre donne une prédiction.
L'écart-type des prédictions = intervalle de confiance !

```python
# Dans predict()
predictions = [tree.predict(df) for tree in model.estimators_]
mean_pred = np.mean(predictions)
std_pred = np.std(predictions)

return {
    'prediction': mean_pred,
    'confidence_interval': (mean_pred - 2*std_pred, mean_pred + 2*std_pred)
}
```

---

## ✅ CHECKLIST FINALE

- [ ] Packages installés (joblib, scikit-learn, pandas, numpy)
- [ ] Script `inspect_model.py` exécuté avec succès
- [ ] Hyperparamètres documentés dans `ANALYSE_MODELES.md`
- [ ] `model_features.json` vérifié ou créé
- [ ] `prediction_logic_v2.py` testé
- [ ] Ancienne version sauvegardée
- [ ] Nouvelle version activée
- [ ] Dashboard relancé et testé
- [ ] Documentation mise à jour

---

**Date : 2026-02-01**  
**Auteur : Antigravity (Assistant IA)**  
**Projet : Seattle CO2 Dashboard - Alignement Modèles ML**

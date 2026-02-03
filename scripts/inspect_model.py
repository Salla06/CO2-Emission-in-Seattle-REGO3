"""
Script d'inspection du modèle sauvegardé
Analyse la structure du pipeline et extrait les informations clés
"""

import joblib
import json
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'pipeline_modele2_best.pkl'

print("=" * 80)
print("INSPECTION DU MODÈLE SAUVEGARDÉ")
print("=" * 80)

# Charger le modèle
print(f"\n📂 Chargement depuis : {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Type principal
print(f"\n🔍 Type principal : {type(model).__name__}")

# Si c'est un Pipeline
if hasattr(model, 'named_steps'):
    print("\n✅ C'est un Pipeline scikit-learn")
    print(f"   Nombre d'étapes : {len(model.named_steps)}")
    
    for i, (name, step) in enumerate(model.named_steps.items(), 1):
        print(f"\n   [{i}] {name}")
        print(f"       Type : {type(step).__name__}")
        
        # Informations spécifiques selon le type
        if hasattr(step, 'get_params'):
            params = step.get_params()
            # Afficher les paramètres principaux
            important_params = {k: v for k, v in params.items() 
                              if not k.startswith('_') and not callable(v)}
            if important_params:
                print(f"       Paramètres clés :")
                for k, v in list(important_params.items())[:5]:  # Limiter à 5
                    print(f"         • {k}: {v}")
        
        # Si c'est le modèle final
        if hasattr(step, 'feature_importances_'):
            print(f"       ✓ Modèle entraîné détecté")
            print(f"       Nombre de features : {len(step.feature_importances_)}")
        
        if hasattr(step, 'n_features_in_'):
            print(f"       Features attendues : {step.n_features_in_}")

elif hasattr(model, 'predict'):
    print("\n⚠️  Modèle seul (pas de pipeline)")
    print(f"   Type : {type(model).__name__}")
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print("\n   Hyperparamètres :")
        for k, v in list(params.items())[:10]:
            if not callable(v):
                print(f"     • {k}: {v}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"\n   Nombre de features : {len(model.feature_importances_)}")
        # Top 10 features importantes
        import numpy as np
        top_indices = np.argsort(model.feature_importances_)[-10:][::-1]
        print("\n   Top 10 features importantes :")
        for idx in top_indices:
            print(f"     Feature {idx}: {model.feature_importances_[idx]:.4f}")

# Vérifier si model_features.json existe
features_path = BASE_DIR / 'utils' / 'model_features.json'
if features_path.exists():
    print(f"\n✅ Fichier model_features.json trouvé")
    with open(features_path, 'r') as f:
        features = json.load(f)
    print(f"   Nombre de features : {len(features)}")
    print(f"   Premières features : {features[:5]}")
else:
    print(f"\n⚠️  Fichier model_features.json INTROUVABLE")
    print(f"   Attendu à : {features_path}")

# Test de prédiction simple
print("\n" + "=" * 80)
print("TEST DE PRÉDICTION")
print("=" * 80)

try:
    # Créer un exemple minimal
    import pandas as pd
    import numpy as np
    
    # Si c'est un pipeline, on peut tester avec des features brutes
    test_data = pd.DataFrame([{
        'Latitude': 47.6097,
        'Longitude': -122.3338,
        'PropertyGFATotal': 50000,
        'NumberofFloors': 5,
        'YearBuilt': 2000,
        'ENERGYSTARScore': 75
    }])
    
    print(f"\n📊 Test avec données minimales...")
    print(f"   Shape des données test : {test_data.shape}")
    
    # Tenter une prédiction
    pred = model.predict(test_data)
    print(f"\n✅ Prédiction réussie : {pred[0]:.4f}")
    print(f"   (valeur log transformée)")
    
    # Inverse transform
    pred_real = np.expm1(pred[0])
    print(f"   Prédiction CO2 réelle : {pred_real:.2f} tonnes")
    
except Exception as e:
    print(f"\n❌ Erreur lors de la prédiction : {e}")
    print(f"   Type d'erreur : {type(e).__name__}")
    print("\n   Le modèle attend probablement TOUTES les features.")
    print("   Consultez model_features.json pour la liste complète.")

print("\n" + "=" * 80)
print("FIN DE L'INSPECTION")
print("=" * 80)

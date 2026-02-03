"""
Script d'analyse du modèle - Version simplifiée
Fonctionne même sans scikit-learn installé
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("ANALYSE DU PIPELINE DE PRÉDICTION")
print("=" * 80)

# Chemins
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'pipeline_modele2_best.pkl'
FEATURES_PATH = BASE_DIR / 'utils' / 'model_features.json'

print(f"\n📂 Répertoire de base : {BASE_DIR}")
print(f"📂 Chemin modèle : {MODEL_PATH}")
print(f"📂 Chemin features : {FEATURES_PATH}")

# Vérifier l'existence des fichiers
print("\n" + "=" * 80)
print("VÉRIFICATION DES FICHIERS")
print("=" * 80)

if MODEL_PATH.exists():
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Modèle trouvé : {MODEL_PATH.name}")
    print(f"   Taille : {size_mb:.2f} MB")
else:
    print(f"❌ Modèle INTROUVABLE : {MODEL_PATH}")
    sys.exit(1)

if FEATURES_PATH.exists():
    print(f"✅ Fichier features trouvé : {FEATURES_PATH.name}")
else:
    print(f"⚠️  Fichier features INTROUVABLE : {FEATURES_PATH}")

# Tenter d'importer joblib
print("\n" + "=" * 80)
print("CHARGEMENT DU MODÈLE")
print("=" * 80)

try:
    import joblib
    print("✅ Module joblib disponible")
    
    # Charger le modèle
    print(f"\n📥 Chargement du modèle...")
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès")
    
    # Analyser le type
    print(f"\n🔍 Type principal : {type(model).__name__}")
    
    # Si c'est un Pipeline
    if hasattr(model, 'named_steps'):
        print("\n✅ C'EST UN PIPELINE SCIKIT-LEARN")
        print(f"   Nombre d'étapes : {len(model.named_steps)}")
        
        for i, (name, step) in enumerate(model.named_steps.items(), 1):
            print(f"\n   Étape {i} : {name}")
            print(f"   Type : {type(step).__name__}")
            
            # Paramètres du modèle
            if hasattr(step, 'get_params'):
                params = step.get_params()
                important = {k: v for k, v in params.items() 
                           if not k.startswith('_') and not callable(v)}
                
                if name == 'model' and important:
                    print("\n   🎯 HYPERPARAMÈTRES DU MODÈLE FINAL :")
                    for k, v in sorted(important.items())[:10]:
                        print(f"      • {k}: {v}")
            
            # Features
            if hasattr(step, 'n_features_in_'):
                print(f"   Features attendues : {step.n_features_in_}")
            
            if hasattr(step, 'feature_importances_'):
                print(f"   ✓ Modèle entraîné (RF/GB)")
                print(f"   Nombre de features : {len(step.feature_importances_)}")
                
                # Top 5 features
                import numpy as np
                top_idx = np.argsort(step.feature_importances_)[-5:][::-1]
                print("\n   🏆 Top 5 features importantes :")
                for idx in top_idx:
                    print(f"      {idx}: {step.feature_importances_[idx]:.4f}")
    
    elif hasattr(model, 'predict'):
        print("\n⚠️  MODÈLE STANDALONE (pas de pipeline)")
        print(f"   Type : {type(model).__name__}")
        
        if hasattr(model, 'n_features_in_'):
            print(f"   Features attendues : {model.n_features_in_}")

except ImportError:
    print("❌ Module 'joblib' NON DISPONIBLE")
    print("\n💡 Pour installer joblib :")
    print("   Option 1 : Utilisez un environnement virtuel")
    print("   Option 2 : Installez avec conda si disponible")
    print("   Option 3 : Vérifiez que vous utilisez le bon Python")
    print(f"\n   Python actuel : {sys.executable}")

# Charger model_features.json si disponible
if FEATURES_PATH.exists():
    print("\n" + "=" * 80)
    print("ANALYSE DU FICHIER model_features.json")
    print("=" * 80)
    
    try:
        import json
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        
        print(f"\n✅ Fichier chargé")
        print(f"   Nombre total de features : {len(features)}")
        
        # Analyser le contenu
        categorical_features = [f for f in features if '_' in f and not f.startswith('GFA') 
                               and not f.startswith('Age') and not f.startswith('Size')]
        numeric_features = [f for f in features if f not in categorical_features]
        
        print(f"   Features numériques : {len(numeric_features)}")
        print(f"   Features catégorielles (one-hot) : {len(categorical_features)}")
        
        print("\n📋 Premières 10 features :")
        for f in features[:10]:
            print(f"      • {f}")
        
        # Compter les types de features
        neighborhoods = len([f for f in features if f.startswith('Neighborhood_')])
        property_types = len([f for f in features if 'PropertyType' in f])
        
        if neighborhoods > 0:
            print(f"\n   Quartiers encodés : {neighborhoods}")
        if property_types > 0:
            print(f"   Types de propriété encodés : {property_types}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la lecture : {e}")

# Recommandations finales
print("\n" + "=" * 80)
print("RECOMMANDATIONS")
print("=" * 80)

print("\n1. ✅ MODÈLE TROUVÉ")
print("   Le fichier pipeline_modele2_best.pkl existe")

if FEATURES_PATH.exists():
    print("\n2. ✅ SCHÉMA DES FEATURES TROUVÉ")
    print("   Le fichier model_features.json est disponible")
else:
    print("\n2. ⚠️  SCHÉMA DES FEATURES MANQUANT")
    print("   Créez model_features.json en extrayant les colonnes")
    print("   du DataFrame d'entraînement (après one-hot encoding)")

print("\n3. 📝 PROCHAINE ÉTAPE")
print("   - Installer les dépendances dans un environnement approprié")
print("   - OU utiliser le Python du projet (pas celui d'Inkscape)")
print("   - Vérifier requirements.txt pour les dépendances")

print("\n4. 🔧 POUR ACTIVER LA NOUVELLE VERSION")
print("   Une fois les tests réussis :")
print("   - Sauvegarder : utils\\prediction_logic.py → utils\\prediction_logic_OLD.py")
print("   - Activer : utils\\prediction_logic_v2.py → utils\\prediction_logic.py")

print("\n" + "=" * 80)
print("FIN DE L'ANALYSE")
print("=" * 80)

import joblib
import json
import os
import sys

MODEL_PATH = r"c:\Users\HP\OneDrive\Bureau\Projet Machine Learning\seattle_dashboard\models\pipeline_modele2_best.pkl"
OUTPUT_PATH = r"c:\Users\HP\OneDrive\Bureau\Projet Machine Learning\seattle_dashboard\utils\model_features.json"

try:
    model = joblib.load(MODEL_PATH)
    features = []
    
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_.tolist()
    elif hasattr(model.steps[-1][1], 'feature_names_in_'):
         features = model.steps[-1][1].feature_names_in_.tolist()
    else:
        print("Feature names not found directly.")
        sys.exit(1)
         
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(features, f)
        
    print(f"Features saved to {OUTPUT_PATH}")
    print(f"Total features: {len(features)}")
    
    print("---POTENTIAL_NUMERICS---")
    # Affiche tout ce qui ne ressemble pas à une catégorie encodée standard
    keywords_cat = ['Neighborhood_', 'BuildingType_', 'PrimaryPropertyType_', 
                    'LargestPropertyUseType_', 'Second', 'Third', 'List', 'SteamUse', 'Electricity', 'NaturalGas']
    
    for f in features:
        if not any(k in f for k in keywords_cat):
            print(f)
            
except Exception as e:
    print(f"Error: {e}")

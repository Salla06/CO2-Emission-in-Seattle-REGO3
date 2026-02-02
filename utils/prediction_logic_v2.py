"""
Module de prédiction CO2 pour le dashboard Seattle
Version corrigée utilisant correctement le pipeline scikit-learn
"""

import os
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
from .constants import MODEL_PATH, BUILDING_TYPE_BENCHMARKS, NEIGHBORHOOD_STATS

class Predictor:
    """
    Classe de prédiction utilisant le pipeline optimisé
    Le pipeline contient : StandardScaler + RandomForestRegressor optimisé
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.is_loaded = False
        self._load_resources()

    def _load_resources(self):
        """Charge le modèle et les métadonnées des features"""
        try:
            # Load Model Pipeline
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.is_loaded = True
                print(f"✓ Modèle chargé depuis : {MODEL_PATH}")
                
                # Vérifier si c'est un pipeline
                if hasattr(self.model, 'named_steps'):
                    print("✓ Pipeline scikit-learn détecté")
                    for step_name in self.model.named_steps.keys():
                        print(f"  - {step_name}")
                else:
                    print("⚠️  Modèle standalone (pas de pipeline)")
            else:
                print(f"❌ Modèle introuvable à {MODEL_PATH}")

            # Load Feature Schema
            feat_path = Path(__file__).parent / 'model_features.json'
            if os.path.exists(feat_path):
                with open(feat_path, 'r') as f:
                    self.feature_columns = json.load(f)
                print(f"✓ Schéma features chargé : {len(self.feature_columns)} features")
            else:
                print(f"⚠️  model_features.json introuvable")
                
        except Exception as e:
            print(f"❌ Erreur chargement : {e}")
            self.is_loaded = False

    def _mock_predict(self, gfa, energy_star, prop_type):
        """Fallback heuristique si le modèle n'est pas chargé"""
        base_eui = BUILDING_TYPE_BENCHMARKS.get(prop_type, 150)
        
        # Ajuster selon Energy Star
        factor_es = 1.4 - (energy_star / 100 * 0.8) if energy_star else 1.0
        
        estimated_co2 = (gfa * base_eui * factor_es) / 1000 * 0.05
        return round(max(estimated_co2, 0.1), 2)

    def _prepare_raw_features(self, input_data):
        """
        Prépare les features BRUTES pour le pipeline
        Le pipeline gère lui-même les transformations
        """
        # Extraire les données d'entrée
        gfa = input_data['gfa']
        year_built = input_data['year_built']
        floors = max(input_data['number_of_floors'], 1)
        e_star = input_data.get('energy_star_score', 60)
        lat = input_data['location']['lat']
        lon = input_data['location']['lon']
        neighborhood = input_data.get('neighborhood', '')
        building_type = input_data['building_type']
        
        # Année de référence pour l'âge
        data_year = 2016
        age = data_year - year_built
        
        # Feature Engineering (comme dans le notebook de preprocessing)
        features = {
            # Géolocalisation
            'Latitude': lat,
            'Longitude': lon,
            
            # Bâtiment
            'YearBuilt': year_built,
            'NumberofBuildings': 1,
            'NumberofFloors': floors,
            'PropertyGFATotal': gfa,
            'PropertyGFAParking': 0,  # Pas demandé à l'utilisateur
            'PropertyGFABuilding(s)': gfa,
            
            # Score énergétique
            'ENERGYSTARScore': e_star,
            
            # Consommations (approximations - non disponibles lors de la construction)
            'SteamUse(kBtu)': 0,
            'Electricity(kWh)': gfa * 10,  # Estimation grossière
            'NaturalGas(therms)': gfa * 0.5,  # Estimation grossière
            
            # Type de propriété (principal)
            'LargestPropertyUseType': building_type,
            'LargestPropertyUseTypeGFA': gfa,
            'SecondLargestPropertyUseType': 'Parking',  # Valeur par défaut
            'SecondLargestPropertyUseTypeGFA': 0,
            
            # Variables dérivées
            'GFA_per_floor': gfa / floors,
            'Parking_ratio': 0.0,
            'Building_age_squared': age ** 2,
            'Is_old_building': 1 if age > 30 else 0,
            'Size_floors': gfa * floors,
            'Age_size': age * gfa,
            'Age_floors': age * floors,
            'GFA_sqrt': np.sqrt(gfa),
            'Floors_squared': floors ** 2,
            
            # Variables catégorielles
            'BuildingType': 'NonResidential',  # Valeur par défaut
            'PrimaryPropertyType': building_type,
            'Neighborhood': neighborhood,
            
            # Encodages statistiques par quartier/type
            # (Ces valeurs devraient venir du fichier des statistiques d'entraînement)
            'Neighborhood_mean': NEIGHBORHOOD_STATS.get(neighborhood, {}).get('avg_co2', 115.6),
            'Neighborhood_std': 50.0,  # Valeur par défaut
            'PrimaryPropertyType_mean': BUILDING_TYPE_BENCHMARKS.get(building_type, 115.6),
            'PrimaryPropertyType_std': 80.0,  # Valeur par défaut
        }
        
        return features

    def predict(self, input_data):
        """
        Effectue une prédiction de CO2
        
        Args:
            input_data (dict): Dictionnaire contenant :
                - gfa (float): Surface totale en pieds²
                - year_built (int): Année de construction
                - number_of_floors (int): Nombre d'étages
                - energy_star_score (float, optional): Score Energy Star (0-100)
                - location (dict): {'lat': float, 'lon': float}
                - neighborhood (str): Quartier
                - building_type (str): Type de bâtiment
        
        Returns:
            float: Prédiction d'émissions CO2 en tonnes
        """
        if not self.is_loaded:
            print("⚠️  Modèle non chargé, utilisation du fallback")
            return self._mock_predict(
                input_data['gfa'], 
                input_data.get('energy_star_score'), 
                input_data['building_type']
            )

        try:
            # Préparer les features brutes
            raw_features = self._prepare_raw_features(input_data)
            
            # Créer un DataFrame avec UNE seule ligne
            df = pd.DataFrame([raw_features])
            
            # Si le modèle attend des features spécifiques
            if self.feature_columns:
                # S'assurer que toutes les colonnes attendues sont présentes
                # One-Hot Encoding sera fait automatiquement si le pipeline le contient
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0
                
                # Réordonner selon l'ordre attendu
                df = df[self.feature_columns]
            
            # Prédiction via le pipeline (gère scaling + modèle)
            log_pred = self.model.predict(df)[0]
            
            # Inverse de log1p
            prediction = np.expm1(log_pred)
            
            return round(max(prediction, 0.1), 1)

        except Exception as e:
            print(f"❌ Erreur lors de la prédiction : {e}")
            print(f"   Type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Fallback en cas d'erreur
            return self._mock_predict(
                input_data['gfa'], 
                input_data.get('energy_star_score'), 
                input_data['building_type']
            )


# ============================================================================
# Interface Publique pour app.py
# ============================================================================

predictor_instance = Predictor()

def predict_co2(data):
    """Point d'entrée principal pour les prédictions"""
    return predictor_instance.predict(data)

def get_seattle_metrics():
    """Retourne des statistiques globales de la ville"""
    return {
        'avg_co2': 115.6,
        'total_buildings': 1332
    }

def generate_report_pdf(prediction_data, plots_images):
    """TODO: Génération PDF"""
    print("⚠️  Génération PDF non implémentée")
    return None

def get_feature_importance():
    """Retourne les importances des features du modèle"""
    if not predictor_instance.is_loaded:
        return {
            'features': ['Surface (GFA)', 'Energy Star Score', 'Type de bâtiment', 
                        'Année Construction', 'Nombre Étages'],
            'importance': [0.45, 0.25, 0.15, 0.10, 0.05]
        }
    
    try:
        # Extraire du modèle si c'est un RandomForest
        model = predictor_instance.model
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['model']
        else:
            actual_model = model
            
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
            features = predictor_instance.feature_columns or [f"Feature_{i}" for i in range(len(importances))]
            
            # Top 10
            top_indices = np.argsort(importances)[-10:][::-1]
            
            return {
                'features': [features[i] for i in top_indices],
                'importance': [float(importances[i]) for i in top_indices]
            }
    except:
        pass
    
    # Fallback
    return {
        'features': ['Surface (GFA)', 'Energy Star Score', 'Usage', 'Année', 'Étages'],
        'importance': [0.45, 0.25, 0.15, 0.10, 0.05]
    }

def get_reliability_info(prediction, inputs):
    """Évalue la fiabilité de la prédiction"""
    score = inputs.get('energy_star_score', 50)
    
    if score > 75:
        return {
            'level': 'Haute', 
            'color': 'success', 
            'desc': 'Données cohérentes avec les benchmarks.'
        }
    elif score < 20:
        return {
            'level': 'Basse', 
            'color': 'danger', 
            'desc': 'Score Energy Star très bas, incertitude accrue.'
        }
    
    return {
        'level': 'Moyenne', 
        'color': 'warning', 
        'desc': 'Estimation standard.'
    }

def get_smart_suggestions(inputs):
    """Suggestions d'amélioration énergétique"""
    btype = inputs['building_type']
    suggs = [
        "Vérifier l'isolation thermique (Toiture/Murs)",
        "Optimiser le système CVC (Chauffage/Ventilation)"
    ]
    
    if 'Office' in btype:
        suggs.append("Installer des détecteurs de présence pour l'éclairage")
    if 'Hospital' in btype or 'Hotel' in btype:
        suggs.append("Optimiser la gestion de l'eau chaude sanitaire")
    
    return suggs

def get_decarbonization_recommendations(current_co2, inputs):
    """Recommandations de décarbonation"""
    potential_gain = current_co2 * 0.20  # 20% de réduction potentielle
    
    return [
        {
            'title': 'Rénovation Énergétique Profonde', 
            'gain': f"-{int(potential_gain)} T", 
            'cost': '$$$'
        },
        {
            'title': 'Passage aux LEDs & Smart Building', 
            'gain': f"-{int(current_co2 * 0.05)} T", 
            'cost': '$'
        },
        {
            'title': 'Installation Panneaux Solaires', 
            'gain': f"-{int(current_co2 * 0.15)} T", 
            'cost': '$$'
        }
    ]

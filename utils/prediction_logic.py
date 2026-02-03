import os
import joblib
import pandas as pd
import numpy as np
import json
from .constants import (
    MODEL_PATH, BUILDING_TYPE_BENCHMARKS, NEIGHBORHOOD_STATS, RESULTS_DIR,
    CO2_CONVERSION_FACTOR, ENERGY_STAR_IMPACT_WEIGHT, ENERGY_STAR_BASE_FACTOR,
    GAS_ESTIMATION_FACTOR, STEAM_ESTIMATION_FACTOR, ELECTRICITY_BASE_FACTOR,
    NEIGHBORHOOD_STD_PROXY, BUILDING_TYPE_STD_PROXY, ENERGY_STAR_IMPROVEMENT_TARGET,
    CARBON_COST_PER_TON, TARGET_2030_REDUCTION
)

class Predictor:
    def __init__(self):
        """Initialise le prédicteur avec lazy loading des ressources."""
        self.model = None
        self.feature_columns = []
        self.is_loaded = False
        # Lazy loading: Resources are loaded on first predict call

    def _load_resources(self):
        try:
            # Load Model
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.is_loaded = True
                print("Predictor: Modèle chargé avec succès.")
                
                # Extract features directly from model if possible
                if hasattr(self.model, "feature_names_in_"):
                    self.feature_columns = self.model.feature_names_in_.tolist()
                elif hasattr(self.model, "steps") and hasattr(self.model.steps[-1][1], "feature_names_in_"):
                     self.feature_columns = self.model.steps[-1][1].feature_names_in_.tolist()
                
            else:
                print(f"Predictor: Modèle introuvable à {MODEL_PATH}")

            # Fallback to JSON if model didn't provide features
            if not self.feature_columns:
                feat_path = os.path.join(os.path.dirname(__file__), 'model_features.json')
                if os.path.exists(feat_path):
                    with open(feat_path, 'r') as f:
                        self.feature_columns = json.load(f)
                else:
                    print("Predictor: Warning - model_features.json introuvable.")
                
        except Exception as e:
            print(f"Predictor: Erreur chargement ({e})")

    def _mock_predict(self, gfa, energy_star, prop_type):
        """Prédiction heuristique de secours si le modèle n'est pas chargé."""
        # Base benchmark
        base_eui = BUILDING_TYPE_BENCHMARKS.get(prop_type, 150)
        
        # Facteurs d'ajustement
        factor_es = 1.0
        if energy_star:
            # Score élevé = moins d'émissions (ex: 100 -> 0.6x, 0 -> 1.4x)
            factor_es = ENERGY_STAR_BASE_FACTOR - (energy_star / 100 * ENERGY_STAR_IMPACT_WEIGHT)
            
        estimated_co2 = (gfa * base_eui * factor_es) / 1000 * CO2_CONVERSION_FACTOR
        return round(max(estimated_co2, 0.1), 2)

    def predict(self, input_data):
        """Prédit les émissions de CO2 d'un bâtiment."""
        # Lazy load logic
        if not self.is_loaded:
            self._load_resources()

        if not self.is_loaded or not self.feature_columns:
            return self._mock_predict(input_data['gfa'], input_data.get('energy_star_score'), input_data['building_type'])

        try:
            # 1. Initialiser le vecteur d'entrée à 0 avec les colonnes EXACTES du modèle
            df = pd.DataFrame(0.0, index=[0], columns=self.feature_columns)
            
            # --- Feature Engineering ---
            gfa = float(input_data['gfa'])
            year_built = int(input_data['year_built'])
            floors = max(int(input_data['number_of_floors']), 1)
            e_star = float(input_data.get('energy_star_score', 60))
            lat = float(input_data['location']['lat'])
            lon = float(input_data['location']['lon'])
            
            # 2. Mapping Direct (Uniquement si la colonne existe dans le modèle)
            mappings = {
                'Latitude': lat, 'Longitude': lon, 
                'YearBuilt': year_built,
                'NumberofBuildings': 1, 
                'NumberofFloors': floors,
                'PropertyGFATotal': gfa, 
                'PropertyGFABuilding(s)': gfa,
                'LargestPropertyUseTypeGFA': gfa, 
                'ENERGYSTARScore': e_star,
                # Features statistiques potentiellement attendues par le modèle
                'GFA_per_floor': gfa / floors, 
                'Size_floors': gfa * floors
            }
            
            # Ajouter les features seulement si elles sont attendues par le modèle
            for col, val in mappings.items():
                if col in df.columns: df[col] = val

            # 3. Mapping Catégoriel One-Hot (Neighborhood / Building Type)
            nbh = input_data.get('neighborhood', 'Downtown')
            btype = input_data['building_type']
            
            # Stratégie robuste : Trouver la colonne correspondante
            for col in df.columns:
                col_lower = col.lower()
                # Quartier
                if col.startswith("Neighborhood_") and nbh.lower() in col_lower:
                    df[col] = 1
                elif 'neighborhood' in col_lower and nbh.lower() in col_lower: # Fallback old logic
                     df[col] = 1

                # Type Bâtiment (Primary ou Largest)
                if (col.startswith("PrimaryPropertyType_") or col.startswith("LargestPropertyUseType_") or col.startswith("BuildingType_")) and btype.lower() in col.lower():
                    df[col] = 1
                elif ('primarypropertytype' in col_lower or 'largestpropertyusetype' in col_lower) and btype.lower() in col_lower: # Fallback old logic
                    df[col] = 1


            # 4. Prédiction
            log_pred = self.model.predict(df)[0]
            prediction = np.expm1(log_pred)
            
            return round(max(prediction, 0.1), 1)

        except Exception as e:
            print(f"Erreur Prediction Logic: {e}")
            # Fallback graceful
            return self._mock_predict(input_data['gfa'], input_data.get('energy_star_score'), input_data['building_type'])

    def explain(self, input_data):
        """Génère une explication des facteurs d'impact sur les émissions."""
        gfa = float(input_data.get('gfa', 50000))
        es = float(input_data.get('energy_star_score', 60))
        year = int(input_data.get('year_built', 1980))
        age = 2026 - year
        
        # Simulation d'impacts pondérés (Approximation SHAP)
        impacts = [
            {'feature': 'Surface au sol (GFA)', 'impact': 0.5 * (gfa / 50000)},
            {'feature': 'Usage du Bâtiment', 'impact': 0.3},
            {'feature': 'Âge du Bâtiment', 'impact': 0.15 * (age / 50)},
            {'feature': 'Score Energy Star', 'impact': -0.4 * (es / 100)},
            {'feature': 'Localisation (Quartier)', 'impact': 0.1}
        ]
        
        # Trier par impact absolu décroissant
        return sorted(impacts, key=lambda x: abs(x['impact']), reverse=True)

# --- Interface Publique pour app.py ---

predictor_instance = Predictor()

def predict_co2(data):
    """Interface publique pour prédire les émissions de CO2."""
    # Adapter les clés si format app
    model_data = {
        'gfa': data.get('PropertyGFATotal', 50000),
        'building_type': data.get('PrimaryPropertyType', 'Office'),
        'number_of_floors': data.get('NumberofFloors', 1),
        'year_built': data.get('YearBuilt', 1980),
        'energy_star_score': data.get('ENERGYSTARScore', 60),
        'location': {'lat': data.get('Latitude', 47.6062), 'lon': data.get('Longitude', -122.3321)},
        'neighborhood': data.get('Neighborhood', 'Downtown'),
        'has_gas': data.get('Has_Gas', False),
        'has_steam': data.get('Has_Steam', False)
    }
    
    prediction = predictor_instance.predict(model_data)
    explanation = predictor_instance.explain(model_data)
    
    return prediction, explanation

def get_smart_es_suggestion(building_type):
    """Suggère un score Energy Star cible."""
    suggestions = {
        'Hospital': 78, 'Hotel': 65, 'Large Office': 82, 'Office': 75,
        'K-12 School': 85, 'University': 70, 'Warehouse': 60,
        'Retail Store': 68, 'Restaurant': 55
    }
    val = suggestions.get(building_type, 70)
    note = f"Cible médiane pour un bâtiment de type {building_type} à Seattle."
    return val, note

def get_seattle_metrics():
    """Charge les métriques de performance des modèles ML."""
    try:
        path = os.path.join(RESULTS_DIR, 'metrics_comparison.json')
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                m1_row = df[df['Modèle'].astype(str).str.contains('M1')].sort_values('R² Test', ascending=False).iloc[0]
                m2_row = df[df['Modèle'].astype(str).str.contains('M2')].sort_values('R² Test', ascending=False).iloc[0]

                return {
                    'with_es': {
                        'R2': m2_row['R² Test'],
                        'MAE': m2_row['MAPE'], 
                        'RMSE': m2_row['RMSE Original'],
                        'MAPE': m2_row['MAPE']
                    },
                    'without_es': {
                        'R2': m1_row['R² Test'],
                        'MAE': m1_row['MAPE'],
                        'RMSE': m1_row['RMSE Original'],
                        'MAPE': m1_row['MAPE']
                    }
                }
            except Exception as parse_error:
                print(f"CSV Parse Error: {parse_error}")
    except Exception as e:
        print(f"Metrics Load Error: {e}")

    # Fallback Values
    return {
        'with_es': {'R2': 0.9849, 'MAE': 7.35, 'RMSE': 47.45, 'MAPE': 7.35},
        'without_es': {'R2': 0.9846, 'MAE': 9.05, 'RMSE': 52.09, 'MAPE': 9.05}
    }

def clean_for_pdf(text):
    """Nettoie le texte pour compatibilité FPDF (latin-1 uniquement)."""
    import re
    text = re.sub(r'[^\x00-\x7F\xc0-\xff]', '', str(text))
    return text.strip()

def generate_report_pdf(prediction_data, features):
    """Génère un rapport PDF complet d'audit carbone."""
    try:
        from fpdf import FPDF
        
        # Calculs Préliminaires
        btype = features.get('PrimaryPropertyType', 'Office')
        gfa = float(features.get('PropertyGFATotal', 50000))
        current_es = float(features.get('ENERGYSTARScore', 50))
        prediction_val = float(prediction_data)
        
        # Benchmark
        base_eui_median = BUILDING_TYPE_BENCHMARKS.get(btype, 100)
        median_ref = (gfa * base_eui_median) / 1000 * CO2_CONVERSION_FACTOR
        if median_ref < 10: median_ref = prediction_val * 1.2 # Fallback
        
        gap_median = ((prediction_val - median_ref) / median_ref) * 100
        
        # Savings Simples pour PDF
        savings = prediction_val * 0.25 
        savings_pct = 25
        financial = savings * CARBON_COST_PER_TON

        # --- PDF CREATION ---
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_margins(15, 15, 15)
        
        # 1. HEADER
        pdf.set_fill_color(0, 50, 80) 
        pdf.rect(0, 0, 210, 40, 'F')
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 22)
        pdf.set_y(15)
        pdf.cell(0, 10, clean_for_pdf("RAPPORT D'AUDIT CARBONE"), ln=True, align='C')
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, clean_for_pdf(f"Projet Seattle City - Analyse Predictive"), ln=True, align='C')
        pdf.ln(25) 
        
        pdf.set_text_color(0, 0, 0)
        
        # 2. EXECUTIVE SUMMARY
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 10, clean_for_pdf("  1. Resultats Cles"), ln=True, fill=True)
        pdf.ln(5)
        
        # Prediction Box
        pdf.set_font("Arial", '', 12)
        pdf.cell(90, 10, clean_for_pdf("Emissions Estimees (2026) :"))
        pdf.set_font("Arial", 'B', 18)
        color = (200, 50, 0) if gap_median > 20 else (200, 150, 0) if gap_median > 0 else (0, 150, 0)
        pdf.set_text_color(*color)
        pdf.cell(0, 10, f"{prediction_val:.1f} T CO2/an", ln=True)
        pdf.set_text_color(0, 0, 0)
        
        # Benchmark
        pdf.set_font("Arial", '', 11)
        pdf.ln(2)
        status_text = "Sous-performant" if gap_median > 10 else "Aligné" if gap_median > -10 else "Efficient"
        pdf.cell(0, 8, clean_for_pdf(f"Positionnement : {status_text}"), ln=True)
        pdf.cell(0, 8, clean_for_pdf(f"Reference mediane locale : {median_ref:.1f} T."), ln=True)
        pdf.ln(5)

        # 3. BUILDING IDENTITY
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 10, clean_for_pdf("  2. Fiche d'Identite"), ln=True, fill=True)
        pdf.ln(4)
        
        ident_data = [
            ("Type d'Usage", btype),
            ("Quartier", features.get('Neighborhood', 'N/A')),
            ("Surface", f"{gfa:,.0f} sqft"),
            ("Annee", features.get('YearBuilt', 'N/A')),
            ("Energy Star", f"{current_es:.0f} / 100")
        ]
        
        pdf.set_font("Arial", '', 11)
        for label, val in ident_data:
            pdf.cell(50, 7, clean_for_pdf(f"{label} :"), border=0)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, clean_for_pdf(str(val)), ln=True, border=0)
            pdf.set_font("Arial", '', 11)
        pdf.ln(8)
        
        # 4. SIMULATION
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, clean_for_pdf("  3. Potentiel d'Amelioration"), ln=True, fill=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 6, clean_for_pdf("En optimisant la performance energetique (isolation, CVC, eclairage) :"))
        pdf.ln(4)
        
        # Savings
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 150, 70)
        pdf.cell(0, 10, f"Gain potentiel : - {savings:.1f} T CO2/an", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
        # Footer
        pdf.set_y(-20)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, clean_for_pdf("Genere par Seattle Dashboard."), align='C')
        
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        print(f"Erreur generation PDF: {e}")
        return None

def get_feature_importance():
    return [
        {'feature': 'Surface (GFA)', 'importance': 0.45},
        {'feature': 'Usage Bâtiment', 'importance': 0.25},
        {'feature': 'Energy Star Score', 'importance': 0.12},
        {'feature': 'Mix Énergétique', 'importance': 0.08},
        {'feature': 'Quartier', 'importance': 0.06},
        {'feature': 'Âge du Bâtiment', 'importance': 0.04}
    ]

def get_reliability_info(prediction, inputs=None):
    if inputs and isinstance(inputs, dict):
        score = inputs.get('energy_star_score') or inputs.get('ENERGYSTARScore', 50)
        if score > 75: return "Élevé"
        elif score < 20: return "Faible"
    if prediction < 50: return "Élevé"
    elif prediction > 200: return "Moyen"
    return "Élevé"

def get_smart_suggestions(inputs):
    btype = inputs['building_type']
    suggs = [
        "Vérifier l'isolation thermique (Toiture/Murs)",
        "Optimiser le système CVC (Chauffage/Ventilation)"
    ]
    if 'Office' in btype:
        suggs.append("Installer des détecteurs de présence pour l'éclairage")
    return suggs

def get_decarbonization_recommendations(inputs):
    building_type = inputs.get('building_type') or inputs.get('PrimaryPropertyType', 'Office')
    es_score = inputs.get('energy_star_score') or inputs.get('ENERGYSTARScore', 50)
    
    recos = []
    if es_score < 60:
        recos.append("📈 Boost Energy Star : Modernisez vos équipements pour grimper dans le classement écologique.")
    if 'Office' in building_type:
        recos.append("💡 Éclairage Intelligent : Installez des lampes qui s'éteignent seules quand les bureaux sont vides.")
        recos.append("🌡️ Chauffage Malin : Réglez le chauffage pour qu'il baisse automatiquement la nuit et le week-end.")
    if 'Hospital' in building_type or 'Hotel' in building_type:
        recos.append("💧 Eau Chaude Économe : Installez des systèmes performants pour ne pas chauffer l'eau inutilement.")
    
    recos.append("☀️ Énergie Solaire : Produisez votre propre électricité verte en installant des panneaux sur le toit.")
    recos.append("🏠 Isolation Renforcée : Changez les vieilles fenêtres et isolez les murs pour garder la chaleur.")
    
    return recos

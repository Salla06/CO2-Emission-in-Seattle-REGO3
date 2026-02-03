"""
Tests unitaires pour le module prediction_logic.

Pour exécuter les tests:
    pytest tests/test_prediction_logic.py -v
"""

import pytest
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.prediction_logic import (
    predict_co2, 
    get_smart_es_suggestion, 
    clean_for_pdf,
    get_decarbonization_recommendations
)
from utils.constants import BUILDING_TYPES, NEIGHBORHOODS


class TestPredictCO2:
    """Tests pour la fonction de prédiction principale."""
    
    def test_predict_basic_office(self):
        """Test prédiction basique pour un bureau."""
        data = {
            'PropertyGFATotal': 50000,
            'PrimaryPropertyType': 'Office',
            'YearBuilt': 1980,
            'NumberofFloors': 5,
            'ENERGYSTARScore': 60,
            'Neighborhood': 'Downtown',
            'Latitude': 47.6062,
            'Longitude': -122.3321,
            'Has_Gas': False,
            'Has_Steam': False
        }
        
        prediction, explanation = predict_co2(data)
        
        # Vérifications
        assert isinstance(prediction, (int, float)), "La prédiction doit être un nombre"
        assert prediction > 0, "La prédiction doit être positive"
        assert isinstance(explanation, list), "L'explication doit être une liste"
        assert len(explanation) > 0, "L'explication ne doit pas être vide"
    
    def test_predict_with_high_energy_star(self):
        """Test que Energy Star élevé réduit les émissions."""
        base_data = {
            'PropertyGFATotal': 50000,
            'PrimaryPropertyType': 'Office',
            'YearBuilt': 1980,
            'NumberofFloors': 5,
            'ENERGYSTARScore': 50,
            'Neighborhood': 'Downtown',
            'Latitude': 47.6062,
            'Longitude': -122.3321,
            'Has_Gas': False,
            'Has_Steam': False
        }
        
        high_es_data = base_data.copy()
        high_es_data['ENERGYSTARScore'] = 90
        
        pred_base, _ = predict_co2(base_data)
        pred_high, _ = predict_co2(high_es_data)
        
        # Un score Energy Star plus élevé devrait réduire les émissions
        assert pred_high <= pred_base, "Score Energy Star élevé devrait réduire les émissions"
    
    def test_predict_large_building(self):
        """Test prédiction pour un grand bâtiment."""
        data = {
            'PropertyGFATotal': 500000,  # Grand bâtiment
            'PrimaryPropertyType': 'Large Office',
            'YearBuilt': 2010,
            'NumberofFloors': 20,
            'ENERGYSTARScore': 75,
            'Neighborhood': 'Downtown',
            'Latitude': 47.6062,
            'Longitude': -122.3321,
            'Has_Gas': True,
            'Has_Steam': True
        }
        
        prediction, _ = predict_co2(data)
        
        # Un grand bâtiment devrait avoir des émissions significatives
        assert prediction > 10, "Un grand bâtiment devrait avoir des émissions > 10 T"


class TestSmartESSuggestion:
    """Tests pour les suggestions Energy Star."""
    
    def test_suggestion_hospital(self):
        """Test suggestion pour un hôpital."""
        score, note = get_smart_es_suggestion('Hospital')
        
        assert isinstance(score, int), "Le score doit être un entier"
        assert 0 <= score <= 100, "Le score doit être entre 0 et 100"
        assert isinstance(note, str), "La note doit être une chaîne"
        assert len(note) > 0, "La note ne doit pas être vide"
    
    def test_suggestion_unknown_type(self):
        """Test suggestion pour un type inconnu."""
        score, note = get_smart_es_suggestion('Unknown Building Type')
        
        # Devrait retourner une valeur par défaut
        assert score == 70, "Score par défaut devrait être 70"


class TestCleanForPDF:
    """Tests pour le nettoyage de texte PDF."""
    
    def test_clean_basic_text(self):
        """Test nettoyage texte basique."""
        text = "Rapport d'audit carbone"
        result = clean_for_pdf(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_clean_emoji(self):
        """Test suppression des emojis."""
        text = "Émissions 🌍 CO2 ⚡"
        result = clean_for_pdf(text)
        
        # Les emojis devraient être supprimés
        assert "🌍" not in result
        assert "⚡" not in result
    
    def test_clean_none(self):
        """Test avec None."""
        result = clean_for_pdf(None)
        assert result == ""


class TestDecarbonizationRecommendations:
    """Tests pour les recommandations de décarbonation."""
    
    def test_recommendations_low_es(self):
        """Test recommandations pour faible Energy Star."""
        inputs = {
            'building_type': 'Office',
            'ENERGYSTARScore': 40
        }
        
        recos = get_decarbonization_recommendations(inputs)
        
        assert isinstance(recos, list)
        assert len(recos) > 0
        # Devrait recommander d'améliorer Energy Star
        assert any('Energy Star' in r for r in recos)
    
    def test_recommendations_office(self):
        """Test recommandations spécifiques aux bureaux."""
        inputs = {
            'building_type': 'Office',
            'ENERGYSTARScore': 70
        }
        
        recos = get_decarbonization_recommendations(inputs)
        
        # Devrait contenir des recommandations spécifiques aux bureaux
        assert any('détecteur' in r.lower() or 'cvc' in r.lower() for r in recos)


class TestConstants:
    """Tests pour vérifier la cohérence des constantes."""
    
    def test_building_types_not_empty(self):
        """Vérifier que la liste des types de bâtiments n'est pas vide."""
        assert len(BUILDING_TYPES) > 0
    
    def test_neighborhoods_not_empty(self):
        """Vérifier que la liste des quartiers n'est pas vide."""
        assert len(NEIGHBORHOODS) > 0
    
    def test_building_types_sorted(self):
        """Vérifier que les types de bâtiments sont triés."""
        assert BUILDING_TYPES == sorted(BUILDING_TYPES)
    
    def test_neighborhoods_sorted(self):
        """Vérifier que les quartiers sont triés."""
        assert NEIGHBORHOODS == sorted(NEIGHBORHOODS)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

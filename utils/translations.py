"""
Dictionnaire de traductions FR/EN pour l'application Dash
"""

TRANSLATIONS = {
    'FR': {
        # Navigation
        'title': 'Seattle CO2',
        'subtitle': 'Analyse des Émissions de Bâtiments',
        'nav_insights': 'Vue d\'Ensemble',
        'nav_analysis': 'Analyse Exploratoire',
        'nav_modeling': 'Modélisation',
        'nav_predict': 'Calculateur CO2',
        'nav_sim': 'Simulateur What-If',
        'nav_star': 'Impact Energy Star',
        'nav_2050': 'Benchmark 2050',
        
        # Insights
        'kpi_total_co2': 'Émissions Totales',
        'kpi_avg_co2': 'Émissions Moyennes',
        'kpi_avg_es': 'Score Energy Star Moyen',
        'map_title': 'Répartition Géographique des Émissions',
        
        # Analysis
        'eda_audit': 'Audit de la Donnée',
        'eda_target': 'Analyse de la Cible (Log)',
        'eda_factors': 'Facteurs d\'Influence',
        
        # Modeling
        'model_perf': 'Performance des Modèles',
        'model_compare': 'Comparaison des Algorithmes',
        'feat_imp': 'Importance des Variables',
        
        # Predict
        'pred_title': 'Prédire les Émissions',
        'input_type': 'Type d\'Usage',
        'input_nbh': 'Quartier',
        'input_sqft': 'Surface Totale (sq ft)',
        'input_year': 'Année de Construction',
        'input_floors': 'Nombre d\'Étages',
        'input_es': 'Score ENERGY STAR',
        'input_energy': 'Sources d\'Énergie',
        'btn_predict': 'Lancer la Prédiction',
        'res_predicted': 'Émissions Estimées',
        'res_confidence': 'Indice de Confiance',
        
        # Common
        'loading': 'Chargement...',
        'error': 'Une erreur est survenue',
        'no_data': 'Aucune donnée disponible',
        'download_report': 'Télécharger le Rapport',
    },
    'EN': {
        # Navigation
        'title': 'Seattle CO2',
        'subtitle': 'Building Emissions Analysis',
        'nav_insights': 'Overview',
        'nav_analysis': 'Exploratory Analysis',
        'nav_modeling': 'Modeling',
        'nav_predict': 'CO2 Calculator',
        'nav_sim': 'What-If Simulator',
        'nav_star': 'Energy Star Impact',
        'nav_2050': '2050 Benchmark',
        
        # Insights
        'kpi_total_co2': 'Total Emissions',
        'kpi_avg_co2': 'Average Emissions',
        'kpi_avg_es': 'Avg Energy Star Score',
        'map_title': 'Geographic Distribution of Emissions',
        
        # Analysis
        'eda_audit': 'Data Audit',
        'eda_target': 'Target Analysis (Log)',
        'eda_factors': 'Influence Factors',
        
        # Modeling
        'model_perf': 'Model Performance',
        'model_compare': 'Algorithm Comparison',
        'feat_imp': 'Feature Importance',
        
        # Predict
        'pred_title': 'Predict Emissions',
        'input_type': 'Usage Type',
        'input_nbh': 'Neighborhood',
        'input_sqft': 'Total Area (sq ft)',
        'input_year': 'Year Built',
        'input_floors': 'Number of Floors',
        'input_es': 'ENERGY STAR Score',
        'input_energy': 'Energy Sources',
        'btn_predict': 'Run Prediction',
        'res_predicted': 'Estimated Emissions',
        'res_confidence': 'Confidence Index',
        
        # Common
        'loading': 'Loading...',
        'error': 'An error occurred',
        'no_data': 'No data available',
        'download_report': 'Download Report',
    }
}

def get_trans(lang, key):
    """Récupère une traduction sécurisée"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['FR']).get(key, key)

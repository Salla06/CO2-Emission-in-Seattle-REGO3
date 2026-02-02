
import os
import pandas as pd

# Chemins de fichiers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Fichiers spécifiques
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'pipeline_modele2_best.pkl')

# Listes de référence
NEIGHBORHOODS = sorted([
    'Ballard', 'Central', 'Delridge', 'Downtown', 'East', 
    'Greater Duwamish', 'Lake Union', 'Magnolia / Queen Anne', 
    'North', 'Northeast', 'Northwest', 'Southeast', 'Southwest'
])

BUILDING_TYPES = sorted([
    'Distribution Center', 'Hospital', 'Hotel', 'K-12 School', 
    'Laboratory', 'Large Office', 'Low-Rise Multifamily', 
    'Medical Office', 'Mixed Use Property', 'Office', 
    'Other', 'Refrigerated Warehouse', 'Residence Hall', 
    'Restaurant', 'Retail Store', 'Self-Storage Facility', 
    'Senior Care Community', 'Small- and Mid-Sized Office', 
    'Supermarket / Grocery Store', 'University', 
    'Warehouse', 'Worship Facility'
])

# Mapping normalisation
NEIGHBORHOOD_MAPPING = {
    'BALLARD': 'Ballard', 'CENTRAL': 'Central', 'DELRIDGE': 'Delridge',
    'DELRIDGE NEIGHBORHOODS': 'Delridge', 'DOWNTOWN': 'Downtown', 'EAST': 'East',
    'GREATER DUWAMISH': 'Greater Duwamish', 'LAKE UNION': 'Lake Union',
    'MAGNOLIA / QUEEN ANNE': 'Magnolia / Queen Anne', 'NORTH': 'North',
    'NORTHEAST': 'Northeast', 'NORTHWEST': 'Northwest', 'SOUTHEAST': 'Southeast',
    'SOUTHWEST': 'Southwest'
}

# Statistiques globales (Source: train + test = 1666 bâtiments au total)
# Calculé sur l'ensemble complet (1666 bâtiments)
CITY_WIDE_STATS = {
    "mean_co2": 115.3,  
    "median_co2": 49.6,  
    "avg_energy_star": 68.4,
    "total_buildings": 1666
}

# Benchmarks par type (Source: train + test)
BUILDING_TYPE_BENCHMARKS = {
    'Hospital': 509.3,
    'Laboratory': 438.4,
    'Senior Care Community': 275.0,
    'Hotel': 256.1,
    'University': 232.6,
    'Supermarket / Grocery Store': 222.7,
    'Large Office': 192.3,
    'Restaurant': 188.6,
    'Medical Office': 173.2,
    'Other': 145.6,
    'Mixed Use Property': 135.8,
    'Residence Hall': 101.3,
    'K-12 School': 89.4,
    'Retail Store': 87.3,
    'Distribution Center': 50.4,
    'Warehouse': 42.7,
    'Worship Facility': 42.3,
    'Small- and Mid-Sized Office': 40.5,
    'Refrigerated Warehouse': 37.6,
    'Self-Storage Facility': 24.6,
    'Low-Rise Multifamily': 19.1,
    'Office': 10.8
}

# Statistiques par quartier (Calculé sur 1666 bâtiments)
NEIGHBORHOOD_STATS = {
    'Ballard': {'lat': 47.6718, 'lon': -122.3749, 'avg_co2': 89.4, 'count': 70, 'avg_gfa': 55451, 'avg_year': 1962, 'avg_floors': 2.2},
    'Central': {'lat': 47.6070, 'lon': -122.3051, 'avg_co2': 83.2, 'count': 56, 'avg_gfa': 67126, 'avg_year': 1963, 'avg_floors': 2.6},
    'Delridge': {'lat': 47.5466, 'lon': -122.3569, 'avg_co2': 93.4, 'count': 47, 'avg_gfa': 72148, 'avg_year': 1976, 'avg_floors': 1.7},
    'Downtown': {'lat': 47.6078, 'lon': -122.3362, 'avg_co2': 161.3, 'count': 360, 'avg_gfa': 148500, 'avg_year': 1946, 'avg_floors': 6.5},
    'East': {'lat': 47.6150, 'lon': -122.3197, 'avg_co2': 155.8, 'count': 121, 'avg_gfa': 94969, 'avg_year': 1948, 'avg_floors': 3.8},
    'Greater Duwamish': {'lat': 47.5657, 'lon': -122.3252, 'avg_co2': 68.0, 'count': 346, 'avg_gfa': 67962, 'avg_year': 1961, 'avg_floors': 1.7},
    'Lake Union': {'lat': 47.6356, 'lon': -122.3380, 'avg_co2': 144.1, 'count': 148, 'avg_gfa': 126107, 'avg_year': 1977, 'avg_floors': 4.0},
    'Magnolia / Queen Anne': {'lat': 47.6358, 'lon': -122.3610, 'avg_co2': 106.0, 'count': 151, 'avg_gfa': 84408, 'avg_year': 1971, 'avg_floors': 3.0},
    'North': {'lat': 47.7046, 'lon': -122.3110, 'avg_co2': 83.4, 'count': 67, 'avg_gfa': 74833, 'avg_year': 1973, 'avg_floors': 2.3},
    'Northeast': {'lat': 47.6669, 'lon': -122.3045, 'avg_co2': 111.4, 'count': 127, 'avg_gfa': 80437, 'avg_year': 1965, 'avg_floors': 2.8},
    'Northwest': {'lat': 47.6998, 'lon': -122.3426, 'avg_co2': 116.7, 'count': 86, 'avg_gfa': 66921, 'avg_year': 1971, 'avg_floors': 2.0},
    'Southeast': {'lat': 47.5559, 'lon': -122.2907, 'avg_co2': 102.8, 'count': 46, 'avg_gfa': 67135, 'avg_year': 1978, 'avg_floors': 1.9},
    'Southwest': {'lat': 47.5624, 'lon': -122.3773, 'avg_co2': 111.0, 'count': 41, 'avg_gfa': 55970, 'avg_year': 1960, 'avg_floors': 2.1}
}

# Couleurs du thème
COLORS = {
    'background': '#0f172a',
    'text': '#e2e8f0',
    'primary': '#3b82f6',
    'secondary': '#64748b',
    'accent': '#10b981',
    'danger': '#ef4444',
    'card': '#1e293b',
    'success': '#10b981'
}

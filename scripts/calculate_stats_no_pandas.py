"""
Script de calcul de statistiques robuste (sans dépendance externe comme pandas).
Utilise uniquement le module csv standard.
"""

import csv
import math
import statistics
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
TRAIN_PATH = BASE_DIR / 'data' / 'processed_data' / 'train_processed.csv'
TEST_PATH = BASE_DIR / 'data' / 'processed_data' / 'test_processed.csv'

def load_data(filepath):
    data = []
    headers = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                if len(row) == len(headers):
                    data.append(row)
    except Exception as e:
        print(f"Erreur lecture {filepath}: {e}")
    return headers, data

print("=" * 80)
print("CALCUL STATISTIQUES (MODE SANS PANDAS)")
print("=" * 80)

# Charger les données
h_train, d_train = load_data(TRAIN_PATH)
h_test, d_test = load_data(TEST_PATH)

print(f"Train: {len(d_train)}")
print(f"Test:  {len(d_test)}")

all_data = d_train + d_test
headers = h_train # On suppose que c'est les mêmes

print(f"Total: {len(all_data)}")

# Trouver les index des colonnes
try:
    idx_co2 = headers.index('TotalGHGEmissions')
    idx_estar = headers.index('ENERGYSTARScore')
    idx_nbh = headers.index('Neighborhood')
    idx_lat = headers.index('Latitude')
    idx_lon = headers.index('Longitude')
    idx_gfa = headers.index('PropertyGFATotal')
    idx_year = headers.index('YearBuilt')
    idx_floors = headers.index('NumberofFloors')
    idx_ptype = headers.index('PrimaryPropertyType')
except ValueError as e:
    print(f"Erreur: Colonne manquante {e}")
    print(f"Colonnes dispos: {headers}")
    exit(1)

# Extraire les valeurs pour calcul
co2_values = []
estar_values = []
nbh_data = {}
ptype_data = {}

for row in all_data:
    try:
        co2 = float(row[idx_co2]) if row[idx_co2] else 0.0
        co2_values.append(co2)
        
        # Energy Star (gérer les vides)
        if row[idx_estar] and row[idx_estar].strip():
            estar = float(row[idx_estar])
            estar_values.append(estar)
            
        nbh = row[idx_nbh]
        # Nettoyage Quartier
        nbh_clean = nbh.title()
        if 'Delridge' in nbh_clean: nbh_clean = 'Delridge'
        if 'Magnolia' in nbh_clean: nbh_clean = 'Magnolia / Queen Anne'
        
        if nbh_clean not in nbh_data:
            nbh_data[nbh_clean] = {'co2': [], 'gfa': [], 'year': [], 'floors': [], 'lat': [], 'lon': []}
        
        nbh_data[nbh_clean]['co2'].append(co2)
        nbh_data[nbh_clean]['gfa'].append(float(row[idx_gfa]))
        nbh_data[nbh_clean]['year'].append(float(row[idx_year]))
        nbh_data[nbh_clean]['floors'].append(float(row[idx_floors]))
        nbh_data[nbh_clean]['lat'].append(float(row[idx_lat]))
        nbh_data[nbh_clean]['lon'].append(float(row[idx_lon]))

        ptype = row[idx_ptype]
        if ptype not in ptype_data:
            ptype_data[ptype] = []
        ptype_data[ptype].append(co2)

    except ValueError:
        continue

# Calculs Globaux
mean_co2 = statistics.mean(co2_values)
median_co2 = statistics.median(co2_values)
avg_estar = statistics.mean(estar_values)

print("\nRÉSULTATS GLOBAUX (Train + Test):")
print(f"MEAN_CO2 = {mean_co2:.1f}")
print(f"MEDIAN_CO2 = {median_co2:.1f}")
print(f"AVG_ESTAR = {avg_estar:.1f}")

print("\nSTATS PAR QUARTIER:")
for nbh, data in sorted(nbh_data.items()):
    c_mean = statistics.mean(data['co2'])
    count = len(data['co2'])
    gfa = statistics.mean(data['gfa'])
    year = statistics.mean(data['year'])
    floors = statistics.mean(data['floors'])
    lat = statistics.mean(data['lat'])
    lon = statistics.mean(data['lon'])
    
    # Format dictionnaire pour constants.py
    print(f"'{nbh}': {{'lat': {lat:.4f}, 'lon': {lon:.4f}, 'avg_co2': {c_mean:.1f}, 'count': {count}, 'avg_gfa': {int(gfa)}, 'avg_year': {int(year)}, 'avg_floors': {floors:.1f}}},")

print("\nBENCHMARKS PAR TYPE:")
for ptype, vals in sorted(ptype_data.items()):
    p_mean = statistics.mean(vals)
    print(f"'{ptype}': {p_mean:.1f},")


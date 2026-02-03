"""
Script pour calculer les VRAIES statistiques sur l'ensemble complet (train + test)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
TRAIN_PATH = BASE_DIR / 'data' / 'processed_data' / 'train_processed.csv'
TEST_PATH = BASE_DIR / 'data' / 'processed_data' / 'test_processed.csv'

print("=" * 80)
print("CALCUL DES STATISTIQUES RÉELLES - DATASET COMPLET")
print("=" * 80)

try:
    # Charger les deux datasets
    print(f"\n📂 Chargement de {TRAIN_PATH}")
    df_train = pd.read_csv(TRAIN_PATH)
    print(f"   ✓ Train: {len(df_train)} bâtiments")
    
    print(f"\n📂 Chargement de {TEST_PATH}")
    df_test = pd.read_csv(TEST_PATH)
    print(f"   ✓ Test: {len(df_test)} bâtiments")
    
    # Fusionner
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"\n✅ Dataset complet: {len(df)} bâtiments")
    
    # === KPIs GLOBAUX ===
    print("\n" + "=" * 80)
    print("STATISTIQUES GLOBALES")
    print("=" * 80)
    
    total_buildings = len(df)
    mean_co2 = df['TotalGHGEmissions'].mean()
    median_co2 = df['TotalGHGEmissions'].median()
    
    # Energy Star Score (avec gestion des valeurs manquantes)
    if 'ENERGYSTARScore' in df.columns:
        avg_estar = df['ENERGYSTARScore'].dropna().mean()
    else:
        avg_estar = None
    
    print(f"\n📊 Total bâtiments: {total_buildings}")
    print(f"📊 Moyenne CO2: {mean_co2:.2f} tonnes/an")
    print(f"📊 Médiane CO2: {median_co2:.2f} tonnes/an")
    if avg_estar:
        print(f"📊 Moyenne Energy Star Score: {avg_estar:.1f}")
    
    # === COMPARAISON AVEC L'ACTUEL (train seulement) ===
    print("\n" + "=" * 80)
    print("COMPARAISON : Train seul vs Dataset complet")
    print("=" * 80)
    
    mean_co2_train = df_train['TotalGHGEmissions'].mean()
    avg_estar_train = df_train['ENERGYSTARScore'].dropna().mean() if 'ENERGYSTARScore' in df_train.columns else None
    
    print(f"\nMoyenne CO2:")
    print(f"  Train seul:      {mean_co2_train:.2f} tonnes")
    print(f"  Dataset complet: {mean_co2:.2f} tonnes")
    print(f"  Différence:      {abs(mean_co2 - mean_co2_train):.2f} tonnes ({((mean_co2 - mean_co2_train) / mean_co2_train * 100):+.2f}%)")
    
    if avg_estar and avg_estar_train:
        print(f"\nMoyenne Energy Star:")
        print(f"  Train seul:      {avg_estar_train:.1f}")
        print(f"  Dataset complet: {avg_estar:.1f}")
        print(f"  Différence:      {abs(avg_estar - avg_estar_train):.1f} ({((avg_estar - avg_estar_train) / avg_estar_train * 100):+.2f}%)")
    
    # === STATS PAR QUARTIER (sur dataset complet) ===
    print("\n" + "=" * 80)
    print("STATISTIQUES PAR QUARTIER (dataset complet)")
    print("=" * 80)
    
    # Normalisation
    df['Neighborhood_Clean'] = df['Neighborhood'].str.title()
    mapping = {
        'Delridge Neighborhoods': 'Delridge',
        'Magnolia / Queen Anne': 'Magnolia / Queen Anne'
    }
    df['Neighborhood_Clean'] = df['Neighborhood_Clean'].replace(mapping)
    
    neighborhood_stats = {}
    for name, group in df.groupby('Neighborhood_Clean'):
        stats = {
            'lat': round(group['Latitude'].mean(), 4),
            'lon': round(group['Longitude'].mean(), 4),
            'avg_co2': round(group['TotalGHGEmissions'].mean(), 1),
            'count': int(len(group)),
            'avg_gfa': int(group['PropertyGFATotal'].mean()),
            'avg_year': int(group['YearBuilt'].mean()),
            'avg_floors': round(group['NumberofFloors'].mean(), 1)
        }
        neighborhood_stats[name] = stats
        print(f"\n{name:25} | Count: {stats['count']:4} | Avg CO2: {stats['avg_co2']:6.1f} T")
    
    # === BENCHMARKS PAR TYPE ===
    print("\n" + "=" * 80)
    print("BENCHMARKS PAR TYPE DE BÂTIMENT (dataset complet)")
    print("=" * 80)
    
    benchmarks = df.groupby('PrimaryPropertyType')['TotalGHGEmissions'].mean().round(1).to_dict()
    for prop_type, avg in sorted(benchmarks.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{prop_type:35} | {avg:6.1f} T CO2/an")
    
    # === EXPORT JSON ===
    print("\n" + "=" * 80)
    print("EXPORT JSON POUR constants.py")
    print("=" * 80)
    
    output_data = {
        'CITY_WIDE_STATS': {
            'mean_co2': round(mean_co2, 1),
            'median_co2': round(median_co2, 1),
            'avg_energy_star': round(avg_estar, 1) if avg_estar else 68.5,
            'total_buildings': total_buildings
        },
        'NEIGHBORHOOD_STATS': neighborhood_stats,
        'BUILDING_TYPE_BENCHMARKS': {k: round(v, 1) for k, v in benchmarks.items()}
    }
    
    print("\n---JSON_START---")
    print(json.dumps(output_data, indent=2, default=str))
    print("---JSON_END---")
    
    # Sauvegarder dans un fichier
    output_file = BASE_DIR / 'complete_stats.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✅ Statistiques sauvegardées dans : {output_file}")
    
    # === RECOMMANDATIONS ===
    print("\n" + "=" * 80)
    print("RECOMMANDATIONS POUR constants.py")
    print("=" * 80)
    
    print("\nValeurs à mettre à jour :")
    print(f"  'mean_co2': {round(mean_co2, 1)},  # Actuellement: 115.6")
    print(f"  'median_co2': {round(median_co2, 1)},  # Actuellement: 50.1")
    print(f"  'avg_energy_star': {round(avg_estar, 1) if avg_estar else 68.5},  # Actuellement: 68.5")
    print(f"  'total_buildings': {total_buildings}  # ✓ Déjà corrigé à 1666")

except Exception as e:
    print(f"\n❌ Erreur : {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("FIN DU CALCUL")
print("=" * 80)

import pandas as pd
import numpy as np
import json

# Chemin hardcodé
CSV_PATH = r"c:\Users\HP\OneDrive\Bureau\Projet Machine Learning\seattle_dashboard\data\processed_data\train_processed.csv"

try:
    df = pd.read_csv(CSV_PATH)
    
    # KPIs Globaux
    total_buildings = len(df)
    mean_co2 = df['TotalGHGEmissions'].mean()
    median_co2 = df['TotalGHGEmissions'].median()
    avg_estar = df['ENERGYSTARScore'].mean()

    # --- NORMALISATION ET FUSION DES QUARTIERS ---
    # 1. Mise en titre (BALLARD -> Ballard)
    df['Neighborhood_Clean'] = df['Neighborhood'].str.title()
    
    # 2. Corrections spécifiques
    mapping = {
        'Delridge Neighborhoods': 'Delridge',
        'Magnolia / Queen Anne': 'Magnolia / Queen Anne' # Déjà bon avec title() mais on s'assure
    }
    df['Neighborhood_Clean'] = df['Neighborhood_Clean'].replace(mapping)

    # Stats par quartier (Sur la colonne fusionnée)
    neighborhood_stats = {}
    
    for name, group in df.groupby('Neighborhood_Clean'):
        stats = {
            'lat': group['Latitude'].mean(),
            'lon': group['Longitude'].mean(),
            'avg_co2': group['TotalGHGEmissions'].mean(),
            'count': int(len(group)),
            'avg_gfa': group['PropertyGFATotal'].mean(),
            'avg_year': group['YearBuilt'].mean(),
            'avg_floors': group['NumberofFloors'].mean()
        }
        neighborhood_stats[name] = stats

    # Benchmarks par type
    benchmarks = df.groupby('PrimaryPropertyType')['TotalGHGEmissions'].mean().to_dict()

    print("---JSON_START---")
    print(json.dumps({
        'CITY_WIDE_STATS': {
            'mean_co2': round(mean_co2, 2),
            'median_co2': round(median_co2, 2),
            'avg_energy_star': round(avg_estar, 1),
            'total_buildings': total_buildings
        },
        'NEIGHBORHOOD_STATS': neighborhood_stats, # Maintenant propre et dédoublonnée
        'BUILDING_TYPE_BENCHMARKS': benchmarks
    }, default=str))
    print("---JSON_END---")

except Exception as e:
    print(f"Error: {e}")

# Constants and Labels from Seattle Dataset
BUILDING_TYPES = [
    'Office', 'Hotel', 'Large Office', 'Retail Store', 
    'Non-Refrigerated Warehouse', 'K-12 School', 'Medical Office',
    'Small- and Mid-Sized Office', 'Self-Storage Facility',
    'Distribution Center', 'Senior Care Community'
]

NEIGHBORHOODS = [
    'Downtown', 'Magnolia / Queen Anne', 'Greater Duwamish',
    'Lake Union', 'East', 'Northeast', 'Northwest', 'South',
    'Southeast', 'Central', 'Ballard'
]

# Real Seattle Benchmarking Statistics (approximate based on project reports)
CITY_WIDE_STATS = {
    'total_buildings': 3376,
    'avg_emissions': 119.7,
    'median_emissions': 33.9,
    'total_gfa_sqft': 450000000,
    'avg_energy_star': 67.9
}

# Median Total GHG Emissions by Type (Tonnes/yr) - Sourced from Seattle 2016 Report
BUILDING_TYPE_BENCHMARKS = {
    'Office': 35.0,
    'Hotel': 145.0,
    'Large Office': 180.0,
    'Small- and Mid-Sized Office': 20.0,
    'Retail Store': 25.0,
    'Non-Refrigerated Warehouse': 15.0,
    'K-12 School': 45.0,
    'Senior Care Community': 110.0,
    'Distribution Center': 40.0,
    'Medical Office': 30.0,
    'Self-Storage Facility': 10.0,
    'Other': 40.0
}

# Neighborhood coordinates and typical building stats for interactive mapping
NEIGHBORHOOD_STATS = {
    'Downtown': {'lat': 47.6062, 'lon': -122.3321, 'avg_gfa': 150000, 'avg_year': 1975, 'avg_floors': 12, 'avg_co2': 180, 'count': 850, 'avg_estar': 75},
    'Magnolia / Queen Anne': {'lat': 47.6419, 'lon': -122.3848, 'avg_gfa': 45000, 'avg_year': 1968, 'avg_floors': 3, 'avg_co2': 40, 'count': 320, 'avg_estar': 65},
    'Greater Duwamish': {'lat': 47.5510, 'lon': -122.3330, 'avg_gfa': 85000, 'avg_year': 1972, 'avg_floors': 2, 'avg_co2': 90, 'count': 410, 'avg_estar': 55},
    'Lake Union': {'lat': 47.6254, 'lon': -122.3375, 'avg_gfa': 65000, 'avg_year': 1990, 'avg_floors': 5, 'avg_co2': 75, 'count': 550, 'avg_estar': 80},
    'East': {'lat': 47.6160, 'lon': -122.3150, 'avg_gfa': 40000, 'avg_year': 1955, 'avg_floors': 3, 'avg_co2': 35, 'count': 280, 'avg_estar': 60},
    'Northeast': {'lat': 47.6750, 'lon': -122.3000, 'avg_gfa': 55000, 'avg_year': 1965, 'avg_floors': 3, 'avg_co2': 45, 'count': 300, 'avg_estar': 62},
    'Northwest': {'lat': 47.6950, 'lon': -122.3550, 'avg_gfa': 35000, 'avg_year': 1960, 'avg_floors': 2, 'avg_co2': 30, 'count': 250, 'avg_estar': 58},
    'South': {'lat': 47.5300, 'lon': -122.2850, 'avg_gfa': 70000, 'avg_year': 1982, 'avg_floors': 2, 'avg_co2': 65, 'count': 380, 'avg_estar': 50},
    'Southeast': {'lat': 47.5500, 'lon': -122.2800, 'avg_gfa': 30000, 'avg_year': 1958, 'avg_floors': 2, 'avg_co2': 25, 'count': 220, 'avg_estar': 55},
    'Central': {'lat': 47.6100, 'lon': -122.3000, 'avg_gfa': 45000, 'avg_year': 1945, 'avg_floors': 3, 'avg_co2': 40, 'count': 190, 'avg_estar': 45},
    'Ballard': {'lat': 47.6680, 'lon': -122.3850, 'avg_gfa': 38000, 'avg_year': 1970, 'avg_floors': 3, 'avg_co2': 38, 'count': 180, 'avg_estar': 60}
}

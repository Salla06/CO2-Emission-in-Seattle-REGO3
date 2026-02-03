import json
import base64
import os
from pathlib import Path

# Charger le notebook
notebook_path = "notebooks/04_modelisation_finale.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Créer le dossier de sortie
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Compteur d'images sauvegardées
saved_count = 0
saved_files = []

# Parcourir toutes les cellules
for cell_idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        for output in cell['outputs']:
            # Chercher les images PNG dans les outputs
            if 'data' in output and 'image/png' in output['data']:
                png_data = output['data']['image/png']
                
                # Chercher un nom de fichier dans le code de la cellule
                cell_source = ''.join(cell.get('source', [])).lower()
                
                # Patterns de noms de fichiers - recherche plus flexible
                filename = None
                
                # Résidus (priorité haute car spécifique)
                if 'residual' in cell_source or 'résidu' in cell_source:
                    if 'm1' in cell_source or 'modele1' in cell_source or 'modèle 1' in cell_source:
                        filename = 'residuals_m1.png'
                    elif 'm2' in cell_source or 'modele2' in cell_source or 'modèle 2' in cell_source:
                        filename = 'residuals_m2.png'
                    else:
                        filename = f'residuals_unknown_{cell_idx}.png'
                
                # Comparaisons
                elif 'comparison_rigorous' in cell_source or 'comparaison rigoureuse' in cell_source:
                    filename = 'comparison_rigorous.png'
                elif 'comparison' in cell_source and ('m1' in cell_source and 'm2' in cell_source):
                    filename = 'comparison_m1_m2.png'
                elif 'final_comparison' in cell_source or 'comparaison finale' in cell_source:
                    filename = 'final_comparison_optimized.png'
                
                # Features importance
                elif 'feature' in cell_source or 'importance' in cell_source:
                    if 'm1' in cell_source or 'modele1' in cell_source:
                        filename = 'feature_importance_m1.png'
                    elif 'm2' in cell_source or 'modele2' in cell_source:
                        filename = 'feature_importance_m2.png'
                
                # Prédictions
                elif 'prediction' in cell_source or 'prédiction' in cell_source:
                    if 'm1' in cell_source or 'modele1' in cell_source:
                        filename = 'predictions_m1.png'
                    elif 'm2' in cell_source or 'modele2' in cell_source:
                        filename = 'predictions_m2.png'
                
                # Baseline
                elif 'baseline' in cell_source:
                    if 'm1' in cell_source or 'modele1' in cell_source:
                        filename = 'baseline_m1.png'
                    elif 'm2' in cell_source or 'modele2' in cell_source:
                        filename = 'baseline_m2.png'
                
                # Autres
                elif 'energy_star' in cell_source or 'energystar' in cell_source:
                    filename = 'energy_star_correlation.png'
                elif 'target' in cell_source and 'distribution' in cell_source:
                    filename = 'target_distribution.png'
                
                # Si toujours pas de nom, utiliser un nom générique
                if not filename:
                    filename = f'graph_cell_{cell_idx}.png'
                
                # Éviter les doublons
                if filename in saved_files:
                    base, ext = os.path.splitext(filename)
                    filename = f'{base}_{cell_idx}{ext}'
                
                # Décoder et sauvegarder l'image
                output_path = output_dir / filename
                with open(output_path, 'wb') as img_file:
                    img_file.write(base64.b64decode(png_data))
                print(f"✓ Sauvegardé: {filename} (cellule {cell_idx})")
                saved_count += 1
                saved_files.append(filename)

print(f"\n✅ Total: {saved_count} graphiques extraits et sauvegardés dans {output_dir}")
print(f"\nFichiers sauvegardés:")
for f in sorted(set(saved_files)):
    print(f"  - {f}")

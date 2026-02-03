import json
import base64
import os
from pathlib import Path

# Charger le notebook d'optimisation
notebook_path = "notebooks/05_optimization.ipynb"
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
                
                # Patterns de noms de fichiers
                filename = None
                
                # Comparaison finale/exhaustive
                if 'final_comparison' in cell_source or 'comparaison finale' in cell_source or 'exhaustive' in cell_source:
                    filename = 'final_comparison_optimized.png'
                elif 'comparison' in cell_source and 'optimized' in cell_source:
                    filename = 'final_comparison_optimized.png'
                
                # Si pas de nom trouvé, utiliser un nom générique
                if not filename:
                    filename = f'optimization_graph_cell_{cell_idx}.png'
                
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

print(f"\n✅ Total: {saved_count} graphiques extraits du notebook d'optimisation")
print(f"\nFichiers sauvegardés:")
for f in sorted(set(saved_files)):
    print(f"  - {f}")

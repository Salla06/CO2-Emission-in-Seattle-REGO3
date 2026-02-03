"""
Page Trajectory 2050 - Trajectoire de décarbonation vers 2050
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def layout(lang='FR'):
    """
    Layout de la page Trajectory 2050 avec benchmark et objectifs
    """
    t = get_translations(lang)
    
    return html.Div([
        html.H2(t['2050'], className="mb-4 text-center"),
        
        # Placeholder pour le contenu dynamique
        html.Div(id="dynamic-bench-container", children=[
            html.Div([
                html.I(className="fas fa-chart-line fa-4x mb-4", style={"color": "#00fa9a"}),
                html.H2(t['2050']),
                html.P("Veuillez d'abord effectuer une prédiction dans l'onglet 'Calculateur CO2' pour voir votre trajectoire." if lang=='FR' else "Please first make a prediction in the 'CO2 Calculator' tab to see your trajectory.", className="text-muted"),
                dbc.Button(t['predict'], href="/predict", color="success", className="mt-3")
            ], className="text-center py-5 glass-card")
        ], style={"maxWidth": "800px", "margin": "0 auto"})
    ])


def get_translations(lang):
    """Traductions temporaires"""
    if lang == 'FR':
        return {
            '2050': 'Trajectoire 2050',
            'predict': 'Calculateur CO2'
        }
    else:
        return {
            '2050': '2050 Trajectory',
            'predict': 'CO2 Calculator'
        }

"""
Page Simulator - Simulateur What-If pour optimisations énergétiques
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def layout(lang='FR'):
    """
    Layout de la page Simulator avec scénarios d'amélioration
    """
    t = get_translations(lang)
    
    return html.Div([
        html.H1(t['sim_title'], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H4(t['sim_scenario']),
                    html.P("Simulez l'impact d'une amélioration de l'isolation ou des équipements." if lang=='FR' else "Simulate impact of insulation or equipment upgrades.", className="text-muted"),
                    html.Label(t['sim_boost_label'], className="mt-3"),
                    dcc.Slider(id="sim-es-boost", min=0, max=50, step=5, value=20, marks={i: f"+{i}" for i in range(0, 51, 10)}, className="mb-2"),
                    html.Div([
                        html.Div(id="sim-boost-display", style={
                            "backgroundColor": "rgba(0,250,154,0.1)", 
                            "border": "1px solid #00fa9a", 
                            "borderRadius": "5px", 
                            "padding": "5px", 
                            "color": "#00fa9a", 
                            "fontWeight": "bold", 
                            "fontSize": "1.2rem",
                            "display": "inline-block",
                            "minWidth": "100px"
                        }),
                    ], className="text-center mb-4"),
                ], className="glass-card")
            ], width=5),
            dbc.Col([
                dbc.Card([
                    html.H4(t['sim_savings'], className="text-center"),
                    html.Div(id="sim-savings-summary", className="text-center mb-3", style={"fontSize": "1.3rem"}),
                    dcc.Graph(id="sim-comparison-graph")
                ], className="glass-card")
            ], width=7),
        ])
    ])


def get_translations(lang):
    """Traductions temporaires"""
    if lang == 'FR':
        return {
            'sim_title': 'Simulateur What-If',
            'sim_scenario': 'Scénario d\'Amélioration',
            'sim_boost_label': 'Amélioration du Score ENERGY STAR',
            'sim_savings': 'Économies Potentielles'
        }
    else:
        return {
            'sim_title': 'What-If Simulator',
            'sim_scenario': 'Improvement Scenario',
            'sim_boost_label': 'ENERGY STAR Score Improvement',
            'sim_savings': 'Potential Savings'
        }

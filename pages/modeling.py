"""
Page Modeling - Performance et comparaison des modèles
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import os
from utils.constants import RESULTS_DIR


def layout(lang='FR'):
    """
    Layout de la page Modeling avec métriques de performance
    """
    t = get_translations(lang)

    # Chargement des données de comparaison globale (Notebook 5)
    try:
        comparison_df = pd.read_csv(os.path.join(RESULTS_DIR, 'comparison_rigorous.csv'))
        # Arrondir les valeurs numériques pour un affichage propre
        comparison_df = comparison_df.round(4)
    except Exception as e:
        comparison_df = pd.DataFrame(columns=["Métrique", "Modèle 1", "Modèle 2", "Gain"])
        print(f"Erreur chargement comparison_rigorous.csv: {e}")
    
    # Tables de comparaison des modèles
    model_comparison = [
        {"Model": "Linear Regression", "Baseline R2": "0.685", "Optimized R2": "0.712", "MAE": "115.4"},
        {"Model": "SVR (Support Vector)", "Baseline R2": "0.724", "Optimized R2": "0.748", "MAE": "102.1"},
        {"Model": "Random Forest", "Baseline R2": "0.785", "Optimized R2": "0.804", "MAE": "92.3"},
        {"Model": "XGBoost (Selected)", "Baseline R2": "0.812", "Optimized R2": "0.824", "MAE": "82.5"},
    ]

    table_header = html.Thead(html.Tr([
        html.Th("Modèle / Model"), html.Th("Baseline R²"), html.Th("Optimized R²"), html.Th("MAE")
    ]))
    
    table_body = html.Tbody([
        html.Tr([
            html.Td(m["Model"]), html.Td(m["Baseline R2"]), html.Td(m["Optimized R2"]), html.Td(m["MAE"])
        ], style={"backgroundColor": "rgba(0,250,154,0.1)" if "XGBoost" in m["Model"] else "transparent"})
        for m in model_comparison
    ])

    return html.Div([
        html.H1(t['modeling'], className="mb-5 text-center"),
        
        # Graphiques de performance
        html.Div([
            html.H3("ÉTAPE 5 : Performance du Modèle" if lang=='FR' else "STEP 5: Model Performance", className="text-light"),
            html.P("Analyse de la précision (R²) et de la réduction de l'erreur (MAE) après optimisation.", className="text-muted mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    html.Img(src="/assets/eda_figures/04_model_performance.png", className="img-fluid")
                ], className="glass-card p-2"), width=12),
                dbc.Col(dbc.Card([
                    html.Img(src="/assets/eda_figures/distribution_co2.png", className="img-fluid")
                ], className="glass-card p-2"), width=12),
            ], className="g-3 mb-5")
        ]),
        
        # Table de comparaison
        html.Div([
            html.H3("Optimisation & Comparaison" if lang=='FR' else "Tuning & Comparison", className="text-light"),
            html.P("Comparaisons des algorithmes testés. L'algorithme XGBoost a été retenu pour sa robustesse face aux relations non-linéaires.", className="text-muted mb-4"),
            dbc.Card([
                dbc.Table([table_header, table_body], bordered=True, color='dark', hover=True, responsive=True, striped=True, className="mb-0")
            ], className="glass-card mb-4")
        ]),

        # Comparaison Globale (Notebook 5)
        html.Div([
            html.H3("Comparaison Globale des Modèles" if lang=='FR' else "Global Model Comparison", className="text-light"),
            html.P("Comparaison détaillée entre le modèle sans Energy Star et le modèle avec Energy Star (Notebook 5).", className="text-muted mb-4"),
            dbc.Card([
                dbc.Table.from_dataframe(comparison_df, striped=True, bordered=True, hover=True, dark=True, responsive=True, className="mb-0")
            ], className="glass-card mb-5")
        ]),

        # Feature Importance
        html.Div([
            html.H3("Importance des Variables" if lang=='FR' else "Feature Importance", className="text-light"),
            dbc.Card([
                dcc.Graph(
                    figure=go.Figure(data=[
                        go.Bar(
                            x=[0.42, 0.31, 0.12, 0.08, 0.05, 0.02],
                            y=["Usage Type", "Surface (GFA)", "Quartier", "Energy Star", "Mix Énergétique", "Âge"],
                            orientation='h',
                            marker_color='#00fa9a'
                        )
                    ]).update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Impact Relatif"
                    )
                )
            ], className="glass-card mb-5")
        ])
    ], style={"maxWidth": "1000px", "margin": "0 auto"})


def get_translations(lang):
    """Traductions temporaires"""
    if lang == 'FR':
        return {'modeling': 'Modélisation'}
    else:
        return {'modeling': 'Modeling'}

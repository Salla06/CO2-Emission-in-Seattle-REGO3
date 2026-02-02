"""
Page Energy Star - Impact de la certification Energy Star
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


def layout(lang='FR'):
    """
    Layout de la page Energy Star avec métriques d'impact
    """
    t = get_translations(lang)
    
    # Métriques simulées (à remplacer par les vraies données)
    metrics = {
        'with_es': {'R2': 0.824, 'MAE': 82.5, 'RMSE': 0.589},
        'without_es': {'R2': 0.785, 'MAE': 105.2, 'RMSE': 0.642}
    }
    
    importance = [
        {"feature": "PrimaryPropertyType_mean", "importance": 0.42},
        {"feature": "GFA_sqrt", "importance": 0.31},
        {"feature": "Neighborhood_mean", "importance": 0.12},
        {"feature": "ENERGYSTARScore", "importance": 0.08},
        {"feature": "Energy Mix (Steam/Gas)", "importance": 0.05},
        {"feature": "Building_age", "importance": 0.02},
    ]
    
    r2_gain = ((metrics['with_es']['R2'] / metrics['without_es']['R2']) - 1) * 100
    mae_reduction = ((1 - metrics['with_es']['MAE'] / metrics['without_es']['MAE'])) * 100

    return html.Div([
        html.H1(t['star_title'], className="mb-4 text-center"),
        
        # Row 1: Key Metrics (Cards)
        dbc.Row([
            dbc.Col(dbc.Card([
                html.H6("Gain Précision (R²)" if lang=='FR' else "R² Precision Gain"),
                html.H2(f"+{r2_gain:.1f}%", style={"color": "#00fa9a"})
            ], className="glass-card text-center p-3"), width=6),
            dbc.Col(dbc.Card([
                html.H6("Réduction Erreur (MAE)" if lang=='FR' else "MAE Reduction"),
                html.H2(f"-{mae_reduction:.1f}%", style={"color": "#ffd700"})
            ], className="glass-card text-center p-3"), width=6),
        ], className="mb-4"),

        # Row 2: Graph Comparison
        dbc.Row([
            dbc.Col(dbc.Card([
                html.H5("Performance Prédictive" if lang=='FR' else "Predictive Performance"),
                dcc.Graph(figure=go.Figure(data=[
                    go.Bar(name='SANS Energy Star' if lang=='FR' else 'WITHOUT Energy Star', x=['R²', 'Error Scaled'], y=[metrics['without_es']['R2'], metrics['without_es']['MAE']/200], marker_color='#888'),
                    go.Bar(name='AVEC Energy Star' if lang=='FR' else 'WITH Energy Star', x=['R²', 'Error Scaled'], y=[metrics['with_es']['R2'], metrics['with_es']['MAE']/200], marker_color='#00fa9a')
                ]).update_layout(template="plotly_dark", barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300))
            ], className="glass-card"), width=7),
            
            dbc.Col(dbc.Card([
                html.H5("Impact Réel (Tonnes CO2)" if lang=='FR' else "Real Impact (CO2 Tonnes)"),
                html.Div([
                    html.Div([
                        html.Small("Modèle de Base" if lang=='FR' else "Baseline Model", className="text-muted"),
                        html.H4(f"{metrics['without_es']['MAE']} T", className="text-secondary")
                    ], className="mb-3"),
                    html.Div([
                        html.Small("Modèle Optimisé ES" if lang=='FR' else "ES Optimized Model", className="text-muted"),
                        html.H4(f"{metrics['with_es']['MAE']} T", className="text-success")
                    ])
                ], className="py-4 text-center"),
            ], className="glass-card"), width=5),
        ], className="mb-4"),

        # Row 3: Feature Importance
        dbc.Row([
            dbc.Col(dbc.Card([
                html.H5("Hiérarchie des Facteurs d'Impact (Scientific)" if lang=='FR' else "Scientific Impact Factors Hierarchy"),
                dcc.Graph(figure=go.Figure(data=[
                    go.Bar(
                        y=[d['feature'] for d in importance[::-1]], 
                        x=[d['importance'] for d in importance[::-1]], 
                        orientation='h', 
                        marker_color='#00fa9a'
                    )
                ]).update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=10, r=10, t=30, b=10)))
            ], className="glass-card"), width=12),
        ])
    ])


def get_translations(lang):
    """Traductions temporaires"""
    if lang == 'FR':
        return {'star_title': 'Impact ENERGY STAR'}
    else:
        return {'star_title': 'ENERGY STAR Impact'}

"""
Page Insights - Vue d'ensemble des émissions de Seattle
"""
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def layout(lang='FR'):
    """
    Layout de la page Insights avec carte interactive
    """
    # Traductions
    t = get_translations(lang)
    
    # Statistiques globales (à remplacer par les vraies données)
    CITY_STATS = {
        'total_buildings': 3376,
        'avg_emissions': 119.7,
        'avg_energy_star': 67.9
    }
    
    # Données des quartiers (à remplacer par les vraies données)
    NEIGHBORHOODS = [
        'Downtown', 'Magnolia / Queen Anne', 'Greater Duwamish',
        'Lake Union', 'East', 'Northeast', 'Northwest', 'South',
        'Southeast', 'Central', 'Ballard'
    ]
    
    return html.Div([
        html.H1(t['insights'], className="mb-4"),
        
        # KPIs
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div(t['kpi_buildings'], className="stat-label"),
                html.Div(f"{CITY_STATS['total_buildings']}", id="kpi-count", className="stat-value")
            ], className="glass-card"), width=4),
            dbc.Col(dbc.Card([
                html.Div(t['kpi_emissions'], className="stat-label"),
                html.Div(f"{CITY_STATS['avg_emissions']} T", id="kpi-mean-co2", className="stat-value")
            ], className="glass-card"), width=4),
            dbc.Col(dbc.Card([
                html.Div(t['kpi_estar'], className="stat-label"),
                html.Div(f"{CITY_STATS['avg_energy_star']}", id="kpi-mean-estar", className="stat-value")
            ], className="glass-card"), width=4),
        ]),
        
        # Carte interactive
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div([
                    html.Div([
                        html.H4(f"🗺️ {t['chart_dist']}", className="mb-0"),
                        html.Small("Filtrez les quartiers pour une analyse ciblée" if lang=='FR' else "Filter neighborhoods for targeted analysis", className="text-muted")
                    ]),
                    html.Div([
                        dcc.Dropdown(
                            id="map-filter-nbh",
                            options=[{'label': n, 'value': n} for n in NEIGHBORHOODS],
                            multi=True,
                            placeholder="Choisir quartiers..." if lang=='FR' else "Choose neighborhoods...",
                            style={'width': '300px', 'backgroundColor': 'white', 'color': 'black'}
                        )
                    ])
                ], className="mb-3 d-flex justify-content-between align-items-center"),
                html.Div([
                    dcc.Loading(dcc.Graph(id="insights-map", style={"height": "600px"}), type="circle")
                ], style={"position": "relative", "overflow": "hidden", "borderRadius": "10px"}),
                html.Div([
                    html.Span("⬤ Taille : Émissions (Tonnes CO2)", className="me-3 small text-muted"),
                    html.Span("⬤ Couleur : Intensité Carbone", className="small text-muted")
                ], className="text-center mt-2")
            ], className="p-4 glass-card", style={"position": "relative"}), width=12),
        ]),
        
        # Toast for feedback
        dbc.Toast(
            id="map-toast",
            header="Configuration Prête",
            is_open=False,
            dismissable=True,
            duration=4000,
            icon="success",
            style={"position": "fixed", "top": 80, "right": 20, "width": 350, "zIndex": "3000"},
        ),
    ])


def get_translations(lang):
    """Traductions temporaires (à remplacer par le module utils.translations)"""
    if lang == 'FR':
        return {
            'insights': 'Vue d\'Ensemble',
            'kpi_buildings': 'Bâtiments',
            'kpi_emissions': 'Émissions Moyennes',
            'kpi_estar': 'Score Energy Star Moyen',
            'chart_dist': 'Distribution Géographique'
        }
    else:
        return {
            'insights': 'Overview',
            'kpi_buildings': 'Buildings',
            'kpi_emissions': 'Average Emissions',
            'kpi_estar': 'Average Energy Star Score',
            'chart_dist': 'Geographic Distribution'
        }

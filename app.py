import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import os
import json
import warnings

# Suppress sklearn version warnings to prevent console clutter
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Internal imports
from utils.constants import BUILDING_TYPES, NEIGHBORHOODS, CITY_WIDE_STATS, BUILDING_TYPE_BENCHMARKS, NEIGHBORHOOD_STATS
from utils.prediction_logic import (
    predict_co2, get_seattle_metrics, generate_report_pdf, 
    get_feature_importance, get_reliability_info, 
    get_smart_suggestions, get_decarbonization_recommendations,
    get_smart_es_suggestion
)
from utils.translations import TRANSLATIONS

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
server = app.server

# --- COMPONENTS ---

def get_sidebar_content(lang):
    t = TRANSLATIONS[lang]
    return html.Div(id="sidebar-content-toggleable", children=[
        html.Div([
            html.I(className="fas fa-city me-2", style={"fontSize": "1.5rem", "color": "#00fa9a"}),
            html.H2("Seattle Dashboard", className="d-inline", style={"color": "#00fa9a", "fontSize": "1.5rem"})
        ], className="text-center mb-3", style={"marginTop": "60px"}),
        html.P(t['subtitle'], className="text-muted small text-center mb-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-chart-pie me-2"), t['nav_insights']], href="/", active="exact"),
                dbc.NavLink([html.I(className="fas fa-search-plus me-2"), t['nav_analysis']], href="/analysis", active="exact"),
                dbc.NavLink([html.I(className="fas fa-brain me-2"), t['nav_modeling']], href="/modeling", active="exact"),
                dbc.NavLink([html.I(className="fas fa-magic me-2"), t['nav_predict']], href="/predict", active="exact"),
                dbc.NavLink([html.I(className="fas fa-tools me-2"), t['nav_sim']], href="/sim", active="exact"),
                dbc.NavLink([html.I(className="fas fa-star me-2"), t['nav_star']], href="/star", active="exact"),
                dbc.NavLink([html.I(className="fas fa-check-circle me-2"), t['nav_2050']], href="/2050", active="exact"),
            ],
            vertical=True,
            pills=True,
            className="sidebar-nav",
        ),
        html.Div([
            html.P("© Seattle City 2026", className="text-muted tiny text-center"),
        ], style={"position": "absolute", "bottom": "20px", "width": "90%"})
    ])

content = html.Div(id="page-content")

#--- HEADER (Language Switch + Menu Toggle) ---
header_static = html.Div([
    # Menu Toggle Button
    html.Div([
        html.Button([
            html.I(className="fas fa-bars"),
            html.Span(" MENU", style={"marginLeft": "10px", "fontSize": "1rem", "fontWeight": "bold"})
        ],
            id="sidebar-toggle-btn",
            n_clicks=0,
            style={
                "position": "fixed",
                "top": "10px",  # Plus haut pour dégager le titre
                "left": "10px", # Plus à gauche
                "zIndex": "9999",
                "background": "rgba(0, 250, 154, 0.9)",
                "border": "2px solid #fff",
                "borderRadius": "8px",
                "padding": "10px 15px",
                "color": "#000",
                "cursor": "pointer",
                "boxShadow": "0 4px 15px rgba(0,0,0,0.5)",
                "transition": "all 0.3s ease"
            }
        )
    ]),
    # Language Switch
    html.Div([
        html.Span("🌍 FR/EN", style={"fontSize": "0.9rem", "fontWeight": "bold", "color": "#888", "marginRight": "10px"}),
        dcc.RadioItems(
            id='lang-switch',
            options=[{'label': ' FR', 'value': 'FR'}, {'label': ' EN', 'value': 'EN'}],
            value='FR',
            labelStyle={'display': 'inline-block', 'marginRight': '15px', 'color': '#00fa9a', 'fontWeight': 'bold'},
            inputStyle={'marginRight': '5px'}
        )
    ], style={"position": "fixed", "top": "20px", "right": "40px", "zIndex": "2000", "background": "rgba(0,0,0,0.5)", "padding": "10px 20px", "borderRadius": "30px", "backdropFilter": "blur(10px)", "border": "1px solid rgba(255,255,255,0.1)"})
])

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="stored-building-features", storage_type="session", data={
        'PropertyGFATotal': 50000,
        'YearBuilt': 1980,
        'PrimaryPropertyType': 'Office',
        'ENERGYSTARScore': 50,
        'Neighborhood': 'Downtown',
        'NumberofFloors': 1,
        'NumberofBuildings': 1,
        'Latitude': 47.6062,
        'Longitude': -122.3321,
        'Has_Gas': False,
        'Has_Steam': False,
        'Has_Parking': False
    }),
    dcc.Store(id="baseline-prediction", storage_type="session"),
    dcc.Store(id="current-prediction", storage_type="session"),
    dcc.Store(id="sidebar-toggle-stored", data=True, storage_type="local"),
    header_static,
    html.Div([
        # Dynamic Sidebar Content (Fixed, no toggle button)
        html.Div(id="sidebar-info-container")
    ], id="sidebar-container", className="sidebar", style={"zIndex": 5000}), 
    content,
    
    # Modal for EDA Images
    dbc.Modal([
        dbc.ModalHeader(id="eda-modal-header"),
        dbc.ModalBody(html.Img(id="eda-modal-img", className="img-fluid")),
        dbc.ModalFooter(dbc.Button("Fermer", id="eda-modal-close", className="ms-auto"))
    ], id="eda-modal", size="xl", is_open=False),
])

# --- PAGE 1: INSIGHTS (Improvement 1: Real Map) ---
def layout_insights(lang):
    t = TRANSLATIONS[lang]
    # NEIGHBORHOODS est déjà triée et unique dans constants.py
    return html.Div([
        html.H1([html.I(className="fas fa-chart-pie me-3"), t['nav_insights']], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div("Total Bâtiments" if lang=='FR' else "Total Buildings", className="stat-label"),
                html.Div(f"{CITY_WIDE_STATS['total_buildings']}", id="kpi-count", className="stat-value")
            ], className="glass-card"), width=4),
            dbc.Col(dbc.Card([
                html.Div(t['kpi_avg_co2'], className="stat-label"),
                html.Div(f"{CITY_WIDE_STATS['mean_co2']} T", id="kpi-mean-co2", className="stat-value")
            ], className="glass-card"), width=4),
            dbc.Col(dbc.Card([
                html.Div(t['kpi_avg_es'], className="stat-label"),
                html.Div(f"{CITY_WIDE_STATS['avg_energy_star']}", id="kpi-mean-estar", className="stat-value")
            ], className="glass-card"), width=4),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div([
                    html.Div([
                        html.H4(f"🗺️ {t['map_title']}", className="mb-0"),
                        html.Small("Filtrez les quartiers pour une analyse ciblée" if lang=='FR' else "Filter neighborhoods for targeted analysis", className="text-muted")
                    ]),
                    html.Div([
                        dcc.Dropdown(
                            id="map-filter-nbh",
                            options=[{'label': n, 'value': n} for n in sorted(NEIGHBORHOOD_STATS.keys())],
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

# --- PAGE 2: CALCULATEUR (With Scientific Insights) ---
def layout_predict(lang):
    t = TRANSLATIONS[lang]
    return html.Div([
        html.H1([html.I(className="fas fa-magic me-3"), t['pred_title']], className="mb-4 text-center"),
        
        dbc.Row([
            # Sidebar: Inputs
            dbc.Col([
                dbc.Card([
                    html.H4("Caractéristiques" if lang=='FR' else "Detailed Features", className="mb-4"),
                    
                    html.Label(t['input_type']),
                    dcc.Dropdown(id="in-type", options=[{'label': b, 'value': b} for b in BUILDING_TYPES], value='Office', className="mb-3", style={'color': '#333'}),

                    html.Label(t['input_nbh']),
                    dcc.Dropdown(id="in-nbh", options=[{'label': n, 'value': n} for n in sorted(NEIGHBORHOOD_STATS.keys())], value='Downtown', className="mb-3", style={'color': '#333'}),
                    
                    html.Label(t['input_sqft']),
                    dcc.Input(id="in-surface", type="number", value=50000, className="form-control mb-3"),

                    html.Label(t['input_floors']),
                    dcc.Input(id="in-floors", type="number", value=1, min=1, className="form-control mb-3"),
                    
                    html.Label(t['input_year']),
                    dcc.Slider(id="in-year", min=1900, max=2023, step=1, value=1980, marks={i: str(i) for i in range(1900, 2030, 20)}, className="mb-2"),
                    html.Div(id="year-display", className="text-center mb-3", style={"color": "#00fa9a", "fontWeight": "bold"}),

                    html.Div([
                        html.Label(t['input_es']),
                        dbc.Button("💡 Suggérer" if lang=='FR' else "💡 Suggest", id="btn-smart-es", color="link", size="sm", className="p-0 ms-2", style={"textDecoration": "none", "fontSize": "0.8rem"})
                    ], className="d-flex align-items-center mb-1"),
                    dcc.Input(id="in-es", type="number", value=60, min=0, max=100, className="form-control mb-1"),
                    html.Div(id="smart-es-note", className="small text-muted mb-3"),

                    html.Label("Sources d'Énergie", className="mt-3"),
                    dbc.Checklist(
                        options=[
                            {"label": "Gaz Naturel / Natural Gas", "value": "gas"},
                            {"label": "Vapeur (Chauffage urbain) / Steam", "value": "steam"},
                        ],
                        value=[],
                        id="in-energy-sources",
                        switch=True,
                        className="mb-4"
                    ),

                    dbc.Button(t['btn_predict'], id="btn-predict", color="primary", className="w-100", style={"backgroundColor": "#00fa9a", "borderColor": "#00fa9a", "color": "#000", "fontWeight": "bold"}),
                ], className="glass-card mb-4")
            ], width=5),
            
            # Main Area: Results & Reliability
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.H2(t['res_predicted'], className="text-secondary mb-0"),
                        html.Div(id="reliability-badge-container")
                    ], className="d-flex justify-content-between align-items-center mb-0"),
                    html.H1(id="prediction-output", className="display-1 mb-0", style={"color": "#00fa9a", "fontWeight": "bold", "textShadow": "0 0 20px rgba(0,250,154,0.3)"}),
                    html.P("Tonnes CO2 / an", className="text-muted"),
                    
                    html.Hr(),
                    
                    # Decarbonization Recommendations Section
                    html.Div(id="decarbonization-container"),
                    
                    html.Hr(),
                    html.H5("Explicabilité du Modèle (XAI)" if lang=='FR' else "Model Explainability (XAI)", className="mb-3 mt-4"),
                    dcc.Graph(id="xai-graph", style={"height": "300px"}),
                    dbc.Button([html.I(className="fas fa-file-pdf me-2"), "Télécharger Rapport PDF"], id="btn-download-pdf", color="success", className="w-100 mt-4"),
                    dcc.Download(id="download-pdf-obj")
                ], className="glass-card mb-4")
            ], width=7),
        ]),
        
        # Section Batch
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H3("Prédiction par Lots (CSV)" if lang=='FR' else "Batch Prediction (CSV)", className="mb-3"),
                    html.P("Analysez un portefeuille immobilier complet. Format: CSV avec colonnes PropertyGFATotal, PrimaryPropertyType, etc.", className="text-muted"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Déposez un fichier ou ', html.A('cliquez ici')]),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'},
                        multiple=False
                    ),
                    html.Div([
                        dbc.Button("🗑️ Effacer", id="btn-clear-batch", color="danger", size="sm", className="mt-2", style={"display": "none"})
                    ], id="clear-btn-container"),
                    html.Div(id='output-data-upload', className="mt-3")
                ], className="glass-card")
            ], width=12)
        ])
    ], style={"paddingBottom": "100px"})

# --- HELPER: GALLERY UI ---
def build_gallery_ui(title, sections, lang):
    t = TRANSLATIONS[lang]
    gallery_items = []
    
    if title:
        gallery_items.append(html.H2(title, className="mb-4 mt-5 text-center"))
    
    for section in sections:
        section_id = section['id']
        section_title = section['title']
        section_desc = section['desc']
        section_content = section.get('content')
        images = section.get('images', [])
        
        items = [
            html.H3(section_title, className="mt-5 mb-3", id=section_id),
            html.P(section_desc, className="text-muted mb-4")
        ]
        
        if section_content:
            items.append(section_content)
        
        # Images Grid (Respecting "one image per line" if width=12)
        img_rows = []
        for img in images:
            width = img.get('width', 12)
            row = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        html.Div([
                            html.Img(src=img['src'], className="img-fluid rounded", style={"cursor": "pointer", "width": "100%"},
                                     id={'type': 'eda-image', 'index': img['src'], 'title': img['title']}),
                        ], className="overflow-hidden"),
                        html.Div(img['title'], className="p-2 text-center small text-muted")
                    ], className="glass-card mb-4")
                ], width=width)
            ], justify="center")
            img_rows.append(row)
        
        items.extend(img_rows)
        gallery_items.append(html.Div(items))
    
    return html.Div(gallery_items)

# --- PAGE 1: ANALYSE EXPLORATOIRE ---
def layout_analysis(lang):
    t = TRANSLATIONS[lang]
    
    sections = [
        {
            "id": "missing",
            "title": "🔍 Audit des Données Manquantes",
            "desc": "Analyse du taux de complétude des variables avant nettoyage.",
            "images": [{"src": "/assets/eda_figures/auto_missing_values.png", "title": "Bilan des valeurs manquantes", "width": 12}]
        },
        {
            "id": "target_before",
            "title": "📉 Distribution de la Cible (Brute)",
            "desc": "Distribution initiale des émissions de CO2 mettant en évidence les valeurs extrêmes.",
            "images": [{"src": "/assets/eda_figures/01_target_original_distribution.png", "title": "Distribution GHG Emissions", "width": 12}]
        },
        {
            "id": "target_log",
            "title": "📈 Distribution de la Cible (Log)",
            "desc": "Impact de la transformation logarithmique pour normaliser la distribution.",
            "images": [{"src": "/assets/eda_figures/02_target_log_distribution.png", "title": "Distribution Log(CO2)", "width": 12}]
        },
        {
            "id": "buildings",
            "title": "🏢 Caractéristiques Structurelles",
            "desc": "Répartition des bâtiments par type d'usage principal.",
            "images": [{"src": "/assets/eda_figures/auto_BuildingType_distribution.png", "title": "Distribution par Type d'Usage", "width": 12}]
        },
        {
            "id": "neighborhoods",
            "title": "🏘️ Répartition Géographique",
            "desc": "Nombre de bâtiments référencés par quartier.",
            "images": [{"src": "/assets/eda_figures/auto_Neighborhood_distribution.png", "title": "Distribution par Quartier", "width": 12}]
        },
        {
            "id": "surface",
            "title": "📐 Analyse des Surfaces",
            "desc": "Distribution de la surface totale au sol (GFA).",
            "images": [{"src": "/assets/eda_figures/auto_PropertyGFATotal_distribution.png", "title": "Distribution de la Surface", "width": 12}]
        },
        {
            "id": "energy_star",
            "title": "⭐ Impact Energy Star",
            "desc": "Relation entre le score Energy Star et les émissions de gaz à effet de serre.",
            "images": [{"src": "/assets/eda_figures/energy_star_correlation.png", "title": "Corrélation CO2 vs Energy Star Score", "width": 12}]
        }
    ]

    return html.Div([
        html.H1([html.I(className="fas fa-search me-3"), t['nav_analysis']], className="mb-4"),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),
        build_gallery_ui(None, sections, lang)
    ], style={"paddingBottom": "100px"})

# --- PAGE 0.2: MODÉLISATION ---
def layout_modeling(lang):
    t = TRANSLATIONS[lang]
    
    # Données exhaustives issues des notebooks d'optimisation
    model_comparison = [
        {"Model": "M2 - Random Forest", "Détail": "Optimisé (RandomSearch)", "Algo": "Random Forest", "R2": "0.9849", "RMSE": "47.45", "MAPE": "7.35%"},
        {"Model": "M1 - Gradient Boosting", "Détail": "Optimisé (RandomSearch)", "Algo": "Gradient Boosting", "R2": "0.9846", "RMSE": "52.09", "MAPE": "9.05%"},
        {"Model": "M2 - Gradient Boosting", "Détail": "Comparaison", "Algo": "Gradient Boosting", "R2": "0.9844", "RMSE": "48.93", "MAPE": "7.58%"},
        {"Model": "M1 - Random Forest", "Détail": "Comparaison", "Algo": "Random Forest", "R2": "0.9840", "RMSE": "49.94", "MAPE": "7.70%"},
        {"Model": "Modèle 1 (Baseline)", "Détail": "Sans Energy Star", "Algo": "Gradient Boosting", "R2": "0.9846", "RMSE": "52.09", "MAPE": "9.05%"},
    ]

    table_header = html.Thead(html.Tr([
        html.Th("Modèle" if lang=='FR' else "Model"), 
        html.Th("Configuration"), 
        html.Th("Algorithme" if lang=='FR' else "Algorithm"), 
        html.Th("R²"), 
        html.Th("RMSE"), 
        html.Th("MAPE")
    ]))
    
    table_body = html.Tbody([
        html.Tr([
            html.Td(html.Strong(m["Model"])), 
            html.Td(m["Détail"]), 
            html.Td(m["Algo"]), 
            html.Td(m["R2"]), 
            html.Td(m["RMSE"]), 
            html.Td(m["MAPE"])
        ], style={"backgroundColor": "rgba(0,250,154,0.15)" if "M2" in m["Model"] else "transparent", "borderLeft": "4px solid #00fa9a" if "M2" in m["Model"] else "none"})
        for m in model_comparison
    ])

    sections = [
        {
            "id": "model-comparison",
            "title": "📊 Comparaison des Performances",
            "desc": "Analyse comparative des modèles optimisés. Le Gradient Boosting sur le Modèle 2 (avec Energy Star) obtient le meilleur R².",
            "content": dbc.Card([
                dbc.Table([table_header, table_body], bordered=False, hover=True, responsive=True, className="mb-0 text-light")
            ], className="glass-card p-3 mb-4"),
            "images": [
                {"src": "/assets/model_figures/comparison_m1_m2.png", "title": "Évolution des métriques (M1 vs M2)", "width": 12}
            ]
        },
        {
            "id": "baseline-results",
            "title": "📉 Modèles de Base (Benchmarks)",
            "desc": "Performances initiales des différents algorithmes testés avant optimisation.",
            "images": [
                {"src": "/assets/model_figures/baseline_m1.png", "title": "Benchmark - Modèle 1 (Sans Energy Star)", "width": 12},
                {"src": "/assets/model_figures/baseline_m2.png", "title": "Benchmark - Modèle 2 (Avec Energy Star)", "width": 12}
            ]
        },
        {
            "id": "feat-importance",
            "title": "🎯 Importance des Variables",
            "desc": "Quels facteurs influencent le plus les émissions ? La surface totale (GFA) et le type d'usage sont prédominants.",
            "images": [
                {"src": "/assets/model_figures/feature_importance_m1.png", "title": "Importance des Features - Modèle 1", "width": 12},
                {"src": "/assets/model_figures/feature_importance_m2.png", "title": "Importance des Features - Modèle 2", "width": 12}
            ]
        },
        {
            "id": "pred-analysis",
            "title": "📈 Analyse des Prédictions et Résidus",
            "desc": "Validation visuelle de la qualité des prédictions (valeurs réelles vs prédites) et analyse des erreurs (résidus).",
            "images": [
                {"src": "/assets/model_figures/predictions_m1.png", "title": "Prédictions vs Réalité - Modèle 1", "width": 12},
                {"src": "/assets/model_figures/residuals_m1.png", "title": "Distribution des Résidus - Modèle 1", "width": 12},
                {"src": "/assets/model_figures/predictions_m2.png", "title": "Prédictions vs Réalité - Modèle 2", "width": 12},
                {"src": "/assets/model_figures/residuals_m2.png", "title": "Distribution des Résidus - Modèle 2", "width": 12}
            ]
        }
    ]

    return html.Div([
        html.H1([html.I(className="fas fa-microchip me-3"), t['nav_modeling']], className="mb-4 text-center"),
        html.P("Cette section présente l'intégralité des résultats de la phase de modélisation, des benchmarks initiaux aux modèles optimisés finaux.", className="text-center text-muted mb-5"),
        build_gallery_ui(None, sections, lang)
    ], style={"paddingBottom": "100px"})


def build_gallery_ui(title, sections, lang):
    ui_elements = []
    if title:
        ui_elements.append(html.H1(title, className="mb-5 text-center"))
        
    for sec in sections:
        section_content = [
            html.H3(sec['title'], className="text-light"),
            html.P(sec['desc'], className="text-muted mb-4")
        ]
        
        # Affichage du contenu personnalisé s'il existe
        if 'content' in sec:
            section_content.append(html.Div(sec['content'], className="mb-4"))
            
        # Affichage des images si elles existent
        images = sec.get('images', [])
        if images:
            section_content.append(dbc.Row([
                dbc.Col(dbc.Card([
                    html.H6(img.get('title', ''), className="text-center small mb-2"),
                    html.Img(src=img['src'], className="img-fluid rounded zoomable-img", 
                             id={'type': 'eda-image', 'index': img['src'], 'title': img.get('title', 'Image')})
                ], className="glass-card p-2"), width=img.get('width', 12)) for img in images
            ], className="g-3 mb-5"))
            
        ui_elements.append(html.Div(section_content))
        
    return html.Div(ui_elements, style={"maxWidth": "1000px", "margin": "0 auto"})

# --- PAGE 3: WHAT-IF SIMULATOR (DYNAMIC) ---
def layout_sim(lang):
    t = TRANSLATIONS[lang]
    return html.Div([
        html.H1([html.I(className="fas fa-tools me-3"), t['nav_sim']], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H4("Scénario d'Amélioration" if lang=='FR' else "Improvement Scenario"),
                    html.P("Simulez l'impact d'une amélioration de l'isolation ou des équipements." if lang=='FR' else "Simulate impact of insulation or equipment upgrades.", className="text-muted"),
                    html.Label("Gain score Energy Star (+)" if lang=='FR' else "Energy Star score boost (+)", className="mt-3"),
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
                    html.H4("Économies Projetées" if lang=='FR' else "Projected Savings", className="text-center"),
                    html.Div(id="sim-savings-summary", className="text-center mb-3", style={"fontSize": "1.3rem"}),
                    dcc.Graph(id="sim-comparison-graph")
                ], className="glass-card")
            ], width=7),
        ])
    ])

@app.callback(
    [Output("sim-comparison-graph", "figure"), Output("sim-savings-summary", "children")],
    [Input("sim-es-boost", "value"), Input("lang-switch", "value")],
    [State("stored-building-features", "data")]
)
def update_sim_graph(boost, lang, stored_data):
    lang = lang or 'FR'
    t = TRANSLATIONS[lang]
    
    base_features = stored_data or {'PropertyGFATotal': 50000, 'YearBuilt': 1980, 'PrimaryPropertyType': 'Office', 'ENERGYSTARScore': 50}
    
    base_es = float(base_features.get('ENERGYSTARScore', 0))
    if base_es <= 0: base_es = 50 
    
    final_es = min(base_es + boost, 100)
    is_capped = (base_es + boost) > 100
    
    # Val base
    val_base, _ = predict_co2(base_features)
    
    # Val improved
    improved_features = base_features.copy()
    improved_features['ENERGYSTARScore'] = final_es
    val_improved, _ = predict_co2(improved_features)
    
    if boost > 30:
        val_improved *= 0.95 

    savings = val_base - val_improved
    percent = (savings / val_base) * 100 if val_base > 0 else 0
    
    labels = ['Avant', 'Après'] if lang == 'FR' else ['Before', 'After']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels, 
            y=[val_base, val_improved],
            marker_color=['#888', '#00fa9a'],
            text=[f"{val_base:.1f} T", f"{val_improved:.1f} T"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Tonnes CO2 / an")
    
    summary_text = [
        html.Span(f"- {savings:.1f} T ", style={"color": "#00fa9a", "fontWeight": "bold"}),
        html.Span(f"({percent:.1f}%)", style={"color": "#00fa9a", "fontSize": "0.9rem"}),
    ]
    if is_capped:
        summary_text.append(dbc.Alert("⚠️ Score max 100 atteint.", color="warning", className="mt-2 py-1 small"))
    
    return fig, summary_text
       
def layout_star(lang):
    t = TRANSLATIONS[lang]
    metrics = get_seattle_metrics()
    importance = get_feature_importance()
    
    r2_gain = ((metrics['with_es']['R2'] / metrics['without_es']['R2']) - 1) * 100
    mae_reduction = ((1 - metrics['with_es']['MAE'] / metrics['without_es']['MAE'])) * 100

    return html.Div([
        html.H1([html.I(className="fas fa-star me-3"), t['nav_star']], className="mb-4 text-center"),
        # Row 1: Key Metrics
        dbc.Row([
            dbc.Col(dbc.Card([
                html.H6("Gain Précision (R²)" if lang=='FR' else "R² Precision Gain"),
                html.H2(f"+{r2_gain:.2f}%", style={"color": "#00fa9a"})
            ], className="glass-card text-center p-3"), width=6),
            dbc.Col(dbc.Card([
                html.H6("Baisse Erreur Moyenne (MAPE)" if lang=='FR' else "MAPE Reduction"),
                html.H2(f"-{mae_reduction:.2f}%", style={"color": "#ffd700"})
            ], className="glass-card text-center p-3"), width=6),
        ], className="mb-4"),
        # Row 2: Correlation Analysis
        dbc.Row([
            dbc.Col(dbc.Card([
                html.H5("Corrélation : Score vs Émissions" if lang=='FR' else "Correlation: Score vs Emissions"),
                html.P("Le score Energy Star est inversement proportionnel aux émissions de CO2.", className="text-muted small"),
                html.Img(src="/assets/eda_figures/energy_star_correlation.png", className="img-fluid rounded")
            ], className="glass-card p-3 mb-4"), width=12),
        ]),
        # Row 3: Feature Importance
        dbc.Row([
            dbc.Col(dbc.Card([
                html.H5("Poids dans le Modèle 2" if lang=='FR' else "Weight in Model 2"),
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

def layout_benchmark(lang):
    t = TRANSLATIONS[lang]
    return html.Div(id="dynamic-bench-container")

@app.callback(
    Output("dynamic-bench-container", "children"),
    [Input("lang-switch", "value"), Input("url", "pathname")],
    [State("stored-building-features", "data"), State("current-prediction", "data"), State("baseline-prediction", "data")]
)
def update_benchmark_page(lang, pathname, stored_data, last_pred, baseline_val):
    lang = lang or 'FR'
    t = TRANSLATIONS[lang]
    
    if not last_pred:
        return html.Div([
            html.Div([
                html.I(className="fas fa-chart-line fa-4x mb-4", style={"color": "#00fa9a"}),
                html.H2(t['nav_2050']),
                html.P("Veuillez d'abord effectuer une prédiction.", className="text-muted"),
                dbc.Button(t['nav_predict'], href="/predict", color="success", className="mt-3")
            ], className="text-center py-5 glass-card")
        ], style={"maxWidth": "800px", "margin": "0 auto"})

    try:
        curr_val = float(last_pred)
        base_val = float(baseline_val) if baseline_val else curr_val
        b_type = stored_data.get('PrimaryPropertyType', 'Office')
        median_ref = BUILDING_TYPE_BENCHMARKS.get(b_type, 50.0)
        
        years = [2016, 2030, 2040, 2050]
        targets = [median_ref, median_ref * 0.60, median_ref * 0.30, 0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=targets, name="Cible Seattle", line=dict(color='#00fa9a', dash='dash', width=3)))
        fig.add_trace(go.Scatter(x=[2026], y=[curr_val], name="Bâtiment Actuel", marker=dict(color='#ffd700', size=20, symbol='star')))
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          yaxis_title="Tonnes CO2 / an", margin=dict(l=20, r=20, t=20, b=20))

        target_2030 = median_ref * 0.60
        
        deadline = "OPTIMAL" if curr_val <= target_2030 else "VIGILANCE"
        status_color = "#00fa9a" if curr_val <= target_2030 else "#ff4d4d"
        status_icon = "fa-check-circle" if curr_val <= target_2030 else "fa-exclamation-triangle"

        return html.Div([
            html.H2([html.I(className="fas fa-check-circle me-3"), t['nav_2050']], className="mb-4 text-center"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    html.Div([
                        html.H5([html.I(className="fas fa-chart-line me-2"), f"Trajectoire : {b_type}"]),
                        dcc.Graph(figure=fig, config={'displayModeBar': False})
                    ], className="p-3")
                ], className="glass-card"), width=8),
                dbc.Col([
                    dbc.Card([
                         html.H4("Statut 2030", className="text-center mt-3", style={"color": "#fff"}),
                         html.Div([
                            html.I(className=f"fas {status_icon} fa-4x", style={"color": status_color, "marginBottom": "15px"}),
                            html.H3(deadline, style={"color": status_color, "fontWeight": "bold"}),
                            html.Hr(),
                            html.Div(f"Gap : {max(0, curr_val - target_2030):.1f} Tonnes", className="small mt-2 text-muted text-center")
                        ], className="text-center p-3")
                    ], className="glass-card mb-4")
                ], width=4),
            ])
        ])
    except Exception as e:
        return dbc.Alert(f"Erreur d'affichage du benchmark : {str(e)}", color="danger")

# --- GLOBAL CALLBACKS ---

@app.callback(Output("sidebar-info-container", "children"), [Input("lang-switch", "value")])
def update_lang_nav(lang):
    return get_sidebar_content(lang or 'FR')

@app.callback(Output("page-content", "children"), [Input("url", "pathname"), Input("lang-switch", "value")])
def render_page_content(pathname, lang):
    lang = lang or 'FR'
    if pathname == "/": return layout_insights(lang)
    elif pathname == "/analysis": return layout_analysis(lang)
    elif pathname == "/modeling": return layout_modeling(lang)
    elif pathname == "/predict": return layout_predict(lang)
    elif pathname == "/sim": return layout_sim(lang)
    elif pathname == "/star": return layout_star(lang)
    elif pathname == "/2050": return layout_benchmark(lang)
    return html.Div([html.H1("404", className="text-danger"), html.P(f"Chemin {pathname} inconnu.")], className="p-5 text-center")

@app.callback(
    [Output("prediction-output", "children"), 
     Output("reliability-badge-container", "children"),
     Output("decarbonization-container", "children"),
     Output("xai-graph", "figure"),
     Output("stored-building-features", "data", allow_duplicate=True),
     Output("current-prediction", "data"),
     Output("baseline-prediction", "data")],
    [Input("btn-predict", "n_clicks")],
    [State("in-type", "value"), State("in-nbh", "value"), State("in-surface", "value"), 
     State("in-floors", "value"), State("in-year", "value"), State("in-es", "value"), 
     State("in-energy-sources", "value"), State("lang-switch", "value"),
     State("baseline-prediction", "data")],
    prevent_initial_call=True
)
def update_prediction(n_clicks, b_type, nbh, surface, floors, year, es, energy_sources, lang, baseline):
    if not n_clicks: 
        return "", "", "", go.Figure().update_layout(template="plotly_dark"), dash.no_update, dash.no_update, dash.no_update
    
    lang = lang or 'FR'
    es_val = float(es) if es is not None else 60
    
    # Localisation dynamique basée sur le quartier
    nbh_data = NEIGHBORHOOD_STATS.get(nbh, NEIGHBORHOOD_STATS['Downtown'])
    
    features = {
        'PrimaryPropertyType': b_type, 
        'Neighborhood': nbh,
        'PropertyGFATotal': surface, 
        'NumberofFloors': floors,
        'YearBuilt': year, 
        'ENERGYSTARScore': es_val,
        'Latitude': nbh_data['lat'],
        'Longitude': nbh_data['lon'],
        'Has_Gas': "gas" in (energy_sources or []),
        'Has_Steam': "steam" in (energy_sources or [])
    }
    
    val, explanation = predict_co2(features)
    
    # XAI Graph
    df_xai = pd.DataFrame(explanation)
    df_xai['color'] = df_xai['impact'].apply(lambda x: '#ff4d4d' if x > 0 else '#00fa9a')
    
    fig_xai = go.Figure(go.Bar(
        x=df_xai['impact'], y=df_xai['feature'], orientation='h', marker_color=df_xai['color']
    ))
    fig_xai.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          margin=dict(l=0, r=20, t=30, b=20), yaxis=dict(autorange="reversed"))
    
    # Badge de fiabilité dynamique
    rel = get_reliability_info(val, features) 
    badge = dbc.Badge(rel, color="success" if rel=="Élevé" else "warning", className="ms-2")
    
    # Recommandations
    recos = get_decarbonization_recommendations(features)
    decarbon_ui = html.Div([dbc.Card(html.Div(r, className="p-2"), className="mb-2 glass-card") for r in recos])

    new_baseline = val if baseline is None else baseline
    
    return f"{val:.2f}", badge, decarbon_ui, fig_xai, features, val, new_baseline

@app.callback(
    Output("download-pdf-obj", "data"),
    [Input("btn-download-pdf", "n_clicks")],
    [State("stored-building-features", "data"), State("prediction-output", "children")],
    prevent_initial_call=True
)
def download_pdf_report_callback(n_clicks, features, prediction):
    if not n_clicks or not features or not prediction: 
        return dash.no_update
    
    try:
        val = float(prediction)
        pdf_bytes = generate_report_pdf(val, features)
        if pdf_bytes:
            return dcc.send_bytes(pdf_bytes, f"Rapport_Seattle_{features.get('PrimaryPropertyType', 'Batiment')}.pdf")
    except Exception as e:
         print(f"Callback Error PDF: {e}")
    return dash.no_update

@app.callback(
    [Output("in-es", "value"), Output("smart-es-note", "children")],
    [Input("btn-smart-es", "n_clicks")],
    [State("in-type", "value")]
)
def update_smart_es(n_clicks, b_type):
    if not n_clicks: return dash.no_update, ""
    val, note = get_smart_es_suggestion(b_type)
    return val, note

@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents'), Input('btn-clear-batch', 'n_clicks')],
    [State('upload-data', 'filename'), State('lang-switch', 'value')]
)
def update_output(contents, n_clicks, filename, lang):
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'btn-clear-batch' or not contents: return None
    
    content_type, content_string = contents.split(',')
    decoded = base64.decodebytes(content_string.encode('utf-8'))
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Vérification des colonnes minimales
        required = ['PropertyGFATotal', 'PrimaryPropertyType']
        missing = [c for c in required if c not in df.columns]
        if missing:
             return dbc.Alert(f"Colonnes manquantes : {', '.join(missing)}", color="danger")
        
        # Prédictions
        results = []
        for _, row in df.iterrows():
            # Merge with defaults for missing columns
            row_dict = row.to_dict()
            if 'Neighborhood' not in row_dict: row_dict['Neighborhood'] = 'Downtown'
            if 'YearBuilt' not in row_dict: row_dict['YearBuilt'] = 1980
            
            pred, _ = predict_co2(row_dict)
            results.append(pred)
        
        df['CO2_Emissions_Predicted'] = results
        
        # Stats
        avg_co2 = df['CO2_Emissions_Predicted'].mean()
        total_co2 = df['CO2_Emissions_Predicted'].sum()
        max_co2 = df['CO2_Emissions_Predicted'].max()
        
        return html.Div([
            dbc.Alert([
                html.H5("✅ Traitement par lot réussi !"),
                html.P(f"Fichier : {filename} ({len(df)} bâtiments traités)"),
                dbc.Row([
                    dbc.Col([html.Strong("Total : "), f"{total_co2:.1f} T"], width=4),
                    dbc.Col([html.Strong("Moyenne : "), f"{avg_co2:.1f} T"], width=4),
                    dbc.Col([html.Strong("Max : "), f"{max_co2:.1f} T"], width=4),
                ], className="mb-2"),
                dbc.Button([html.I(className="fas fa-file-csv me-2"), "Télécharger les résultats"], 
                           id="btn-download-batch-csv", color="success", size="sm", className="mt-2"),
                dcc.Download(id="download-batch-csv-obj")
            ], color="success", className="glass-card"),
            
            dash_table.DataTable(
                data=df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1e293b', 'color': 'white', 'fontWeight': 'bold'},
                style_cell={'backgroundColor': 'rgba(255,255,255,0.05)', 'color': '#e2e8f0', 'textAlign': 'left'},
                page_size=5
            ),
            # Stocker les résultats complets pour le téléchargement
            dcc.Store(id='batch-results-store', data=df.to_dict('records'))
        ])
    except Exception as e:
        return html.Div([f'Erreur lors du traitement : {str(e)}'], className="text-danger")

@app.callback(
    Output("download-batch-csv-obj", "data"),
    [Input("btn-download-batch-csv", "n_clicks")],
    [State("batch-results-store", "data")],
    prevent_initial_call=True
)
def download_batch_csv(n_clicks, data):
    if not n_clicks or not data: return dash.no_update
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "seattle_batch_results.csv", index=False)

@app.callback(
    Output("btn-clear-batch", "style"),
    [Input("output-data-upload", "children")]
)
def toggle_clear_button(children):
    return {"display": "block"} if children else {"display": "none"}

@app.callback(
    Output("year-display", "children"),
    [Input("in-year", "value")]
)
def update_year_display(year):
    return f"Année : {year}"

@app.callback(
    Output("sim-boost-display", "children"),
    [Input("sim-es-boost", "value")]
)
def update_sim_boost_display(boost):
    return f"+ {boost} points"

@app.callback(
    [Output("eda-modal", "is_open"), Output("eda-modal-img", "src"), Output("eda-modal-header", "children")],
    [Input({'type': 'eda-image', 'index': ALL, 'title': ALL}, 'n_clicks'), Input("eda-modal-close", "n_clicks")],
    [State("eda-modal", "is_open")]
)
def toggle_eda_zoom(n_clicks_list, close_click, is_open):
    ctx = callback_context
    if not ctx.triggered: return is_open, "", ""
    triggered_id = ctx.triggered[0]['prop_id']
    if "eda-modal-close" in triggered_id: return False, "", ""
    if any(n_clicks_list):
        try:
            prop_id_str = triggered_id.split('.')[0]
            info = json.loads(prop_id_str)
            return True, info['index'], info['title']
        except: return False, "", ""
    return is_open, "", ""

@app.callback(
    [Output("stored-building-features", "data", allow_duplicate=True), 
     Output("map-toast", "is_open"), 
     Output("map-toast", "children")],
    [Input("insights-map", "clickData")],
    [State("stored-building-features", "data")],
    prevent_initial_call=True
)
def handle_map_click(clickData, current_data):
    if not clickData: return dash.no_update, False, ""
    neighborhood = clickData['points'][0]['hovertext']
    stats = NEIGHBORHOOD_STATS.get(neighborhood)
    if stats:
        new_data = current_data.copy()
        new_data['Neighborhood'] = neighborhood
        # Mock updates for other fields based on neighborhood stats if available
        return new_data, True, html.P(f"✅ Quartier {neighborhood} sélectionné.")
    return dash.no_update, False, ""

@app.callback(
    [Output("in-nbh", "value"), Output("in-surface", "value"), 
     Output("in-year", "value"), Output("in-floors", "value")],
    [Input("url", "pathname")],
    [State("stored-building-features", "data")]
)
def sync_inputs_with_store(pathname, stored_data):
    if pathname == "/predict" and stored_data:
        return (
            stored_data.get('Neighborhood', 'Downtown'),
            stored_data.get('PropertyGFATotal', 50000),
            stored_data.get('YearBuilt', 1980),
            stored_data.get('NumberofFloors', 1)
        )
    return dash.no_update

@app.callback(
    [Output("kpi-count", "children"),
     Output("kpi-mean-co2", "children"),
     Output("kpi-mean-estar", "children")],
    [Input("map-filter-nbh", "value")]
)
def update_kpis(selected_nbh):
    if not selected_nbh:
        return (
            f"{CITY_WIDE_STATS['total_buildings']}",
            f"{CITY_WIDE_STATS['mean_co2']} T",
            f"{CITY_WIDE_STATS['avg_energy_star']}"
        )
    
    selected_stats = [NEIGHBORHOOD_STATS[n] for n in selected_nbh if n in NEIGHBORHOOD_STATS]
    if not selected_stats: return "0", "0 T", "0"

    count = sum(s['count'] for s in selected_stats)
    total_co2 = sum(s['avg_co2'] * s['count'] for s in selected_stats)
    # Mock aggregation for Energy Star as it's not in neighborhood stats
    avg_estar = 65 

    avg_co2 = total_co2 / count if count else 0

    return f"{int(count)}", f"{avg_co2:.1f} T", f"{int(avg_estar)}"

@app.callback(
    Output("insights-map", "figure"),
    [Input("map-filter-nbh", "value"), Input("lang-switch", "value")]
)
def update_insights_map(selected_nbh, lang):
    lang = lang or 'FR'
    rows = []
    for name, stats in NEIGHBORHOOD_STATS.items():
        if not selected_nbh or name in selected_nbh:
            rows.append({
                'Neighborhood': name,
                'lat': stats['lat'], 'lon': stats['lon'],
                'emissions': stats['avg_co2']
            })
    df_map = pd.DataFrame(rows)
    
    if df_map.empty: return go.Figure().update_layout(template="plotly_dark", title="Aucune donnée")

    fig_map = px.scatter_map(df_map, lat="lat", lon="lon", size="emissions", color="emissions",
                               hover_name="Neighborhood", 
                               color_continuous_scale=px.colors.sequential.Viridis, 
                               size_max=35, zoom=10.2, map_style="open-street-map")
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, 
        paper_bgcolor='rgba(0,0,0,0)', 
        coloraxis_colorbar=dict(title="CO2 (T)", bgcolor='rgba(0,0,0,0.4)')
    )
    return fig_map


# --- SIDEBAR TOGGLE CALLBACK ---
@app.callback(
    [Output("sidebar-container", "style"), 
     Output("page-content", "style"),
     Output("sidebar-toggle-stored", "data")],
    [Input("sidebar-toggle-btn", "n_clicks")],
    [State("sidebar-toggle-stored", "data")]
)
def toggle_sidebar(n_clicks, is_open):
    # Gestion du premier chargement
    if n_clicks is None:
        return {"zIndex": 5000}, {"marginLeft": "300px", "transition": "margin-left 0.3s ease"}, True

    # Inversion de l'état
    new_state = not is_open
    
    if new_state:
        # Sidebar visible
        return {"zIndex": 5000}, {"marginLeft": "300px", "transition": "margin-left 0.3s ease"}, True
    else:
        # Sidebar cachée
        return {"zIndex": 5000, "transform": "translateX(-100%)", "transition": "transform 0.3s ease"}, {"marginLeft": "0", "transition": "margin-left 0.3s ease"}, False


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True, port=port)

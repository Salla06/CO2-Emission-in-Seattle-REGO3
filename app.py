print("Chargement des bibliothèques en cours... Veuillez patienter.")
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
import traceback
import functools

# Suppress sklearn version warnings to prevent console clutter
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Internal imports
from utils.constants import BUILDING_TYPES, NEIGHBORHOODS, CITY_WIDE_STATS, BUILDING_TYPE_BENCHMARKS, NEIGHBORHOOD_STATS, RESULTS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH
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
    suppress_callback_exceptions=True,
    use_pages=False  # DÉSACTIVER le chargement automatique du dossier pages/
)

# Custom HTML Template to prevent browser auto-translation
app.index_string = '''
<!DOCTYPE html>
<html lang="fr" translate="no">
    <head>
        <meta charset="UTF-8">
        <meta name="google" content="notranslate">
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body class="notranslate">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
server = app.server

# --- COMPONENTS ---

def get_static_sidebar():
    # Initial content in French (default)
    return html.Div(children=[
        html.Div([
            html.I(className="fas fa-city me-2", style={"fontSize": "1.5rem", "color": "#00fa9a"}),
            html.H2("Seattle Dashboard", className="d-inline", style={"color": "#00fa9a", "fontSize": "1.5rem"})
        ], className="text-center mb-3", style={"marginTop": "60px"}),
        html.P("Tableau de bord de performance énergétique", id="sidebar-subtitle", className="text-muted small text-center mb-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-chart-pie me-2"), "Vue d'ensemble"], href="/", active="exact", id="nav-home"),
                dbc.NavLink([html.I(className="fas fa-search-plus me-2"), "Analyse"], href="/analysis", active="exact", id="nav-analysis"),
                dbc.NavLink([html.I(className="fas fa-brain me-2"), "Modélisation"], href="/modeling", active="exact", id="nav-modeling"),
                dbc.NavLink([html.I(className="fas fa-magic me-2"), "Prédiction"], href="/predict", active="exact", id="nav-predict"),
                dbc.NavLink([html.I(className="fas fa-tools me-2"), "Simulateur"], href="/sim", active="exact", id="nav-sim"),
                dbc.NavLink([html.I(className="fas fa-star me-2"), "Energy Star"], href="/star", active="exact", id="nav-star"),
                dbc.NavLink([html.I(className="fas fa-check-circle me-2"), "Objectif 2050"], href="/2050", active="exact", id="nav-2050"),
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
        ),
        html.Div(style={"width": "1px", "height": "20px", "background": "rgba(255,255,255,0.2)", "display": "inline-block", "margin": "0 15px", "verticalAlign": "middle"}),
        dbc.Button(
            html.I(className="fas fa-moon", id="theme-icon"),
            id="theme-switch",
            color="link",
            style={"color": "#ffd700", "fontSize": "1.2rem", "border": "none", "textDecoration": "none", "padding": "0"},
            title="Mode Sombre/Clair"
        )
    ], style={"position": "fixed", "top": "20px", "right": "40px", "zIndex": "2000", "background": "rgba(0,0,0,0.5)", "padding": "10px 20px", "borderRadius": "30px", "backdropFilter": "blur(10px)", "border": "1px solid rgba(255,255,255,0.1)"})
])



app.layout = html.Div(id="theme-wrapper-div", children=[
    dcc.Store(id="theme-store", storage_type="local", data="dark"),
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="last-nav-timestamp", storage_type="memory", data=0),
    html.Div(id="nav-debug-signals", style={"display": "none"}),
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
    dcc.Store(id="sidebar-toggle-stored", data=False, storage_type="local"),
    header_static,
    html.Div([
        # Static Sidebar Content
        get_static_sidebar()
    ], id="sidebar-container", className="sidebar", style={"zIndex": 5000, "backgroundColor": "#161b22"}), 
    content,
    
    # Modal for EDA Images
    dbc.Modal([
        dbc.ModalHeader(id="eda-modal-header"),
        dbc.ModalBody(html.Img(id="eda-modal-img", className="img-fluid")),
        dbc.ModalFooter(dbc.Button("Fermer", id="eda-modal-close", className="ms-auto"))
    ], id="eda-modal", size="xl", is_open=False),
])

# --- PAGE 1: INSIGHTS (Improvement 1: Real Map) ---
# --- PAGE 1: INSIGHTS (Improvement 1: Real Map) ---
def layout_insights(lang, theme='dark'):
    t = TRANSLATIONS[lang]
    template = "plotly_white" if theme == 'light' else "plotly_dark"
    mapbox = "carto-positron" if theme == 'light' else "carto-darkmatter"
    
    # Recalcul des KPIs pour être sûr (ou utiliser CITY_WIDE_STATS)
    df = load_full_data()
    
    # MAP Generation
    if not df.empty and 'Latitude' in df.columns:
        fig_map = px.scatter_map(
            df, 
            lat="Latitude", 
            lon="Longitude", 
            color="TotalGHGEmissions",
            size="TotalGHGEmissions",
            size_max=20, 
            zoom=10.5,
            map_style=mapbox,
            hover_name="Address",
            hover_data=["BuildingType", "YearBuilt", "ENERGYSTARScore"],
            color_continuous_scale='RdBu_r',
            template=template
        )
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'black' if theme=='light' else 'white'}
        )
    else:
        fig_map = go.Figure()

    return html.Div([
        html.H1([html.I(className="fas fa-chart-pie me-3"), t['nav_insights']], className="mb-4 gradient-text"),
        
        # --- LIGNE 1 : KPIs ---
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div("Total Bâtiments" if lang=='FR' else "Total Buildings", className="stat-label"),
                html.Div(f"{len(df)}", id="kpi-count", className="stat-value")
            ], className="glass-card"), width=4),
            dbc.Col(dbc.Card([
                html.Div(t['kpi_avg_co2'], className="stat-label"),
                html.Div(f"{df['TotalGHGEmissions'].mean():.1f} T", id="kpi-mean-co2", className="stat-value")
            ], className="glass-card"), width=4),
            dbc.Col(dbc.Card([
                html.Div(t['kpi_avg_es'], className="stat-label"),
                html.Div(f"{df['ENERGYSTARScore'].mean():.1f}", id="kpi-mean-estar", className="stat-value")
            ], className="glass-card"), width=4),
        ], className="mb-4"),

        # --- LIGNE 2 : CARTE ---
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
                            # Utiliser NEIGHBORHOOD_STATS pour garantir la cohérence avec la carte
                            options=[{'label': n, 'value': n} for n in sorted(NEIGHBORHOOD_STATS.keys())],
                            multi=True,
                            placeholder="Choisir quartiers...",
                            style={'width': '300px', 'color': 'black'}
                        )
                    ])
                ], className="mb-3 d-flex justify-content-between align-items-center"),
                html.Div([
                    dcc.Loading(dcc.Graph(id="insights-map", figure=fig_map, style={"height": "450px"}), type="circle")
                ], style={"position": "relative", "overflow": "hidden", "borderRadius": "10px"}),
            ], className="p-4 glass-card", style={"position": "relative"}), width=12),
        ], className="mb-4"),

        # --- LIGNE 3 : ANALYSES DÉTAILLÉES ---
        dbc.Row([
            dbc.Col(dbc.Card([
                html.H5("🏭 Intensité carbone par Type" if lang=='FR' else "Carbon Intensity by Type", className="mb-3"),
                dcc.Graph(id="insights-bar-chart", style={"height": "300px"}, config={'displayModeBar': False})
            ], className="glass-card p-3"), width=6),
            
            dbc.Col(dbc.Card([
                html.H5("🏢 Répartition du Parc Immobilier" if lang=='FR' else "Building Types Distribution", className="mb-3"),
                dcc.Graph(id="insights-pie-chart", style={"height": "300px"}, config={'displayModeBar': False})
            ], className="glass-card p-3"), width=6),
        ]),

        # Toast
        dbc.Toast(
            id="map-toast",
            header=t['toast_config_ready'],
            is_open=False,
            dismissable=True,
            duration=4000,
            icon="success",
            style={"position": "fixed", "top": 80, "right": 20, "width": 350, "zIndex": "3000"},
        ),
    ], style={"paddingBottom": "100px"})

# --- PAGE 2: CALCULATEUR (With Scientific Insights) ---
def _layout_predict_impl(lang):
    t = TRANSLATIONS[lang]
    return html.Div([
        html.H1([html.I(className="fas fa-magic me-3"), t['pred_title']], className="mb-4 text-center gradient-text"),
        
        dbc.Row([
            # Sidebar: Inputs
            dbc.Col([
                dbc.Card([
                    html.H4("Caractéristiques" if lang=='FR' else "Detailed Features", className="mb-4"),
                    
                    html.Label(t['input_type']),
                    dcc.Dropdown(
                        id="in-type", 
                        options=[{'label': b, 'value': b} for b in BUILDING_TYPES], 
                        value='Office', 
                        className="mb-3",
                        style={'backgroundColor': '#0f172a', 'color': '#00ff00', 'fontWeight': 'bold'}
                    ),

                    html.Label(t['input_nbh']),
                    dcc.Dropdown(
                        id="in-nbh", 
                        options=[{'label': n, 'value': n} for n in sorted(NEIGHBORHOOD_STATS.keys())], 
                        value='Downtown', 
                        className="mb-3",
                        style={'backgroundColor': '#0f172a', 'color': '#00ff00', 'fontWeight': 'bold'}
                    ),
                    
                    html.Label(t['input_sqft']),
                    dcc.Input(id="in-surface", type="number", value=50000, className="form-control mb-3", style={"backgroundColor": "#0f172a", "color": "#00ff00", "fontWeight": "bold", "borderColor": "#334155"}),

                    html.Label(t['input_floors']),
                    dcc.Input(id="in-floors", type="number", value=1, min=1, className="form-control mb-3", style={"backgroundColor": "#0f172a", "color": "#00ff00", "fontWeight": "bold", "borderColor": "#334155"}),
                    
                    html.Label(t['input_year']),
                    dcc.Slider(id="in-year", min=1900, max=2023, step=1, value=1980, marks={i: str(i) for i in range(1900, 2030, 20)}, className="mb-2"),
                    html.Div(id="year-display", className="text-center mb-3", style={"color": "#00fa9a", "fontWeight": "bold"}),

                    html.Div([
                        html.Label(t['input_es']),
                        dbc.Button("💡 Suggérer" if lang=='FR' else "💡 Suggest", id="btn-smart-es", n_clicks=0, color="link", size="sm", className="p-0 ms-2", style={"textDecoration": "none", "fontSize": "0.8rem"})
                    ], className="d-flex align-items-center mb-1"),
                    dcc.Input(id="in-es", type="number", value=60, min=0, max=100, className="form-control mb-1", style={"backgroundColor": "#0f172a", "color": "#00ff00", "fontWeight": "bold", "borderColor": "#334155"}),
                    html.Div(id="smart-es-note", className="small text-muted mb-3"),
                    html.Div(id="es-error-msg", className="text-danger small fw-bold mb-2"),

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

                    dbc.Button(t['btn_predict'], id="btn-predict", n_clicks=0, color="primary", className="w-100", style={"backgroundColor": "#1e293b", "borderColor": "#00fa9a", "borderWidth": "2px", "color": "#00fa9a", "fontWeight": "bold", "fontSize": "1.1rem"}),
                ], className="glass-card mb-4")
            ], width=5),
            
            # Main Area: Results & Reliability
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.H2(t['res_predicted'], className="text-secondary mb-0"),
                        html.Div(id="reliability-badge-container")
                    ], className="d-flex justify-content-between align-items-center mb-0"),
                    dcc.Loading(
                        id="loading-prediction",
                        type="circle",
                        color="#00fa9a",
                        children=[
                            html.H1(id="prediction-output", className="display-1 mb-0", style={"color": "#00fa9a", "fontWeight": "bold", "textShadow": "0 0 20px rgba(0,250,154,0.3)"})
                        ]
                    ),
                    html.P(t['tonnes_co2_year'], className="text-muted"),
                    
                    html.Hr(),
                    
                    # Decarbonization Recommendations Section
                    html.Div(id="decarbonization-container"),
                    
                    html.Hr(),
                    html.Hr(),
                    html.H5(t['pred_xai'], className="mb-3 mt-4"),
                    dcc.Graph(id="xai-graph", style={"height": "300px"}),
                    dbc.Button([html.I(className="fas fa-file-pdf me-2"), t['pred_pdf_btn']], id="btn-download-pdf", n_clicks=0, color="success", className="w-100 mt-4"),
                    dcc.Download(id="download-pdf-obj")
                ], className="glass-card mb-4")
            ], width=7),
        ]),
        
        # Section Batch
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H3(t['pred_batch_title'], className="mb-3"),
                    html.P(t['pred_batch_desc'], className="text-muted"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Déposez un fichier ou ', html.A('cliquez ici')]),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'},
                        multiple=False,
                        max_size=5_000_000  # 5 MB maximum
                    ),
                    html.Div([
                        dbc.Button("🗑️ Effacer", id="btn-clear-batch", color="danger", size="sm", className="mt-2", style={"display": "none"})
                    ], id="clear-btn-container"),
                    html.Div(id='output-data-upload', className="mt-3")
                ], className="glass-card")
            ], width=12)
        ])
    ], style={"paddingBottom": "100px"})

def layout_predict(lang):
    try:
        return _layout_predict_impl(lang)
    except Exception as e:
        return html.Div([
            html.H3("⚠️ Erreur de Chargement (Layout)"),
            html.Hr(),
            dbc.Alert(str(e), color="danger"),
            html.Pre(traceback.format_exc(), style={"backgroundColor": "#333", "color": "#f88", "padding": "20px", "borderRadius": "5px"})
        ], className="container mt-5")

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
    
    # Audit Statique (Notebook)
    sections = [
        {
            "id": "missing",
            "title": t['eda_title_missing'],
            "desc": t['eda_desc_missing'],
            "images": [{"src": "/assets/eda_figures/auto_missing_values.png", "title": "Bilan des valeurs manquantes", "width": 12}]
        },
        {
            "id": "target_before",
            "title": t['eda_title_dist_raw'],
            "desc": t['eda_desc_dist_raw'],
            "images": [{"src": "/assets/eda_figures/01_target_original_distribution.png", "title": "Distribution GHG Emissions", "width": 12}]
        },
        {
            "id": "target_log",
            "title": t['eda_title_dist_log'],
            "desc": t['eda_desc_dist_log'],
            "images": [{"src": "/assets/eda_figures/02_target_log_distribution.png", "title": "Distribution Log(CO2)", "width": 12}]
        },
        {
            "id": "buildings",
            "title": t['eda_title_struct'],
            "desc": t['eda_desc_struct'],
            "images": [{"src": "/assets/eda_figures/auto_BuildingType_distribution.png", "title": "Distribution par Type d'Usage", "width": 12}]
        },
        {
            "id": "neighborhoods",
            "title": t['eda_title_geo'],
            "desc": t['eda_desc_geo'],
            "images": [{"src": "/assets/eda_figures/auto_Neighborhood_distribution.png", "title": "Distribution par Quartier", "width": 12}]
        },
        {
            "id": "surface",
            "title": t['eda_title_surf'],
            "desc": t['eda_desc_surf'],
            "images": [{"src": "/assets/eda_figures/auto_PropertyGFATotal_distribution.png", "title": "Distribution de la Surface", "width": 12}]
        },
        {
            "id": "energy_star",
            "title": t['eda_title_es'],
            "desc": t['eda_desc_es'],
            "images": [{"src": "/assets/eda_figures/energy_star_correlation.png", "title": "Corrélation CO2 vs Energy Star Score", "width": 12}]
        }
    ]

    # Chargement unique des données pour l'analyse
    df_full = load_full_data()
    
    # 1. Matrice de Corrélation (Calculée une fois)
    numeric_cols = ['totalghgemissions', 'siteenergyuse(kbtu)', 'propertygfatotal', 'yearbuilt', 'numberoffloors', 'energystarscore']
    valid_cols = [c for c in df_full.columns if c.lower() in numeric_cols]
    
    if not valid_cols:
        corr_matrix = pd.DataFrame()
        fig_corr = go.Figure()
    else:
        corr_matrix = df_full[valid_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale='RdBu_r', 
            aspect="auto",
            title=t['eda_corr_title']
        )
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return html.Div([
        html.H1([html.I(className="fas fa-search me-3"), t['nav_analysis']], className="mb-4"),
        
        # --- SECTION 1: RAPPORT STATIQUE (Notebook Reflet) ---
        # html.H3("1. Audit Initial (Notebook EDA)", className="mt-5 mb-3 text-info"), # SUPPRIMÉ
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),
        build_gallery_ui(None, sections, lang),
        
        # --- SECTION 2: EXPLORATION INTERACTIVE (Améliorations) ---
        html.H3("2. Exploration Interactive des Données", className="mt-5 mb-3 text-success"),
        html.P("Explorez l'ensemble du jeu de données (Train + Test) via des graphiques dynamiques.", className="text-muted mb-4"),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),

        # LIGNE 1 : Matrice de Corrélation
        dbc.Card([
            dbc.CardHeader("🔥 2.1 Matrice de Corrélation", className="fw-bold"),
            dbc.CardBody([
                dcc.Graph(figure=fig_corr, config={'displayModeBar': False})
            ])
        ], className="mb-5 glass-card"),

        # LIGNE 2 : Explorateur Scatter Plot
        dbc.Card([
            dbc.CardHeader("📈 2.2 Explorateur de Relations (Scatter Plot)", className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Axe X"),
                        dcc.Dropdown(
                            id='scatter-x',
                            options=[{'label': c, 'value': c} for c in df_full.select_dtypes(include=np.number).columns],
                            value='PropertyGFATotal',
                            className="text-dark"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Axe Y"),
                        dcc.Dropdown(
                            id='scatter-y',
                            options=[{'label': c, 'value': c} for c in df_full.select_dtypes(include=np.number).columns],
                            value='TotalGHGEmissions',
                            className="text-dark"
                        )
                    ], width=6)
                ], className="mb-3"),
                dcc.Graph(id='analysis-scatter-graph')
            ])
        ], className="mb-5 glass-card"),

        # LIGNE 3 : Analyse des Distributions (Boxplot)
        dbc.Card([
            dbc.CardHeader("📦 2.3 Analyse des Distributions & Outliers", className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Variable Numérique (Y)"),
                        dcc.Dropdown(
                            id='boxplot-y',
                            options=[{'label': c, 'value': c} for c in df_full.select_dtypes(include=np.number).columns],
                            value='TotalGHGEmissions',
                            className="text-dark"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Variable Catégorielle (X)"),
                        dcc.Dropdown(
                            id='boxplot-x',
                            options=[{'label': c, 'value': c} for c in df_full.select_dtypes(include='object').columns if df_full[c].nunique() < 50],
                            value='BuildingType',
                            className="text-dark"
                        )
                    ], width=6)
                ], className="mb-3"),
                dcc.Graph(id='analysis-boxplot-graph')
            ])
        ], className="mb-5 glass-card"),

        # LIGNE 4 : Statistiques Descriptives
        dbc.Card([
            dbc.CardHeader("🔢 2.4 Statistiques Descriptives Détaillées", className="fw-bold"),
            dbc.CardBody([
                html.Div(id='analysis-stats-table')
            ])
        ], className="mb-5 glass-card"),

    ], style={"paddingBottom": "100px"})

# --- Helper pour charger les données ---
@functools.lru_cache(maxsize=1)
def load_full_data():
    try:
        # Check files existence
        if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
            print("Erreur: Fichiers de données introuvables.")
            return pd.DataFrame()
            
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        df_test = pd.read_csv(TEST_DATA_PATH)
        
        # Concaténation propre
        df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
        
        # Nettoyage des colonnes dupliquées immédiatement
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Nettoyage de la colonne Neighborhood pour éviter les doublons dans le filtre
        if 'Neighborhood' in df.columns:
            # Standardisation : Title Case et suppression des espaces
            df['Neighborhood'] = df['Neighborhood'].astype(str).str.strip().str.title()
            
            # Correction spécifique pour uniformiser avec NEIGHBORHOOD_STATS
            # Par exemple 'DELRIDGE NEIGHBORHOODS' -> 'Delridge'
            replace_map = {
                'Delridge Neighborhoods': 'Delridge',
                'Magnolia / Queen Anne': 'Magnolia / Queen Anne' # Garder tel quel
            }
            df['Neighborhood'] = df['Neighborhood'].replace(replace_map)
            
        return df
    except Exception as e:
        print(f"Erreur chargement données EDA: {e}")
        return pd.DataFrame()


# --- CALLBACKS EDA INTERACTIF ---
@app.callback(
    [Output("analysis-scatter-graph", "figure"),
     Output("analysis-boxplot-graph", "figure"),
     Output("analysis-stats-table", "children")],
    [Input("scatter-x", "value"), Input("scatter-y", "value"),
     Input("boxplot-x", "value"), Input("boxplot-y", "value")]
)
def update_eda_graphs(sx, sy, bx, by):
    df = load_full_data()
    if df.empty: return go.Figure(), go.Figure(), "Erreur chargement"

    # 1. Scatter Plot
    fig_scatter = px.scatter(
        df, x=sx, y=sy, 
        color="BuildingType" if "BuildingType" in df.columns else None,
        hover_data=['Neighborhood'] if "Neighborhood" in df.columns else None,
        title=f"Relation : {sx} vs {sy}",
        template="plotly_dark",
        opacity=0.7
    )
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # 2. Boxplot
    fig_box = px.box(
        df, x=bx, y=by, 
        color=bx,
        title=f"Distribution : {by} par {bx}",
        template="plotly_dark"
    )
    fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)

    # 3. Stats Table
    # 3. Stats Table
    # Eviter les colonnes dupliquées si sy == by
    cols_to_desc = list(set([sy, by]))
    stats = df[cols_to_desc].describe().reset_index()
    stats_table = dash_table.DataTable(
        data=stats.to_dict('records'),
        columns=[{"name": i, "id": i} for i in stats.columns],
        style_header={'backgroundColor': '#1e293b', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'backgroundColor': 'rgba(255,255,255,0.05)', 'color': '#e2e8f0', 'textAlign': 'left'},
    )

    return fig_scatter, fig_box, stats_table

# --- PAGE 0.2: MODÉLISATION ---
@functools.lru_cache(maxsize=1)
def load_model_metrics():
    try:
        csv_path = os.path.join(RESULTS_DIR, 'metrics_comparison.json')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Erreur chargement metrics: {e}")
    return None

def _layout_modeling_impl(lang):
    t = TRANSLATIONS[lang]
    
    # Chargement des résultats (Cached)
    models_data = []
    best_model = {}
    
    try:
        df_res = load_model_metrics()
        if df_res is not None and not df_res.empty:
            # Trier par R2 décroissant
            df_res = df_res.sort_values(by="R² Test", ascending=False)
            models_data = df_res.to_dict('records')
            if models_data:
                best_model = models_data[0]
        else:
            # Fallback
            models_data = [
                {"Modèle": "M1 - GB (Optimisé)", "R² Test": 0.98, "RMSE Original": 52.09, "MAPE": 0.09},
                {"Modèle": "M2 - RF (Optimisé)", "R² Test": 0.99, "RMSE Original": 47.45, "MAPE": 0.07}
            ]
            best_model = models_data[1]
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")

    # --- 1. BANDEAU MEILLEUR MODÈLE ---
    best_model_banner = dbc.Alert([
        html.H4(f"{t['mod_best_title']} {best_model.get('Modèle', 'N/A')}", className="alert-heading"),
        html.P(f"{t['mod_best_perf']} {best_model.get('R² Test', 0):.4f} {t['mod_best_mape']} {float(best_model.get('MAPE', 0))*100:.1f}%. {t['mod_best_desc']}"),
        html.Hr(),
        html.P(t['mod_best_usage'], className="mb-0 small")
    ], color="success", className="mb-5 glass-card", style={"borderLeft": "5px solid #00fa9a"})

    # --- 2. TABLEAU COMPARATIF EXHAUSTIF ---
    table_header = html.Thead(html.Tr([
        html.Th(t['mod_table_model']), 
        html.Th(t['mod_table_r2']), 
        html.Th(t['mod_table_rmse']), 
        html.Th(t['mod_table_mape']),
        html.Th(t['mod_table_cv'])
    ]))
    
    table_rows = []
    for m in models_data:
        # Style conditionnel pour le meilleur modèle
        is_best = m == best_model
        style = {"fontWeight": "bold", "color": "#00fa9a", "backgroundColor": "rgba(0,250,154,0.1)"} if is_best else {}
        
        table_rows.append(html.Tr([
            html.Td(m.get("Modèle")),
            html.Td(f"{m.get('R² Test', 0):.4f}"),
            html.Td(f"{m.get('RMSE Original', 0):.2f}"),
            html.Td(f"{m.get('MAPE', 0):.4f}"),
            html.Td(f"{m.get('CV R² Mean', 0):.4f}")
        ], style=style))

    full_table = dbc.Table([table_header, html.Tbody(table_rows)], bordered=True, color="dark", hover=True, responsive=True, className="mb-4")

    # --- 3. RADAR CHART COMPARATIF (M1 vs M2) ---
    # On prend le meilleur de M1 et le meilleur de M2 pour le radar
    # Si non trouvé explicitement par nom, on prend les deux premiers du classement
    m1_best = next((m for m in models_data if "M1" in m.get("Modèle", "")), None)
    m2_best = next((m for m in models_data if "M2" in m.get("Modèle", "")), None)
    
    if not m1_best and len(models_data) > 1: m1_best = models_data[1]
    if not m2_best and len(models_data) > 0: m2_best = models_data[0]
    
    fig_radar = go.Figure()
    
    if m1_best and m2_best:
        categories = [t['mod_radar_precision'], t['mod_radar_stability'], t['mod_radar_generalization']]
        
        def safe_get(d, k): return float(d.get(k, 0))
        
        # 1. Trace M2 (Meilleur modèle -> Grand triangle -> En arrière plan)
        fig_radar.add_trace(go.Scatterpolar(
            r=[safe_get(m2_best, 'R² Test'), 1 - safe_get(m2_best, 'MAPE'), safe_get(m2_best, 'CV R² Mean')],
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 250, 154, 0.2)',
            line=dict(color='#00fa9a'),
            name=m2_best['Modèle']
        ))
        
        # 2. Trace M1 (Modèle de base -> Petit triangle -> Au premier plan)
        fig_radar.add_trace(go.Scatterpolar(
            r=[safe_get(m1_best, 'R² Test'), 1 - safe_get(m1_best, 'MAPE'), safe_get(m1_best, 'CV R² Mean')],
            theta=categories,
            fill='toself',
            fillcolor='rgba(255, 99, 132, 0.2)',
            line=dict(color='rgba(255, 99, 132, 1)'),
            name=m1_best['Modèle']
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.5, 1], gridcolor="#444"),  # Zoom pour voir les écarts
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(text=t['mod_radar_chart_title'], x=0.5)
        )
    
    
    # --- SECTIONS DU RAPPORT ---
    sections = [
        {
            "id": "model-comparison",
            "title": t['mod_compare_title'],
            "desc": t['mod_compare_desc'],
            "content": html.Div([
                full_table,
                html.P(t['mod_table_note'], className="text-muted small")
            ]),
            "images": [
                {"src": "/assets/model_figures/final_comparison_optimized.png", "title": t['mod_img_comparison'], "width": 12},
                {"src": "/assets/model_figures/comparison_rigorous.png", "title": "Comparaison Rigoureuse M1 vs M2" if lang=='FR' else "Rigorous M1 vs M2 Comparison", "width": 12}
            ]
        },
        {
            "id": "radar-comparison",
            "title": t['mod_radar_title'],
            "desc": t['mod_radar_desc'],
            "content": dcc.Graph(figure=fig_radar)
        },
        {
            "id": "feat-importance",
            "title": t['mod_feat_title'],
            "desc": t['mod_feat_desc'],
            "images": [
                {"src": "/assets/model_figures/feature_importance_m1.png", "title": t['mod_img_feat_m1'], "width": 12},
                {"src": "/assets/model_figures/feature_importance_m2.png", "title": t['mod_img_feat_m2'], "width": 12}
            ]
        },
        {
            "id": "pred-analysis",
            "title": t['mod_resid_title'],
            "desc": t['mod_resid_desc'],
            "images": [
                {"src": "/assets/model_figures/predictions_m1.png", "title": "Prédictions M1" if lang=='FR' else "M1 Predictions", "width": 12},
                {"src": "/assets/model_figures/predictions_m2.png", "title": "Prédictions M2" if lang=='FR' else "M2 Predictions", "width": 12},
                {"src": "/assets/model_figures/residuals_m1.png", "title": "Résidus M1" if lang=='FR' else "M1 Residuals", "width": 12},
                {"src": "/assets/model_figures/residuals_m2.png", "title": "Résidus M2" if lang=='FR' else "M2 Residuals", "width": 12}
            ]
        }
    ]

    return html.Div([
        html.H1([html.I(className="fas fa-microchip me-3"), t['nav_modeling']], className="mb-4 text-center gradient-text"),
        best_model_banner,
        build_gallery_ui(None, sections, lang)
    ], style={"paddingBottom": "100px"})


def layout_modeling(lang):
    try:
        return _layout_modeling_impl(lang)
    except Exception as e:
        import traceback
        return html.Div([
            html.H3("⚠️ Erreur de Chargement (Modélisation)"),
            html.Hr(),
            dbc.Alert(str(e), color="danger"),
            html.Pre(traceback.format_exc(), style={"backgroundColor": "#333", "color": "#f88", "padding": "20px", "borderRadius": "5px"})
        ], className="container mt-5")


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
    
    sim_options = [
        {'label': t['sim_opt_led'], 'value': 'led', 'disabled': False},
        {'label': t['sim_opt_hvac'], 'value': 'hvac', 'disabled': False},
        {'label': t['sim_opt_windows'], 'value': 'windows', 'disabled': False},
        {'label': t['sim_opt_solar'], 'value': 'solar', 'disabled': False},
        {'label': t['sim_opt_gtb'], 'value': 'gtb', 'disabled': False}
    ]

    return html.Div([
        html.H1([html.I(className="fas fa-tools me-3"), t['nav_sim']], className="mb-4 gradient-text"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H4(t['sim_renov_menu']),
                    html.P(t['sim_select_upgrades'], className="text-muted mb-2"),
                    dbc.Alert([
                        html.H6(t['sim_es_explain_title'], className="alert-heading small fw-bold"),
                        html.P(t['sim_es_explain_1'], className="small mb-1"),
                        html.P(t['sim_es_explain_2'], className="small mb-1"),
                        html.Small(t['sim_es_source'], className="d-block mt-1 fst-italic", style={"color": "white"})
                    ], color="info", className="p-2 mb-3", style={"fontSize": "0.85rem", "backgroundColor": "rgba(13, 202, 240, 0.1)", "borderColor": "rgba(13, 202, 240, 0.2)"}),
                    html.Hr(),
                    dbc.Checklist(
                        options=sim_options,
                        value=[],
                        id="sim-renov-check",
                        switch=True,
                        className="mb-3",
                        style={"fontSize": "1.1rem"}
                    ),
                    html.Hr(),
                    html.Label(t['sim_transition'], className="fw-bold mb-2"),
                    dbc.Switch(
                        id="sim-fuel-switch",
                        label=t['sim_elec_switch'],
                        value=False,
                        className="mb-3 text-warning"
                    ),
                    html.Div(id="sim-score-boost-badge", className="text-center mt-3")
                ], className="glass-card p-3")
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.H4(t['sim_savings'], className="text-center"),
                    html.Div(id="sim-savings-summary", className="text-center mb-3", style={"fontSize": "1.3rem"}),
                    dcc.Loading(dcc.Graph(id="sim-comparison-graph", style={"height": "300px"}), type="circle", color="#00fa9a"),
                    html.Hr(),
                    html.H5(t['sim_sensibility'], className="text-center mt-2"),
                    dcc.Loading(dcc.Graph(id="sim-sensitivity-graph", style={"height": "250px"}), type="circle", color="#3b82f6")
                ], className="glass-card p-3")
            ], width=8),
        ])
    ])

@app.callback(
    [Output("sim-comparison-graph", "figure"), Output("sim-sensitivity-graph", "figure"), Output("sim-savings-summary", "children")],
    [Input("sim-renov-check", "value"), Input("sim-fuel-switch", "value"), Input("lang-switch", "value")],
    [State("stored-building-features", "data")]
)
def update_sim_graph(selected_actions, fuel_switch, lang, stored_data):
    print(f"DEBUG: Sim update. Actions={selected_actions}, Switch={fuel_switch}")
    lang = lang or 'FR'
    t = TRANSLATIONS[lang]
    selected_actions = selected_actions or []
    
    # 1. Calcul du Bonus Score
    scores_map = {'led': 8, 'hvac': 15, 'windows': 10, 'solar': 12, 'gtb': 5}
    boost = sum(scores_map.get(a, 0) for a in selected_actions)

    base_features = stored_data or {'PropertyGFATotal': 50000, 'YearBuilt': 1980, 'PrimaryPropertyType': 'Office', 'ENERGYSTARScore': 50}
    
    base_es = float(base_features.get('ENERGYSTARScore', 0))
    if base_es <= 0: base_es = 50 
    
    final_es = min(base_es + boost, 100)
    is_capped = (base_es + boost) > 100
    
    # 2. Prédiction de Base
    val_base, _ = predict_co2(base_features)
    
    # 3. Prédiction Améliorée (Rénovation + Fuel Switch)
    improved_features = base_features.copy()
    improved_features['ENERGYSTARScore'] = final_es
    
    if fuel_switch:
        improved_features['Has_Gas'] = False
        improved_features['Has_Steam'] = False
        # Le Fuel Switch peut aussi avoir un impact technologique bonus
        
    val_improved, _ = predict_co2(improved_features)
    
    # Bonus technologique supplémentaire (hors score ES)
    tech_bonus = 0.95 if 'hvac' in selected_actions else 1.0
    val_improved *= tech_bonus
    
    # Si tout électrique, on réduit drastiquement les émissions (hypothèse locale: élec décarbonée)
    if fuel_switch:
        val_improved *= 0.6  # Hypothèse Seattle Hydro

    savings = val_base - val_improved
    percent = (savings / val_base) * 100 if val_base > 0 else 0
    
    # 4. Graphique Comparaison (Bar Chart)
    labels = [t['sim_before'], t['sim_after']]
    fig_comp = go.Figure(data=[
        go.Bar(
            x=labels, 
            y=[val_base, val_improved],
            marker_color=['#64748b', '#00fa9a'],
            text=[f"{val_base:.1f} T", f"{val_improved:.1f} T"],
            textposition='auto',
        )
    ])
    fig_comp.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        yaxis_title=t['tonnes_co2_year'],
        title=f"{t['sim_impact_title']} ({t['sim_score']} {final_es:.0f})",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # 5. Graphique Sensibilité (Curve)
    # Calculer CO2 pour ES allant de 10 à 100
    curve_x = list(range(10, 101, 10))
    curve_y = []
    
    temp_feats = improved_features.copy() # On garde le fuel switch actif pour la courbe
    for s in curve_x:
        temp_feats['ENERGYSTARScore'] = s
        p, _ = predict_co2(temp_feats)
        if fuel_switch: p *= 0.6 # Appliquer le facteur Fuel Switch
        curve_y.append(p)
        
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=curve_x, y=curve_y, mode='lines+markers', line=dict(color='#3b82f6', width=3), name=t['sim_potential']))
    
    # Marquer le point actuel et amélioré
    fig_sens.add_trace(go.Scatter(x=[base_es], y=[val_base], mode='markers', marker=dict(color='#ef4444', size=12), name=t['sim_current']))
    fig_sens.add_trace(go.Scatter(x=[final_es], y=[val_improved], mode='markers', marker=dict(color='#00fa9a', size=12), name=t['sim_projected']))
    
    fig_sens.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Score Energy Star",
        yaxis_title=t['2050_emissions'],
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True
    )

    # Résumé Texte
    summary_text = [
        html.Span(f"- {savings:.1f} T ", style={"color": "#00fa9a", "fontWeight": "bold", "fontSize": "1.5rem"}),
        html.Span(f"({percent:.1f}%)", style={"color": "#00fa9a", "fontSize": "1rem"}),
        html.Div(f"{t['sim_score']}: +{boost} pts | {t['sim_all_electric'] if fuel_switch else t['sim_standard_mix']}", className="text-muted small mt-1")
    ]
    if is_capped:
        summary_text.append(dbc.Alert(t['sim_score_cap'], color="warning", className="mt-2 py-1 small"))
    
    return fig_comp, fig_sens, summary_text
       
def layout_star(lang):
    return html.Div(id="dynamic-star-container")

@app.callback(
    Output("dynamic-star-container", "children"),
    [Input("lang-switch", "value"), Input("url", "pathname")],
    [State("stored-building-features", "data"), State("current-prediction", "data")]
)
def update_star_page(lang, pathname, features, prediction):
    if pathname != "/star": return dash.no_update
    lang = lang or 'FR'
    t = TRANSLATIONS[lang]
    
    # 1. State Check
    if not prediction or not features:
        return html.Div([
            html.Div([
                html.I(className="fas fa-star fa-4x mb-4", style={"color": "#00fa9a"}),
                html.H2(t['nav_star']),
                html.P(t['please_predict_first'], className="text-muted"),
                dbc.Button(t['nav_predict'], href="/predict", color="success", className="mt-3")
            ], className="text-center py-5 glass-card")
        ], style={"maxWidth": "800px", "margin": "0 auto"})

    # 2. Metrics & Comparison
    current_val = float(prediction)
    gfa = features.get('PropertyGFATotal', 50000)
    b_type = features.get('PrimaryPropertyType', 'Office')
    
    # Median Reference (approximate based on type)
    base_eui = BUILDING_TYPE_BENCHMARKS.get(b_type, 100)
    median_emissions = (gfa * base_eui) / 1000 * 0.05
    if median_emissions < 10: median_emissions = current_val * 1.1

    ratio = current_val / median_emissions
    
    # Stars Logic (Inverse: Lower ratio is better)
    stars = 5 if ratio < 0.6 else 4 if ratio < 0.8 else 3 if ratio < 1.0 else 2 if ratio < 1.3 else 1

    # Advice Logic
    next_level_target = 0
    if stars < 5:
        target_ratios = {1: 1.3, 2: 1.0, 3: 0.8, 4: 0.6}
        next_r = target_ratios.get(stars, 0.6)
        next_level_target = median_emissions * next_r
        reduction_needed = current_val - next_level_target
        advice_text = f"🎯 {t['star_advice_next']} : {t['star_reduce_by']} {reduction_needed:.1f} T {t['star_to_gain_star']}"
    else:
        advice_text = t['star_leader_msg']
    
    # Leaderboard Chart (Podium)
    top_perf = median_emissions * 0.6
    
    fig_leaderboard = go.Figure()
    
    # Barre Moyenne (Référence)
    fig_leaderboard.add_trace(go.Bar(
        y=[t['star_label_avg']], x=[median_emissions], orientation='h', 
        marker_color='#94a3b8', name='Moyenne Seattle', text=f"{median_emissions:.1f} T", textposition='auto'
    ))
    
    # Barre Vous (Couleur selon performance)
    my_color = '#00fa9a' if current_val <= top_perf else '#ffd700' if current_val <= median_emissions else '#ef4444'
    fig_leaderboard.add_trace(go.Bar(
        y=[t['star_label_us']], x=[current_val], orientation='h', 
        marker_color=my_color, name='Votre Bâtiment', text=f"{current_val:.1f} T", textposition='auto'
    ))
    
    # Barre Top 10% (Objectif)
    fig_leaderboard.add_trace(go.Bar(
        y=[t['star_label_target']], x=[top_perf], orientation='h', 
        marker_color='#00fa9a', name='Objectif', text=f"{top_perf:.1f} T", textposition='auto'
    ))
    
    fig_leaderboard.update_layout(
        title=t['star_chart_pos_title'],
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='group',
        xaxis_title="CO2 (Tonnes/an)",
        margin=dict(l=80, r=20, t=40, b=40),
        height=300,
        showlegend=False
    )

    # Gauge Chart (Compteur Néon Modernisé)
    bar_color = "#00fa9a" if current_val <= median_emissions * 1.1 else "#ef4444"
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': t['star_gauge_title_full'], 'font': {'size': 16, 'color': '#cbd5e1'}},
        
        # Le Delta montre l'écart relatif en %
        delta = {
            'reference': median_emissions, 
            'relative': True, # Afficher en pourcentage 
            'position': "bottom",
            'valueformat': ".1%", 
            'increasing': {'color': "#ef4444", 'symbol': '▲ '}, # Plus d'émissions = Rouge (Mauvais)
            'decreasing': {'color': "#00fa9a", 'symbol': '▼ '}  # Moins d'émissions = Vert (Bon)
        },
        
        number = {'suffix': " T", 'font': {'size': 36, 'color': 'white', 'family': 'Arial Black'}},
        
        gauge = {
            'axis': {'range': [0, max(median_emissions * 1.8, current_val * 1.2)], 'tickwidth': 0, 'tickfont': {'color': '#64748b'}},
            'bar': {'color': bar_color, 'thickness': 0.8}, # Barre principale
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, median_emissions], 'color': "rgba(30, 41, 59, 0.5)"}, # Fond sombre
                {'range': [median_emissions, max(median_emissions * 1.8, current_val * 1.2)], 'color': "rgba(30, 41, 59, 0.5)"} 
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.9,
                'value': median_emissions
            }
        }
    ))
    fig_gauge.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        font={'color': "white"},
        margin=dict(l=40, r=40, t=20, b=50),
        # Légende explicative (Annotations)
        annotations=[
            dict(
                x=0.5, y=-0.15, xref='paper', yref='paper',
                text=t.get('star_gauge_legend_annot', "Vert = Meilleur.<br>Rouge = Moins bon."), # Fallback si clé manque
                showarrow=False,
                font=dict(size=12, color="#94a3b8")
            )
        ]
    )
    return html.Div([
        html.H1([html.I(className="fas fa-star me-3"), t['nav_star']], className="mb-4 text-center gradient-text"),
        
        dbc.Row([
            # 1. Podium (Pleine largeur)
            dbc.Col([
                dbc.Card([
                    html.H5(t['star_podium_title']),
                    dcc.Graph(figure=fig_leaderboard, style={"height": "350px"}), # Hauteur augmentée
                    html.H2(f"{stars}/5 {t['stars']}", className="text-center mb-1", style={"color": "#ffd700"}),
                    html.P([
                        t['star_building_emits'], " ", 
                        html.Strong(f"{current_val:.1f} T"), 
                        " ", t['star_per_year']
                    ], className="text-center"),
                    dbc.Alert(advice_text, color="info" if stars < 5 else "success", className="mt-3"),
                    
                    html.Div([
                        dbc.Accordion([
                            dbc.AccordionItem([
                                html.Ul([
                                    html.Li("⭐⭐⭐⭐⭐ : Émissions < 60% de la moyenne (Excellent)"),
                                    html.Li("⭐⭐⭐⭐ : Émissions < 80% de la moyenne (Très bon)"),
                                    html.Li("⭐⭐⭐ : Dans la moyenne (Standard)"),
                                    html.Li("⭐⭐ : Émissions < 130% de la moyenne (Peut mieux faire)"),
                                    html.Li("⭐ : Émissions > 130% de la moyenne"),
                                ], className="small text-muted mb-0")
                            ], title=t['star_accordion_title'], item_id="info-stars")
                        ], start_collapsed=True, flush=True)
                    ], className="mt-2")
                ], className="glass-card p-4 mb-4")
            ], width=12),
            
            # 2. Jauge (Pleine largeur)
            dbc.Col([
                dbc.Card([
                    html.H5(t['star_gauge_title']),
                    dcc.Graph(figure=fig_gauge, style={"height": "400px"}),
                    
                    # Légende détaillée visuelle
                    html.Div([
                        html.Div([html.I(className="fas fa-square me-2", style={"color": bar_color}), t['star_label_us']], className="d-flex align-items-center me-4"),
                        html.Div([html.I(className="fas fa-minus me-2 text-white", style={"fontWeight": "bold", "fontSize": "1.2em"}), t['star_gauge_avg_sector']], className="d-flex align-items-center me-4"),
                        html.Div([html.I(className="fas fa-percentage me-2 text-info"), t['star_gauge_gap']], className="d-flex align-items-center"),
                    ], className="d-flex justify-content-center mb-3 small"),
                    
                    html.P(t['star_gauge_legend_desc'].format(b_type, median_emissions), className="text-center text-muted small fst-italic")
                ], className="glass-card p-4")
            ], width=12)
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
    if pathname != "/2050": return dash.no_update
    lang = lang or 'FR'
    t = TRANSLATIONS[lang]
    
    if not last_pred or not stored_data:
        return html.Div([
            html.Div([
                html.I(className="fas fa-chart-line fa-4x mb-4", style={"color": "#00fa9a"}),
                html.H2(t['nav_2050']),
                html.P(t['please_predict_first'], className="text-muted"),
                dbc.Button(t['nav_predict'], href="/predict", color="success", className="mt-3")
            ], className="text-center py-5 glass-card")
        ], style={"maxWidth": "800px", "margin": "0 auto"})

    # --- CALCULS STRATEGIQUES ---
    curr_val = float(last_pred)
    gfa = stored_data.get('PropertyGFATotal', 50000)
    b_type = stored_data.get('PrimaryPropertyType', 'Office')
    
    # Cibles (Hypothèse Seattle Climate Commitment)
    base_target = BUILDING_TYPE_BENCHMARKS.get(b_type, 100) * (gfa / 1000 * 0.05)
    
    targets = {
        2024: curr_val,   # Départ
        2030: base_target * 0.7, # -30% vs standard
        2040: base_target * 0.4, # -60% vs standard
        2050: 0             # Net Zero
    }
    
    years = [2024, 2030, 2040, 2050]
    reg_path = [targets[y] for y in years]
    bau_path = [curr_val for _ in years] # Business As Usual (si on ne fait rien)
    
    # Calcul Excess (optionnel, pour info interne ou futur usage)
    excess_total = 0
    for i in range(len(years)):
        gap = max(0, bau_path[i] - reg_path[i])
        excess_total += gap
    
    status_icon = "fa-check-circle" if curr_val <= targets[2030] else "fa-exclamation-triangle"
    status_color = "#00fa9a" if curr_val <= targets[2030] else "#ef4444"
    status_text = t['2050_compliant'] if curr_val <= targets[2030] else t['2050_non_compliant']

    # --- GRAPHIQUE TRAJECTOIRE ---
    fig = go.Figure()
    
    # Zone de Risque (Entre BAU et Target)
    # Zone de Risque
    fig.add_trace(go.Scatter(
        x=years, y=bau_path, mode='lines', 
        name=t['2050_scen_current'], 
        line=dict(color='#ef4444', width=2, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=years, y=reg_path, mode='lines+markers', 
        name=t['2050_scen_target'], 
        line=dict(color='#00fa9a', width=3),
        fill='tonexty', fillcolor='rgba(239, 68, 68, 0.2)' 
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=t['2050_chart_title'], font=dict(size=18, color="white")),
        xaxis_title=t['2050_year'],
        yaxis_title=t['2050_emissions'],
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=20, r=20, t=50, b=20),
        # Annotations explicatives
        annotations=[
            dict(
                x=2040, y=(targets[2040] + bau_path[2])/2,
                text=t['2050_annot_effort'],
                showarrow=False,
                font=dict(color="#f87171", size=11)
            ),
            dict(
                x=2050, y=0,
                xref="x", yref="y",
                text=t['2050_annot_zero'],
                showarrow=True,
                arrowhead=1,
                ax=0, ay=-30,
                font=dict(color="#00fa9a", weight="bold")
            )
        ]
    )

    return html.Div([
        html.H1([html.I(className="fas fa-chart-line me-3"), t['nav_2050']], className="mb-4 text-center"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H5(t['2050_statut']),
                    html.Div([
                        html.I(className=f"fas {status_icon} fa-4x mb-3", style={"color": status_color}),
                        html.H2(status_text, style={"color": status_color}),
                        html.P(f"{t['2050_current_vs_target']} ({curr_val:.1f} T) {t['2050_vs']} ({targets[2030]:.1f} T).", className="text-muted")
                    ], className="text-center my-4")
                ], className="glass-card p-4 h-100")
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    html.H5(t['2050_traj_title']),
                    html.P(t['2050_red_zone_desc'], className="text-muted small"),
                    dcc.Graph(figure=fig, style={"height": "350px"}),
                    html.Small(t['2050_source'], className="text-muted d-block text-end mt-2 fst-italic")
                ], className="glass-card p-3 h-100")
            ], width=8),
        ])
    ])

# --- GLOBAL CALLBACKS ---

@app.callback(
    [Output("sidebar-subtitle", "children"),
     Output("nav-home", "children"), Output("nav-analysis", "children"),
     Output("nav-modeling", "children"), Output("nav-predict", "children"),
     Output("nav-sim", "children"), Output("nav-star", "children"),
     Output("nav-2050", "children")],
    [Input("lang-switch", "value")]
)
def update_sidebar_labels(lang):
    t = TRANSLATIONS[lang or 'FR']
    return (
        t['subtitle'],
        [html.I(className="fas fa-chart-pie me-2"), t['nav_insights']],
        [html.I(className="fas fa-search-plus me-2"), t['nav_analysis']],
        [html.I(className="fas fa-brain me-2"), t['nav_modeling']],
        [html.I(className="fas fa-magic me-2"), t['nav_predict']],
        [html.I(className="fas fa-tools me-2"), t['nav_sim']],
        [html.I(className="fas fa-star me-2"), t['nav_star']],
        [html.I(className="fas fa-check-circle me-2"), t['nav_2050']]
    )

@app.callback(
    [Output("page-content", "children"), Output("last-nav-timestamp", "data")],
    [Input("url", "pathname"), Input("lang-switch", "value"), Input("theme-store", "data")],
    [State("last-nav-timestamp", "data")]
)
def render_page_content(pathname, lang, theme, last_timestamp):
    import time
    current_time = time.time() * 1000  # Convert to milliseconds
    

    
    print(f"✓ ROUTING to {pathname} (allowed)")
    
    lang = lang or 'FR'
    theme = theme or 'dark'
    
    page_content = None
    if pathname == "/": page_content = layout_insights(lang, theme)
    elif pathname == "/analysis": page_content = layout_analysis(lang)
    elif pathname == "/modeling": page_content = layout_modeling(lang)
    elif pathname == "/predict": page_content = layout_predict(lang)
    elif pathname == "/sim": page_content = layout_sim(lang)
    elif pathname == "/star": page_content = layout_star(lang)
    elif pathname == "/2050": page_content = layout_benchmark(lang)
    else: page_content = html.Div([html.H1("404", className="text-danger"), html.P(f"Chemin {pathname} inconnu.")], className="p-5 text-center")
    
    return page_content, current_time

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
    try:
        if not n_clicks: 
            return "", "", "", go.Figure().update_layout(template="plotly_dark"), dash.no_update, dash.no_update, dash.no_update
        
        lang = lang or 'FR'
        
        # === VALIDATION DES INPUTS ===
        errors = []
        
        # Validation Surface
        if surface is None or surface <= 0:
            errors.append("La surface doit être supérieure à 0" if lang == 'FR' else "Surface must be greater than 0")
        elif surface > 10_000_000:  # 10M sqft = limite réaliste
            errors.append("Surface trop grande (max 10M sqft)" if lang == 'FR' else "Surface too large (max 10M sqft)")
        
        # Validation Étages
        if floors is None or floors < 1:
            errors.append("Le nombre d'étages doit être au moins 1" if lang == 'FR' else "Number of floors must be at least 1")
        elif floors > 200:  # Limite réaliste
            errors.append("Nombre d'étages trop élevé (max 200)" if lang == 'FR' else "Too many floors (max 200)")
        
        # Validation Année
        if year is None or year < 1800:
            errors.append("Année de construction invalide (min 1800)" if lang == 'FR' else "Invalid year (min 1800)")
        elif year > 2026:
            errors.append("Année de construction ne peut pas être dans le futur" if lang == 'FR' else "Year cannot be in the future")
        
        # Validation Energy Star
        if es is not None and (es < 0 or es > 100):
            errors.append("Score Energy Star doit être entre 0 et 100" if lang == 'FR' else "Energy Star score must be between 0 and 100")
        
        # Si erreurs, retourner message d'erreur
        if errors:
            error_msg = html.Div([
                html.H4("⚠️ Erreurs de Validation" if lang == 'FR' else "⚠️ Validation Errors", className="text-danger"),
                html.Ul([html.Li(e, className="text-warning") for e in errors])
            ])
            return error_msg, "", "", go.Figure().update_layout(template="plotly_dark"), dash.no_update, dash.no_update, dash.no_update
        
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
        fig_xai.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=0, r=20, t=30, b=40), 
            yaxis=dict(autorange="reversed"),
            # Ajout Légende via Annotations
            annotations=[
                dict(
                    x=0, y=1.1, xref='paper', yref='paper',
                    text="<span style='color:#ff4d4d'>■ Augmente</span> / <span style='color:#00fa9a'>■ Réduit</span> les émissions",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
        
        # Badge de fiabilité dynamique
        rel = get_reliability_info(val, features) 
        badge = dbc.Badge(rel, color="success" if rel=="Élevé" else "warning", className="ms-2")
        
        # Recommandations
        recos = get_decarbonization_recommendations(features)
        decarbon_ui = html.Div([dbc.Card(html.Div(r, className="p-2"), className="mb-2 glass-card") for r in recos])

        new_baseline = val if baseline is None else baseline
        
        return f"{val:.2f}", badge, decarbon_ui, fig_xai, features, val, new_baseline

    except Exception as e:
        print(f"CRITICAL PREDICTION ERROR: {e}")
        traceback.print_exc()
        err_div = html.Div([html.H4("⚠️ Erreur Interne", className="text-danger"), html.P(str(e))])
        return err_div, "", "", go.Figure(), dash.no_update, dash.no_update, dash.no_update

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
        
        # Graphs
        # Graphs
        # 1. Distribution des Émissions (Histogramme amélioré avec marginal)
        fig_hist = px.histogram(
            df, x='CO2_Emissions_Predicted', nbins=30, 
            title="Distribution des Émissions", 
            template="plotly_dark", 
            color_discrete_sequence=['#00fa9a'],
            marginal="box", # Ajoute boîte à moustaches au dessus
            opacity=0.8
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Émissions CO2 (Tonnes)", yaxis_title="Nombre de Bâtiments")

        # 2. Émissions par Type (Histogramme/Barres)
        fig_box = px.histogram(
            df, x='PrimaryPropertyType', y='CO2_Emissions_Predicted', histfunc='sum',
            title="Total Émissions par Type d'Usage", 
            template="plotly_dark", 
            color='PrimaryPropertyType' # Couleurs variées par type
        )
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Type d'Usage", yaxis_title="Total CO2 (Tonnes)")
        
        fig_scatter = px.scatter(df, x='PropertyGFATotal', y='CO2_Emissions_Predicted', color='PrimaryPropertyType', title="Corrélation Surface vs CO2", template="plotly_dark")
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
            
            html.H5("Aperçu des Données", className="mt-4"),
            dash_table.DataTable(
                id='batch-prediction-table',
                data=df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': '#1e293b', 
                    'color': '#ff9800',  # Orange vif
                    'fontWeight': 'bold',
                    'border': '1px solid #475569',
                    'fontSize': '14px'
                },
                style_cell={
                    'backgroundColor': '#0f172a', 
                    'color': '#ffd700',  # Jaune vif
                    'textAlign': 'left',
                    'border': '1px solid #334155',
                    'padding': '10px',
                    'minWidth': '100px',
                    'fontSize': '13px'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'CO2_Emissions_Predicted'},
                        'backgroundColor': '#1a3a2a',
                        'color': '#00ff00',  # Vert vif
                        'fontWeight': 'bold'
                    }
                ],
                page_size=5
            ),
            
            html.H5("Analyse Visuelle du Portefeuille", className="mt-5 mb-4"),
            
            # Graph 1 : Distribution (Pleine largeur)
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_hist, style={"height": "450px"}), width=12)
            ], className="mb-5"),
            
            # Graph 2 : Par Type (Pleine largeur)
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_box, style={"height": "450px"}), width=12)
            ], className="mb-5"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_scatter, style={"height": "400px"}), width=12),
            ], className="mt-4"),

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
    try:
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
    except Exception as e:
        print(f"Error in update_kpis: {e}")
        return "N/A", "N/A", "N/A"

@app.callback(
    Output("insights-map", "figure"),
    [Input("map-filter-nbh", "value"), Input("lang-switch", "value")]
)
def update_insights_map(selected_nbh, lang):
    try:
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
    except Exception as e:
        print(f"Error in update_insights_map: {e}")
        return go.Figure()

@app.callback(
    [Output("insights-bar-chart", "figure"), Output("insights-pie-chart", "figure")],
    [Input("map-filter-nbh", "value")]
)
def update_insights_charts(selected_nbh):
    try:
        df = load_full_data()
        if df.empty: return go.Figure(), go.Figure()

        # Filtre quartier
        if selected_nbh:
            # Filtrage insensible à la casse
            selected_lower = [s.lower() for s in selected_nbh]
            df = df[df['Neighborhood'].astype(str).str.lower().isin(selected_lower)]

        # 1. Bar Chart: Emissions moyennes par type
        if 'BuildingType' in df.columns and 'TotalGHGEmissions' in df.columns:
            avg_emissions = df.groupby('BuildingType')['TotalGHGEmissions'].mean().reset_index()
            avg_emissions = avg_emissions.sort_values('TotalGHGEmissions', ascending=True) # Top pollueurs en bas

            fig_bar = px.bar(
                avg_emissions, 
                x='TotalGHGEmissions', 
                y='BuildingType',
                orientation='h',
                color='BuildingType',
                template="plotly_dark",
                title=None
            )
            fig_bar.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="CO2 Moyen (T)",
                yaxis_title=None,
                showlegend=False
            )
        else:
            fig_bar = go.Figure()

        # 2. Pie Chart: Répartition
        if 'BuildingType' in df.columns:
            res = df['BuildingType'].value_counts().reset_index()
            res.columns = ['BuildingType', 'Count']
            
            fig_pie = px.pie(
                res, 
                values='Count', 
                names='BuildingType',
                template="plotly_dark",
                hole=0.4
            )
            fig_pie.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
        else:
            fig_pie = go.Figure()

        return fig_bar, fig_pie
    except Exception as e:
        print(f"Error in update_insights_charts: {e}")
        return go.Figure(), go.Figure()


# --- SIDEBAR TOGGLE CALLBACK ---
@app.callback(
    [Output("sidebar-container", "style"), 
     Output("page-content", "style"),
     Output("sidebar-toggle-stored", "data")],
    [Input("sidebar-toggle-btn", "n_clicks")],
    [State("sidebar-toggle-stored", "data")]
)
def toggle_sidebar(n_btn, is_open):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Toggle uniquement avec le bouton
    if triggered_id == "sidebar-toggle-btn":
        is_open = not is_open
    
    # Ensure boolean
    is_open = bool(is_open)
    
    if is_open:
        # VISIBLE
        return {
            "zIndex": 5000, 
            "width": "280px", 
            "left": "0", 
            "display": "block",
            "backgroundColor": "#161b22", 
            "position": "fixed",
            "top": 0,
            "bottom": 0,
            "height": "100vh",
            "overflowY": "auto"
        }, {
            "marginLeft": "280px",
            "transition": "margin-left 0.3s ease"
        }, True
    else:
        # HIDDEN
        return {
            "display": "none"
        }, {
            "marginLeft": "0", 
            "transition": "margin-left 0.3s ease"
        }, False


# --- THEME CALLBACK ---
@app.callback(
    [Output("theme-wrapper-div", "data-theme"), Output("theme-icon", "className"), Output("theme-store", "data")],
    [Input("theme-switch", "n_clicks")],
    [State("theme-store", "data")]
)
def toggle_theme(n_clicks, current_theme):
    if n_clicks is None:
        # Initial load: use stored or default dark
        theme = current_theme or 'dark'
        icon = "fas fa-sun" if theme == 'light' else "fas fa-moon"
        return theme, icon, theme

    new_theme = 'light' if current_theme == 'dark' else 'dark'
    icon = "fas fa-sun" if new_theme == 'light' else "fas fa-moon"
    return new_theme, icon, new_theme

# --- VALIDATION CALLBACK ---
@app.callback(
    [Output("btn-predict", "disabled"), Output("es-error-msg", "children")],
    [Input("in-es", "value")]
)
def validate_energy_star(score):
    if score is None:
        return False, ""
    try:
        val = float(score)
        if val < 0 or val > 100:
            return True, "⚠️ Le score doit être compris entre 0 et 100."
    except:
        return True, "valeur invalide"
    
    return False, ""
    return False, ""

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True, port=port)

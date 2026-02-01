import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io

# Internal imports
from utils.constants import BUILDING_TYPES, NEIGHBORHOODS, CITY_WIDE_STATS, BUILDING_TYPE_BENCHMARKS, NEIGHBORHOOD_STATS
from utils.prediction_logic import (
    predict_co2, get_seattle_metrics, generate_report_pdf, 
    get_feature_importance, get_reliability_info, 
    get_smart_suggestions, get_decarbonization_recommendations
)
from utils.translations import TRANSLATIONS

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)



# --- COMPONENTS ---

# --- COMPONENTS ---

def get_sidebar_content(lang):
    t = TRANSLATIONS[lang]
    return html.Div(id="sidebar-content-toggleable", children=[
        html.Div([
            html.I(className="fas fa-city me-2", style={"fontSize": "1.5rem", "color": "#00fa9a"}),
            html.H2(t['title'], className="d-inline", style={"color": "#00fa9a", "fontSize": "1.5rem"})
        ], className="text-center mb-3 mt-2"),
        html.P(t['subtitle'], className="text-muted small text-center mb-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-chart-pie me-2"), t['insights']], href="/", active="exact"),
                dbc.NavLink([html.I(className="fas fa-search-plus me-2"), t['analysis']], href="/analysis", active="exact"),
                dbc.NavLink([html.I(className="fas fa-brain me-2"), t['modeling']], href="/modeling", active="exact"),
                dbc.NavLink([html.I(className="fas fa-magic me-2"), t['predict']], href="/predict", active="exact"),
                dbc.NavLink([html.I(className="fas fa-tools me-2"), t['sim']], href="/sim", active="exact"),
                dbc.NavLink([html.I(className="fas fa-star me-2"), t['star']], href="/star", active="exact"),
                dbc.NavLink([html.I(className="fas fa-check-circle me-2"), t['2050']], href="/2050", active="exact"),
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

# --- HEADER (Language Switch - STATIC for stability) ---
header_static = html.Div([
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
        # Static Toggle Button
        html.Button([
            html.Span("☰ Menu", id="toggle-icon-text", style={"fontSize": "1.2rem", "fontWeight": "bold", "color": "#00fa9a"})
        ], id="btn-toggle-sidebar", className="sidebar-toggle-btn-inline"),
        
        # Dynamic Sidebar Content
        html.Div(id="sidebar-info-container")
    ], id="sidebar-container", className="sidebar"), 
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
    return html.Div([
        html.H1(t['insights'], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div(t['kpi_buildings'], className="stat-label"),
                html.Div(f"{CITY_WIDE_STATS['total_buildings']}", id="kpi-count", className="stat-value")
            ], className="glass-card"), width=4),
            dbc.Col(dbc.Card([
                html.Div(t['kpi_emissions'], className="stat-label"),
                html.Div(f"{CITY_WIDE_STATS['avg_emissions']} T", id="kpi-mean-co2", className="stat-value")
            ], className="glass-card"), width=4),
            dbc.Col(dbc.Card([
                html.Div(t['kpi_estar'], className="stat-label"),
                html.Div(f"{CITY_WIDE_STATS['avg_energy_star']}", id="kpi-mean-estar", className="stat-value")
            ], className="glass-card"), width=4),
        ]),
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

# --- PAGE 2: CALCULATEUR (With Scientific Insights) ---
def layout_predict(lang):
    t = TRANSLATIONS[lang]
    return html.Div([
        html.H1(t['predict'], className="mb-4 text-center"),
        
        dbc.Row([
            # Sidebar: Inputs
            dbc.Col([
                dbc.Card([
                    html.H4(t['chars_title'], className="mb-4"),
                    
                    html.Label(t['label_usage']),
                    dcc.Dropdown(id="in-type", options=[{'label': b, 'value': b} for b in BUILDING_TYPES], value='Office', className="mb-3", style={'color': '#333'}),

                    html.Label(t['label_nbh']),
                    dcc.Dropdown(id="in-nbh", options=[{'label': n, 'value': n} for n in NEIGHBORHOODS], value='Downtown', className="mb-3", style={'color': '#333'}),
                    
                    html.Label(t['label_surface']),
                    dcc.Input(id="in-surface", type="number", value=50000, className="form-control mb-3"),

                    html.Label(t['label_floors']),
                    dcc.Input(id="in-floors", type="number", value=1, min=1, className="form-control mb-3"),
                    
                    html.Label(t['label_year']),
                    dcc.Slider(id="in-year", min=1900, max=2023, step=1, value=1980, marks={i: str(i) for i in range(1900, 2030, 20)}, className="mb-2"),
                    html.Div(id="year-display", className="text-center mb-3", style={"color": "#00fa9a", "fontWeight": "bold"}),

                    html.Div([
                        html.Label(t['label_es']),
                        dbc.Button("💡 Suggérer" if lang=='FR' else "💡 Suggest", id="btn-smart-es", color="link", size="sm", className="p-0 ms-2", style={"textDecoration": "none", "fontSize": "0.8rem"})
                    ], className="d-flex align-items-center mb-1"),
                    dcc.Input(id="in-es", type="number", value=60, min=0, max=100, className="form-control mb-1"),
                    html.Div(id="smart-es-note", className="small text-muted mb-3"),

                    dbc.Checklist(
                        options=[
                            {"label": t['label_gas'], "value": "gas"},
                            {"label": t['label_steam'], "value": "steam"},
                        ],
                        value=[],
                        id="in-energy-sources",
                        switch=True,
                        className="mb-4"
                    ),

                    dbc.Button(t['predict_btn'], id="btn-predict", color="primary", className="w-100", style={"backgroundColor": "#00fa9a", "borderColor": "#00fa9a", "color": "#000", "fontWeight": "bold"}),
                ], className="glass-card mb-4")
            ], width=5),
            
            # Main Area: Results & Reliability
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.H2(t['out_predicted'], className="text-secondary mb-0"),
                        html.Div(id="reliability-badge-container")
                    ], className="d-flex justify-content-between align-items-center mb-0"),
                    html.H1(id="prediction-output", className="display-1 mb-0", style={"color": "#00fa9a", "fontWeight": "bold", "textShadow": "0 0 20px rgba(0,250,154,0.3)"}),
                    html.P(t['out_unit'], className="text-muted"),
                    
                    html.Hr(),
                    
                    # Decarbonization Recommendations Section
                    html.Div(id="decarbonization-container"),
                    
                    html.Hr(),
                    html.H5(t['out_xai'], className="mb-3 mt-4"),
                    dcc.Graph(id="xai-graph", style={"height": "300px"}),
                    dbc.Button([html.I(className="fas fa-file-pdf me-2"), t['download_pdf']], id="btn-download-pdf", color="success", className="w-100 mt-4"),
                    dcc.Download(id="download-pdf-obj")
                ], className="glass-card mb-4")
            ], width=7),
        ]),
        
        # Section Batch
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H3(t['batch_title'], className="mb-3"),
                    html.P(t['batch_text'], className="text-muted"),
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

# --- PAGE 0: GALERIE EDA (Images Statiques) ---
# --- PAGE 0: GALERIE EDA (Storytelling Layout) ---
# --- PAGE 0.1: ANALYSE EDA ---
def layout_analysis(lang):
    t = TRANSLATIONS[lang]
    sections = [
        {
            "id": "audit",
            "title": "Audit de la Donnée brute" if lang=='FR' else "Raw Data Audit",
            "desc": "Analyse de la complétude et nettoyage des valeurs manquantes.",
            "images": [{"src": "/assets/eda_figures/auto_missing_values.png", "title": "Bilan des Manquants", "width": 12}]
        },
        {
            "id": "target",
            "title": "Analyse de la Cibles" if lang=='FR' else "Target Analysis",
            "desc": "Transformation logarithmique pour stabiliser la variance.",
            "images": [
                {"src": "/assets/eda_figures/01_target_original_distribution.png", "title": "Brute", "width": 12},
                {"src": "/assets/eda_figures/02_target_log_distribution.png", "title": "Log Transform", "width": 12}
            ]
        },
        {
            "id": "factors",
            "title": "Facteurs Drivers" if lang=='FR' else "Impact Drivers",
            "desc": "Répartition par quartiers et types d'usages.",
            "images": [
                {"src": "/assets/eda_figures/auto_Neighborhood_distribution.png", "title": "Quartiers", "width": 12},
                {"src": "/assets/eda_figures/auto_BuildingType_distribution.png", "title": "Usages", "width": 12}
            ]
        },
        {
            "id": "size-energy",
            "title": "Surface & Mix Énergétique" if lang=='FR' else "Size & Energy Mix",
            "desc": "Impact du GFA et des sources d'énergie.",
            "images": [
                {"src": "/assets/eda_figures/auto_PropertyGFATotal_distribution.png", "title": "Surfaces", "width": 12},
                {"src": "/assets/eda_figures/conso_distributions.png", "title": "Mix Énergétique", "width": 12}
            ]
        }
    ]
    return build_gallery_ui(t['analysis'], sections, lang)

# --- PAGE 0.2: MODÉLISATION ---
def layout_modeling(lang):
    t = TRANSLATIONS[lang]
    
    # Tables for model comparisons based on the REGO3 project results
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

    sections = [
        {
            "id": "model-perf",
            "title": "ÉTAPE 5 : Performance du Modèle" if lang=='FR' else "STEP 5: Model Performance",
            "desc": "Analyse de la précision (R²) et de la réduction de l'erreur (MAE) après optimisation.",
            "images": [
                {"src": "/assets/eda_figures/04_model_performance.png", "title": "Prédictions vs Réel (Best Model)", "width": 12},
                {"src": "/assets/eda_figures/distribution_co2.png", "title": "Analyse des Résidus", "width": 12}
            ]
        },
        {
            "id": "model-tuning",
            "title": "Optimisation & Comparaison" if lang=='FR' else "Tuning & Comparison",
            "desc": "Comparaisons des algorithmes testés. L'algorithme XGBoost a été retenu pour sa robustesse face aux relations non-linéaires.",
            "images": [] # Content will be the table
        }
    ]

    ui_elements = [
        html.H1(t['modeling'], className="mb-5 text-center"),
        
        # Step 6: Table (renumbered to Step 5)
        html.Div([
            html.H3(sections[1]['title'], className="text-light"),
            html.P(sections[1]['desc'], className="text-muted mb-4"),
            dbc.Card([
                dbc.Table([table_header, table_body], bordered=True, color='dark', hover=True, responsive=True, striped=True, className="mb-0")
            ], className="glass-card mb-4")
        ]),

        # Step 7: Feature Importance (The link between EDA and Modeling)
        html.Div([
            html.H3("Importance des Variables" if lang=='FR' else "Feature Importance", className="text-light"),
            #html.P("C'est ici que l'analyse exploratoire (EDA) rejoint la modélisation. On identifie les variables qui 'poussent' le plus la prédiction.", className="text-muted mb-4"),
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
    ]
    
    return html.Div(ui_elements, style={"maxWidth": "1000px", "margin": "0 auto"})

def build_gallery_ui(title, sections, lang):
    ui_elements = [html.H1(title, className="mb-5 text-center")]
    for sec in sections:
        ui_elements.append(html.Div([
            html.H3(sec['title'], className="text-light"),
            html.P(sec['desc'], className="text-muted mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    html.H6(img['title'], className="text-center small mb-2"),
                    html.Img(src=img['src'], className="img-fluid rounded zoomable-img", 
                             id={'type': 'eda-image', 'index': img['src'], 'title': img['title']})
                ], className="glass-card p-2"), width=img['width']) for img in sec['images']
            ], className="g-3 mb-5")
        ]))
    return html.Div(ui_elements, style={"maxWidth": "1000px", "margin": "0 auto"})

# --- PAGE 3: WHAT-IF SIMULATOR (DYNAMIC) ---
def layout_sim(lang):
    t = TRANSLATIONS[lang]
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

@app.callback(
    [Output("sim-comparison-graph", "figure"), Output("sim-savings-summary", "children")],
    [Input("sim-es-boost", "value"), Input("lang-switch", "value")],
    [State("stored-building-features", "data")]
)
def update_sim_graph(boost, lang, stored_data):
    lang = lang or 'FR'
    t = TRANSLATIONS[lang]
    
    # Use stored building features
    base_features = stored_data or {'PropertyGFATotal': 50000, 'YearBuilt': 1980, 'PrimaryPropertyType': 'Office', 'ENERGYSTARScore': 50}
    
    # Logic Fix: If base score is 0, we treat it as 50 (average) for comparison 
    # to avoid the penalty of 'entering' the certification system with a low score.
    base_es = float(base_features.get('ENERGYSTARScore', 0))
    if base_es <= 0: base_es = 50 
    
    final_es = min(base_es + boost, 100)
    
    # We compare two 'tracked' scenarios (with_energy_star=True)
    base_for_sim = base_features.copy()
    base_for_sim['ENERGYSTARScore'] = base_es
    
    improved_features = base_features.copy()
    improved_features['ENERGYSTARScore'] = final_es
    
    is_capped = (base_es + boost) > 100
    
    val_base, _ = predict_co2(base_for_sim, with_energy_star=True)
    val_improved, _ = predict_co2(improved_features, with_energy_star=True)
    
    # Additional 'Renovation' logic: switching energy sources or improving insulation
    # In a real scenario, +20 ES points often comes with other fixes.
    # We simulate a small 5% extra efficiency gain for a major boost.
    if boost > 30:
        val_improved *= 0.95 

    savings = val_base - val_improved
    percent = (savings / val_base) * 100 if val_base > 0 else 0
    
    labels = ['Avant / Before', 'Après / After'] if lang == 'EN' else ['Avant', 'Après']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels, 
            y=[val_base, val_improved],
            marker_color=['#888', '#00fa9a'],
            text=[f"{val_base:.1f} T", f"{val_improved:.1f} T"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Tonnes CO2 / an"
    )
    
    f = base_features
    cap_note = " (Capped at 100)" if is_capped else ""
    info_text = f"Surface: {f['PropertyGFATotal']:,} sqft | Usage: {f['PrimaryPropertyType']} | Neighborhood: {f.get('Neighborhood', 'Downtown')} | Year: {f['YearBuilt']} | Floors: {f.get('NumberofFloors', 1)} | New Score: {final_es}{cap_note}"
    
    summary_text = [
        html.Span(f"- {savings:.1f} T ", style={"color": "#00fa9a", "fontWeight": "bold"}),
        html.Span(f"({percent:.1f}%)", style={"color": "#00fa9a", "fontSize": "0.9rem"}),
    ]
    if is_capped:
        summary_text.append(dbc.Alert(
            "⚠️ Le score cumulé dépasse 100. Calcul basé sur le maximum (100)." if lang=='FR' else "⚠️ Combined score exceeds 100. Calculation based on maximum (100).",
            color="warning", className="mt-2 py-1 small"
        ))
    
    return fig, summary_text
       
def layout_star(lang):
    t = TRANSLATIONS[lang]
    metrics = get_seattle_metrics()
    importance = get_feature_importance()
    
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
                html.H5("Performance Prédictive (REGO3)" if lang=='FR' else "Predictive Performance (REGO3)"),
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
    
    # If no prediction yet, show a nice landing placeholder
    if not last_pred:
        return html.Div([
            html.Div([
                html.I(className="fas fa-chart-line fa-4x mb-4", style={"color": "#00fa9a"}),
                html.H2(t['2050']),
                html.P("Veuillez d'abord effectuer une prédiction dans l'onglet 'Calculateur CO2' pour voir votre trajectoire." if lang=='FR' else "Please first make a prediction in the 'CO2 Calculator' tab to see your trajectory.", className="text-muted"),
                dbc.Button(t['predict'], href="/predict", color="success", className="mt-3")
            ], className="text-center py-5 glass-card")
        ], style={"maxWidth": "800px", "margin": "0 auto"})

    try:
        curr_val = float(last_pred)
        base_val = float(baseline_val) if baseline_val else curr_val
        b_type = stored_data.get('PrimaryPropertyType', 'Office')
        median_ref = BUILDING_TYPE_BENCHMARKS.get(b_type, 35.0)
        
        years = [2016, 2030, 2040, 2050]
        # Linear targets: 2016=100%, 2030=60%, 2040=30%, 2050=0%
        targets = [median_ref, median_ref * 0.60, median_ref * 0.30, 0]
        
        fig = go.Figure()
        # 1. Target Path
        fig.add_trace(go.Scatter(
            x=years, y=targets, 
            name="Cible Seattle" if lang=='FR' else "Seattle Target", 
            line=dict(color='#00fa9a', dash='dash', width=3),
            mode='lines+markers'
        ))
        
        # 2. Baseline Status
        fig.add_trace(go.Scatter(
            x=[2016], y=[base_val], 
            name="Point de Départ (2016)" if lang=='FR' else "Baseline (2016)", 
            marker=dict(color='#888', size=12, symbol='circle-open'), 
            mode='markers'
        ))

        # 3. Reference Line for Today
        fig.add_vline(x=2026, line_width=2, line_dash="dot", line_color="rgba(255, 255, 255, 0.4)",
                      annotation_text="Aujourd'hui" if lang=='FR' else "Today", 
                      annotation_position="top left")

        # 4. Current Status
        fig.add_trace(go.Scatter(
            x=[2026], y=[curr_val], 
            name="Bâtiment Actuel" if lang=='FR' else "Current Building", 
            marker=dict(color='#ffd700', size=20, symbol='star'), 
            mode='markers+text',
            text=[f"{curr_val:.1f} T"], 
            textposition="top center"
        ))
        
        # History line
        fig.add_trace(go.Scatter(
            x=[2016, 2026], y=[base_val, curr_val], 
            name="Évolution", 
            line=dict(color='#ffd700', width=2, dash='dot'), 
            mode='lines'
        ))

        fig.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            yaxis_title="Tonnes CO2 / an",
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Compliance logic
        target_2030 = median_ref * 0.60
        is_compliant = curr_val <= target_2030
        
        if curr_val > target_2030:
            deadline = "VIGILANCE" if lang=='FR' else "VIGILANCE"
            status_color = "#ff4d4d"
            status_icon = "fa-exclamation-triangle"
            msg = "Consommation au-dessus des cibles environnementales 2030." if lang=='FR' else "Consumption above 2030 environmental targets."
        elif is_compliant:
            deadline = "OPTIMAL" if lang=='FR' else "OPTIMAL"
            status_color = "#00fa9a"
            status_icon = "fa-check-circle"
            msg = "Performance exemplaire, parfaitement alignée avec la trajectoire bas-carbone." if lang=='FR' else "Exemplary performance, perfectly aligned with low-carbon path."
        else:
            deadline = "STANDBY" if lang=='FR' else "STANDBY"
            status_color = "#ffd700"
            status_icon = "fa-info-circle"
            msg = "Bâtiment proche des objectifs. Des optimisations mineures suffiraient." if lang=='FR' else "Building close to targets. Minor optimizations would suffice."

        return html.Div([
            html.H2(t['2050'], className="mb-4 text-center"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    html.Div([
                        html.H5([html.I(className="fas fa-chart-line me-2"), f"Analyse Comparative : {b_type}"]),
                        dcc.Graph(figure=fig, config={'displayModeBar': False})
                    ], className="p-3")
                ], className="glass-card"), width=8),
                dbc.Col([
                    dbc.Card([
                         html.H4("Indice de Performance" if lang=='FR' else "Performance Index", className="text-center mt-3", style={"color": "#fff"}),
                         html.Div([
                            html.I(className=f"fas {status_icon} fa-4x", style={"color": status_color, "marginBottom": "15px"}),
                            html.H3(deadline, style={"color": status_color, "fontWeight": "bold"}),
                            html.P(msg, className="text-muted small px-2"),
                            html.Hr(),
                            dbc.Progress(
                                value=min(100, (curr_val/median_ref)*100), 
                                color="danger" if curr_val > median_ref else "warning", 
                                striped=True, animated=True, style={"height": "10px"}
                            ),
                            html.Div(f"Gap : {max(0, curr_val - target_2030):.1f} Tonnes", className="small mt-2 text-muted text-center")
                        ], className="text-center p-3")
                    ], className="glass-card mb-4")
                ], width=4),
            ])
        ])
    except Exception as e:
        return dbc.Alert(f"Erreur d'affichage du benchmark : {str(e)}", color="danger")


# --- CALLBACKS ---

@app.callback(
    Output("sidebar-info-container", "children"),
    [Input("lang-switch", "value")]
)
def update_lang_nav(lang):
    return get_sidebar_content(lang or 'FR')

@app.callback(Output("page-content", "children"), [Input("url", "pathname"), Input("lang-switch", "value")])
def render_page_content(pathname, lang):
    lang = lang or 'FR'
    # Normalize pathname
    if pathname == "/": 
        return layout_insights(lang)
    elif pathname == "/analysis":
        return layout_analysis(lang)
    elif pathname == "/modeling":
        return layout_modeling(lang)
    elif pathname == "/predict":
        return layout_predict(lang)
    elif pathname == "/sim":
        return layout_sim(lang)
    elif pathname == "/star":
        return layout_star(lang)
    elif pathname == "/2050":
        return layout_benchmark(lang)
    return html.Div([
        html.H1("404: Not found", className="text-danger"),
        html.Hr(),
        html.P(f"Le chemin {pathname} n'est pas reconnu.")
    ], className="p-3 text-center")

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
    
    sources = energy_sources or []
    has_gas = "gas" in sources
    has_steam = "steam" in sources
    
    features = {
        'PrimaryPropertyType': b_type, 
        'Neighborhood': nbh,
        'PropertyGFATotal': surface, 
        'NumberofFloors': floors,
        'YearBuilt': year, 
        'ENERGYSTARScore': es_val,
        'Has_Gas': has_gas,
        'Has_Steam': has_steam
    }
    
    # 1. Prediction & XAI
    val, explanation = predict_co2(features)
    
    # Add hidden/default features for the report Specs
    features['NumberofFloors'] = floors
    features['ENERGYSTARScore'] = es_val
    
    df_xai = pd.DataFrame(explanation)
    # Define colors: Red for increasing CO2, Green for reducing CO2
    df_xai['color'] = df_xai['impact'].apply(lambda x: '#ff4d4d' if x > 0 else '#00fa9a')
    
    fig_xai = go.Figure(go.Bar(
        x=df_xai['impact'], 
        y=df_xai['feature'], 
        orientation='h',
        marker_color=df_xai['color']
    ))
    
    fig_xai.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0, r=20, t=30, b=20),
        title_text="Qu'est-ce qui influence ce résultat ?",
        title_font_size=14,
        xaxis_title=None,
        yaxis=dict(autorange="reversed") # To have critical factors at top
    )
    
    # 2. Reliability Badge
    rel = get_reliability_info(features)
    badge_colors = {"Green": "#00fa9a", "Orange": "#ffa500", "Red": "#ff4d4d"}
    badge_text = {"Green": "FIABLE", "Orange": "ATYPIQUE", "Red": "CRITIQUE"} if lang=='FR' else \
                {"Green": "RELIABLE", "Orange": "ATYPICAL", "Red": "CRITICAL"}
    
    badge = dbc.Badge(
        [html.I(className="fas fa-microscope me-1"), badge_text[rel['level']]],
        style={"backgroundColor": badge_colors[rel['level']], "color": "black", "fontSize": "0.9rem", "padding": "5px 12px"},
        id="rel-tooltip-target"
    )
    rel_tooltip = dbc.Tooltip(
        html.Div([
            html.B(f"Indice de Confiance : {rel['score']}%"),
            html.Ul([html.Li(r) for r in rel['reasons']]) if rel['reasons'] else html.P("Données conformes aux standards Seattle.")
        ]),
        target="rel-tooltip-target"
    )
    badge_container = html.Div([badge, rel_tooltip])

    # 3. Decarbonization Recommendations
    recs = get_decarbonization_recommendations(features, val)
    if not recs:
        decarbon_ui = html.Div([
            html.I(className="fas fa-leaf me-2", style={"color": "#00fa9a"}),
            html.Span("Mix énergétique déjà optimisé (Électrique)." if lang=='FR' else "Energy mix already optimized (Electric).")
        ], className="text-muted small")
    else:
        decarbon_ui = html.Div([
            html.B("🎯 STRATÉGIE DE DÉCARBONATION :" if lang=='FR' else "🎯 DECARBONIZATION STRATEGY :", style={"color": "#00fa9a"}),
            html.Div([
                dbc.Card([
                    html.Div([
                        html.Div([
                            html.Strong(r['title']),
                            html.P(r['action'], className="small mb-0", style={"color": "#aaa"})
                        ]),
                        html.Div([
                            html.H5(f"-{r['saving_pct']:.1f}%", className="text-success mb-0"),
                            html.Small(f"-{r['saving_tonnes']:.1f} T", className="text-muted")
                        ], className="text-end")
                    ], className="d-flex justify-content-between align-items-center p-2")
                ], className="mb-2", style={"backgroundColor": "rgba(255,255,255,0.05)", "border": "1px solid rgba(0,250,154,0.2)"}) for r in recs
            ])
        ])

    # Establish baseline if not exists
    new_baseline = val if baseline is None else baseline

    return f"{val:.2f}", badge_container, decarbon_ui, fig_xai, features, val, new_baseline

@app.callback(
    Output("download-pdf-obj", "data"),
    [Input("btn-download-pdf", "n_clicks")],
    [State("stored-building-features", "data"), State("prediction-output", "children"), State("lang-switch", "value")]
)
def download_pdf_report(n_clicks, features, prediction, lang):
    if not n_clicks or not features or not prediction or prediction == "":
        return dash.no_update
    
    try:
        pred_val = float(prediction)
    except (ValueError, TypeError):
        return dash.no_update
        
    # Re-calculate or use stored explanation if needed
    _, explanation = predict_co2(features)
    pdf_data_uri = generate_report_pdf(features, pred_val, explanation, lang)
    
    # generate_report_pdf returns: data:application/pdf;base64,BASE64_STRING
    try:
        content = pdf_data_uri.split(",")[1]
        return dict(content=content, filename="audit_seattle_notre_projet.pdf", base64=True)
    except Exception:
        return dash.no_update

@app.callback(
    [Output("in-es", "value"), Output("smart-es-note", "children")],
    [Input("btn-smart-es", "n_clicks")],
    [State("in-type", "value"), State("lang-switch", "value")]
)
def update_smart_es(n_clicks, b_type, lang):
    if not n_clicks:
        return dash.no_update, ""
    
    suggestion = get_smart_suggestions(b_type)
    return suggestion['suggested_es'], suggestion['note']

@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents'), Input('btn-clear-batch', 'n_clicks')],
    [State('upload-data', 'filename'), State('lang-switch', 'value')]
)
def update_output(contents, n_clicks, filename, lang):
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'btn-clear-batch':
        return None
        
    if contents is not None:
        return parse_contents(contents, filename, lang)
    
    return dash.no_update

@app.callback(
    Output("btn-clear-batch", "style"),
    [Input("output-data-upload", "children")]
)
def toggle_clear_button(children):
    if children:
        return {"display": "block"}
    return {"display": "none"}

def parse_contents(contents, filename, lang):
    lang = lang or 'FR'
    t = TRANSLATIONS[lang]
    content_type, content_string = contents.split(',')
    decoded = base64.decodebytes(content_string.encode('utf-8'))
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return html.Div(['Ce fichier n\'est pas un CSV.' if lang=='FR' else 'Not a CSV file.'], className="text-danger")
        
        # Security Limit
        if len(df) > 1000:
            return dbc.Alert(t['batch_err_limit'], color="warning", className="mt-2")

        # Normalize and validate columns (Case-insensitive)
        required = ['PrimaryPropertyType', 'PropertyGFATotal', 'YearBuilt', 'ENERGYSTARScore', 'Neighborhood', 'NumberofFloors']
        extra_optional = ['Has_Gas', 'Has_Steam']
        
        col_map = {c.lower().strip(): c for c in df.columns}
        rename_dict = {}
        missing = []
        
        for col in required:
            if col.lower() in col_map:
                rename_dict[col_map[col.lower()]] = col
            else:
                missing.append(col)
        
        for col in extra_optional:
            if col.lower() in col_map:
                rename_dict[col_map[col.lower()]] = col
                
        if missing:
            return dbc.Alert([
                html.Strong(t['batch_err_missing']),
                html.Span(", ".join(missing))
            ], color="danger", className="mt-2")

        df = df.rename(columns=rename_dict)

        # Process results
        results = []
        df = df.replace({np.nan: None}) # Replace NaN with None for row.get()
        
        for index, row in df.iterrows():
            f = {
                'PrimaryPropertyType': row.get('PrimaryPropertyType', 'Office'),
                'PropertyGFATotal': row.get('PropertyGFATotal', 50000),
                'YearBuilt': row.get('YearBuilt', 1980),
                'ENERGYSTARScore': row.get('ENERGYSTARScore', 50),
                'Neighborhood': row.get('Neighborhood', 'Downtown'),
                'NumberofFloors': row.get('NumberofFloors', 1),
                'Has_Gas': str(row.get('Has_Gas', '')).lower() in ['true', '1', 'yes'],
                'Has_Steam': str(row.get('Has_Steam', '')).lower() in ['true', '1', 'yes']
            }
            v, _ = predict_co2(f)
            results.append(round(v, 2))
        
        df['Predicted_CO2(T)'] = results
        
        # 1. Batch Statistics for Dashboard
        total_emissions = df['Predicted_CO2(T)'].sum()
        avg_star = df['ENERGYSTARScore'].mean() if 'ENERGYSTARScore' in df.columns else 0
        max_emiss = df['Predicted_CO2(T)'].max()
        count = len(df)

        # 2. Batch Summary Graph
        fig_batch = px.histogram(df, x='Predicted_CO2(T)', 
                                 title="Distribution des Prédictions du Lot" if lang=='FR' else "Batch Predictions Distribution",
                                 color_discrete_sequence=['#00fa9a'],
                                 nbins=min(count, 20),
                                 template="plotly_dark")
        fig_batch.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            height=300, margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Tonnes CO2", yaxis_title="Nombre de Bâtiments"
        )

        return html.Div([
            # Dashboard Batch Summary
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("TOTAL LOT", className="stat-label"),
                    html.Div(f"{total_emissions:,.1f}", className="stat-value", style={"fontSize": "1.8rem"}),
                    html.Div("Tonnes CO2 / an", className="small text-muted")
                ], className="text-center p-3 glass-card"), width=4),
                dbc.Col(html.Div([
                    html.Div("SCORE MOYEN", className="stat-label"),
                    html.Div(f"{avg_star:.1f}", className="stat-value", style={"fontSize": "1.8rem", "color": "#ffd700"}),
                    html.Div("Energy Star", className="small text-muted")
                ], className="text-center p-3 glass-card"), width=4),
                dbc.Col(html.Div([
                    html.Div("MAX EMISSION", className="stat-label"),
                    html.Div(f"{max_emiss:.1f}", className="stat-value", style={"fontSize": "1.8rem", "color": "#ff4d4d"}),
                    html.Div("Tonnes / bâtiment", className="small text-muted")
                ], className="text-center p-3 glass-card"), width=4),
            ], className="mb-4 g-3"),

            html.Div([
                html.H6(f"📊 {t['batch_title']} : {filename} ({count} bâtiments)", style={"display": "inline-block"}),
            ], className="d-flex justify-content-between align-items-center mb-3"),
            
            dcc.Graph(figure=fig_batch, className="mb-4", config={'displayModeBar': False}),

            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=10,
                export_format='csv',
                export_headers='display',
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1E1E1E', 'color': '#00fa9a', 'fontWeight': 'bold'},
                style_cell={'backgroundColor': '#2D2D2D', 'color': 'white', 'textAlign': 'left', 'padding': '10px'},
                style_data_conditional=[{
                    'if': {'column_id': 'Predicted_CO2(T)'},
                    'color': '#00fa9a',
                    'fontWeight': 'bold'
                }]
            )
        ])
    except Exception as e:
        return html.Div([f'Erreur de lecture : {str(e)}'], className="text-danger")


@app.callback(
    Output("year-display", "children"),
    [Input("in-year", "value"), Input("lang-switch", "value")]
)
def update_year_display(year, lang):
    lang = lang or 'FR'
    text = "Année : " if lang == 'FR' else "Year: "
    return f"{text} {year}" if year else ""

@app.callback(
    Output("sim-boost-display", "children"),
    [Input("sim-es-boost", "value"), Input("lang-switch", "value")]
)
def update_sim_boost_display(boost, lang):
    lang = lang or 'FR'
    return f"+ {boost} points" if lang == 'FR' else f"+ {boost} pts"

@app.callback(
    [Output("eda-modal", "is_open"), Output("eda-modal-img", "src"), Output("eda-modal-header", "children")],
    [Input({'type': 'eda-image', 'index': ALL, 'title': ALL}, 'n_clicks'), Input("eda-modal-close", "n_clicks")],
    [State("eda-modal", "is_open")]
)

def toggle_eda_zoom(n_clicks_list, close_click, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, "", ""
    
    triggered_id = ctx.triggered[0]['prop_id']
    
    # Handle closing
    if "eda-modal-close" in triggered_id:
        return False, "", ""
    
    # Handle opening (pattern matching)
    # Check if any image was actually clicked
    if any(n_clicks_list):
        import json
        # In Dash 2.0+, triggered_id is a dict if it's pattern matching
        # But for triggered[0]['prop_id'], it's a string like '{"index":"...","title":"...","type":"eda-image"}.n_clicks'
        try:
            prop_id_str = triggered_id.split('.')[0]
            info = json.loads(prop_id_str)
            return True, info['index'], info['title']
        except:
            return False, "", ""
            
    return is_open, "", ""



@app.callback(
    [Output("stored-building-features", "data", allow_duplicate=True), 
     Output("map-toast", "is_open"), 
     Output("map-toast", "children")],
    [Input("insights-map", "clickData")],
    [State("stored-building-features", "data"), State("lang-switch", "value")],
    prevent_initial_call=True
)
def handle_map_click(clickData, current_data, lang):
    if not clickData:
        return dash.no_update, False, ""
    
    neighborhood = clickData['points'][0]['hovertext']
    stats = NEIGHBORHOOD_STATS.get(neighborhood)
    
    if stats:
        new_data = current_data.copy()
        new_data['Neighborhood'] = neighborhood
        new_data['PropertyGFATotal'] = stats['avg_gfa']
        new_data['YearBuilt'] = stats['avg_year']
        new_data['NumberofFloors'] = stats['avg_floors']
        
        msg = html.Div([
            html.P(f"✅ Quartier {neighborhood} sélectionné." if lang=='FR' else f"✅ {neighborhood} selected.", className="mb-0")
        ])
        return new_data, True, msg
        
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
    # Retrieve current layout context to verify if IDs exist (Anti-Callback Error)
    ctx = dash.callback_context
    if not ctx:
        return dash.no_update
        
    # If no selection, show city-wide stats
    if not selected_nbh:
        return (
            f"{CITY_WIDE_STATS['total_buildings']}",
            f"{CITY_WIDE_STATS['avg_emissions']} T",
            f"{CITY_WIDE_STATS['avg_energy_star']}"
        )
    
    # Otherwise calculate stats for selected neighborhoods
    selected_stats = [NEIGHBORHOOD_STATS[n] for n in selected_nbh if n in NEIGHBORHOOD_STATS]
    
    if not selected_stats: return "0", "0 T", "0"

    count = sum(s['count'] for s in selected_stats)
    # Weighted average for CO2 and Energy Star
    total_co2 = sum(s['avg_co2'] * s['count'] for s in selected_stats)
    total_estar = sum(s['avg_estar'] * s['count'] for s in selected_stats)

    avg_co2 = total_co2 / count if count else 0
    avg_estar = total_estar / count if count else 0

    return (
        f"{int(count)}",
        f"{avg_co2:.1f} T",
        f"{int(avg_estar)}"
    )

@app.callback(
    Output("insights-map", "figure"),
    [Input("map-filter-nbh", "value"), Input("lang-switch", "value")]
)
def update_insights_map(selected_nbh, lang):
    lang = lang or 'FR'
    t = TRANSLATIONS[lang]
    
    rows = []
    for name, stats in NEIGHBORHOOD_STATS.items():
        if not selected_nbh or name in selected_nbh:
            rows.append({
                'Neighborhood': name,
                'lat': stats['lat'],
                'lon': stats['lon'],
                'emissions': stats['avg_co2'],
                'avg_gfa': stats['avg_gfa'],
                'avg_year': stats['avg_year'],
                'avg_floors': stats['avg_floors'],
                'description': f"{stats['avg_co2']} T (Moyenne)"
            })
    df_map = pd.DataFrame(rows)
    
    if df_map.empty:
        # Prevent error if nothing selected
        fig = go.Figure().update_layout(template="plotly_dark", title="Aucune donnée sélectionnée" if lang=='FR' else "No data selected")
        return fig

    # Translation for labels
    labels = {
        'emissions': 'Emissions (T)',
        'avg_gfa': 'Surface (sq ft)',
        'avg_year': 'Année Const.',
        'avg_floors': 'Étages'
    } if lang=='FR' else {
        'emissions': 'Emissions (T)',
        'avg_gfa': 'Average GFA',
        'avg_year': 'Avg Year Built',
        'avg_floors': 'Floors'
    }

    fig_map = px.scatter_map(df_map, lat="lat", lon="lon", size="emissions", color="emissions",
                               hover_name="Neighborhood", 
                               hover_data={
                                   'lat': False, 'lon': False, 
                                   'emissions': ':.1f',
                                   'avg_gfa': ':,.0f',
                                   'avg_year': True,
                                   'avg_floors': True
                               },
                               labels=labels,
                               color_continuous_scale=px.colors.sequential.Viridis, 
                               size_max=35, zoom=10.2,
                               map_style="open-street-map")
                               
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, 
        paper_bgcolor='rgba(0,0,0,0)', 
        font_color="#ccc",
        clickmode='event+select',
        coloraxis_colorbar=dict(
            title="CO2 (T)",
            thickness=15, 
            len=0.5,
            yanchor="middle", y=0.5,
            bgcolor='rgba(0,0,0,0.4)',
            ticks="outside"
        )
    )
    return fig_map

@app.callback(
    Output("sidebar-toggle-stored", "data"),
    [Input("btn-toggle-sidebar", "n_clicks")],
    [State("sidebar-toggle-stored", "data")]
)
def toggle_sidebar(n_clicks, current_state):
    if n_clicks:
        return not current_state
    return current_state

@app.callback(
    [Output("sidebar-container", "className"), 
     Output("page-content", "className"),
     Output("btn-toggle-sidebar", "className")],
    [Input("sidebar-toggle-stored", "data")]
)
def update_sidebar_layout(is_open):
    side_class = "sidebar" if is_open else "sidebar sidebar-hidden"
    content_class = "" if is_open else "content-collapsed"
    btn_class = "sidebar-toggle-btn-inline" if is_open else "sidebar-toggle-btn-inline btn-floating"
    return side_class, content_class, btn_class

if __name__ == "__main__":

    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(
        debug= True, 
        port=8050
    )



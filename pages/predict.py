"""
Page Predict - Calculateur CO2 avec prédictions et batch processing
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def layout(lang='FR'):
    """
    Layout de la page Predict avec formulaire de prédiction
    """
    t = get_translations(lang)
    
    # Listes temporaires (à remplacer par les vraies données)
    BUILDING_TYPES = [
        'Office', 'Hotel', 'Large Office', 'Retail Store', 
        'Non-Refrigerated Warehouse', 'K-12 School', 'Medical Office',
        'Small- and Mid-Sized Office', 'Self-Storage Facility',
        'Distribution Center', 'Senior Care Community'
    ]
    
    NEIGHBORHOODS = [
        'Downtown', 'Magnolia / Queen Anne', 'Greater Duwamish',
        'Lake Union', 'East', 'Northeast', 'Northwest', 'South',
        'Southeast', 'Central', 'Ballard'
    ]
    
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


def get_translations(lang):
    """Traductions temporaires"""
    if lang == 'FR':
        return {
            'predict': 'Calculateur CO2',
            'chars_title': 'Caractéristiques du Bâtiment',
            'label_usage': 'Type d\'Usage',
            'label_nbh': 'Quartier',
            'label_surface': 'Surface Totale (sq ft)',
            'label_floors': 'Nombre d\'Étages',
            'label_year': 'Année de Construction',
            'label_es': 'Score ENERGY STAR',
            'label_gas': 'Gaz Naturel',
            'label_steam': 'Vapeur Urbaine',
            'predict_btn': 'Prédire les Émissions',
            'out_predicted': 'Émissions Prédites',
            'out_unit': 'Tonnes CO2 / an',
            'out_xai': 'Facteurs d\'Impact',
            'download_pdf': 'Télécharger le Rapport PDF',
            'batch_title': 'Prédictions par Lot',
            'batch_text': 'Uploadez un fichier CSV pour prédire les émissions de plusieurs bâtiments.'
        }
    else:
        return {
            'predict': 'CO2 Calculator',
            'chars_title': 'Building Characteristics',
            'label_usage': 'Usage Type',
            'label_nbh': 'Neighborhood',
            'label_surface': 'Total Surface (sq ft)',
            'label_floors': 'Number of Floors',
            'label_year': 'Year Built',
            'label_es': 'ENERGY STAR Score',
            'label_gas': 'Natural Gas',
            'label_steam': 'District Steam',
            'predict_btn': 'Predict Emissions',
            'out_predicted': 'Predicted Emissions',
            'out_unit': 'Tonnes CO2 / year',
            'out_xai': 'Impact Factors',
            'download_pdf': 'Download PDF Report',
            'batch_title': 'Batch Predictions',
            'batch_text': 'Upload a CSV file to predict emissions for multiple buildings.'
        }

"""
Page Analysis - Galerie EDA avec visualisations
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def layout(lang='FR'):
    """
    Layout de la page Analysis avec galerie de graphiques EDA
    """
    t = get_translations(lang)
    
    sections = [
        {
            "id": "audit",
            "title": "Audit de la Donnée brute" if lang=='FR' else "Raw Data Audit",
            "desc": "Analyse de la complétude et nettoyage des valeurs manquantes.",
            "images": [{"src": "/assets/eda_figures/auto_missing_values.png", "title": "Bilan des Manquants", "width": 12}]
        },
        {
            "id": "target",
            "title": "Analyse de la Cible" if lang=='FR' else "Target Analysis",
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


def build_gallery_ui(title, sections, lang):
    """Construit l'interface de la galerie"""
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


def get_translations(lang):
    """Traductions temporaires"""
    if lang == 'FR':
        return {'analysis': 'Analyse Exploratoire'}
    else:
        return {'analysis': 'Exploratory Analysis'}

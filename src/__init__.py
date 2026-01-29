"""
REGO3 - Package de Prétraitement et Utilitaires
================================================

Ce package contient tous les modules pour le projet REGO3.

Modules disponibles :
- preprocessing : Pipelines de traitement des données
- feature_engineering : Création de nouvelles features
- utils : Fonctions utilitaires

Auteur : Équipe REGO3
Date : Janvier 2026
"""

__version__ = '1.0.0'
__author__ = 'Équipe REGO3'

# Import des fonctions principales pour faciliter l'utilisation
from .preprocessing import (
    fit_missing_values_pipeline,
    transform_missing_values_pipeline,
    fit_outliers_pipeline,
    transform_outliers_pipeline,
    fit_binary_features_pipeline,
    transform_binary_features_pipeline,
    fit_categorical_encoder,
    transform_categorical_encoder,
    fit_complete_pipeline,
    transform_complete_pipeline,
    print_pipeline_summary
)

from .feature_engineering import (
    create_ratio_features,
    create_interaction_features,
    create_temporal_features,
    create_polynomial_features,
    fit_aggregated_features,
    transform_aggregated_features,
    create_all_features,
    print_feature_summary
)

__all__ = [
    # Preprocessing functions
    'fit_missing_values_pipeline',
    'transform_missing_values_pipeline',
    'fit_outliers_pipeline',
    'transform_outliers_pipeline',
    'fit_binary_features_pipeline',
    'transform_binary_features_pipeline',
    'fit_categorical_encoder',
    'transform_categorical_encoder',
    'fit_complete_pipeline',
    'transform_complete_pipeline',
    'print_pipeline_summary',
    # Feature engineering functions
    'create_ratio_features',
    'create_interaction_features',
    'create_temporal_features',
    'create_polynomial_features',
    'fit_aggregated_features',
    'transform_aggregated_features',
    'create_all_features',
    'print_feature_summary'
]

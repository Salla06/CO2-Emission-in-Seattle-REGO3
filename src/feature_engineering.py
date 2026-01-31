"""
REGO3 - Module de Feature Engineering
======================================

Ce module contient toutes les fonctions pour créer de nouvelles features
à partir des variables existantes.

Principe : FIT sur train, TRANSFORM sur train et test
- Les statistiques (moyennes par groupe, etc.) sont calculées sur train uniquement
- Ces mêmes statistiques sont appliquées au test

Types de features créées :
1. Ratios : Combiner deux variables (division)
2. Interactions : Multiplier deux variables
3. Features temporelles : Âge, ancienneté
4. Agrégations : Moyennes par groupe (quartier, type)
5. Features polynomiales : Puissances, racines

Auteur : Équipe REGO3
Date : Janvier 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# FEATURES RATIOS
# ============================================================================

def create_ratio_features(
    df: pd.DataFrame,
    ratio_definitions: Dict[str, Tuple[str, str]]
) -> pd.DataFrame:
    """
    Crée des features de type ratio (division de deux variables).
    
    Exemple : Surface par étage = PropertyGFATotal / NumberofFloors
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset (train ou test)
    ratio_definitions : dict
        Dictionnaire {nom_nouvelle_feature: (numerateur, denominateur)}
    
    Retourne
    --------
    df_with_ratios : pd.DataFrame
        Dataset avec les nouvelles features ratios
    
    Exemple
    -------
    >>> ratios = {
    ...     'GFA_per_floor': ('PropertyGFATotal', 'NumberofFloors'),
    ...     'Parking_ratio': ('PropertyGFAParking', 'PropertyGFATotal')
    ... }
    >>> df_new = create_ratio_features(df, ratios)
    """
    df_with_ratios = df.copy()
    
    for feature_name, (numerator, denominator) in ratio_definitions.items():
        if numerator in df.columns and denominator in df.columns:
            # Éviter division par zéro
            df_with_ratios[feature_name] = df[numerator] / (df[denominator] + 1e-6)
            
            # Remplacer inf et -inf par NaN
            df_with_ratios[feature_name].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df_with_ratios


# ============================================================================
# FEATURES INTERACTIONS
# ============================================================================

def create_interaction_features(
    df: pd.DataFrame,
    interaction_definitions: Dict[str, Tuple[str, str]]
) -> pd.DataFrame:
    """
    Crée des features d'interaction (multiplication de deux variables).
    
    Exemple : Size_floors = PropertyGFATotal * NumberofFloors
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset (train ou test)
    interaction_definitions : dict
        Dictionnaire {nom_nouvelle_feature: (variable1, variable2)}
    
    Retourne
    --------
    df_with_interactions : pd.DataFrame
        Dataset avec les nouvelles features d'interaction
    
    Exemple
    -------
    >>> interactions = {
    ...     'Size_floors': ('PropertyGFATotal', 'NumberofFloors'),
    ...     'Age_size': ('Building_age', 'PropertyGFATotal')
    ... }
    >>> df_new = create_interaction_features(df, interactions)
    """
    df_with_interactions = df.copy()
    
    for feature_name, (var1, var2) in interaction_definitions.items():
        if var1 in df.columns and var2 in df.columns:
            df_with_interactions[feature_name] = df[var1] * df[var2]
    
    return df_with_interactions


# ============================================================================
# FEATURES TEMPORELLES
# ============================================================================

def create_temporal_features(
    df: pd.DataFrame,
    reference_year: int = 2016
) -> pd.DataFrame:
    """
    Crée des features temporelles basées sur les dates.
    
    Features créées :
    - Building_age : Âge du bâtiment (reference_year - YearBuilt)
    - Building_age_squared : Âge au carré (relation non-linéaire)
    - Is_old_building : 1 si âge > 50 ans
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset (train ou test)
    reference_year : int, optional
        Année de référence (défaut: 2016, année du dataset)
    
    Retourne
    --------
    df_with_temporal : pd.DataFrame
        Dataset avec les nouvelles features temporelles
    
    Exemple
    -------
    >>> df_new = create_temporal_features(df, reference_year=2016)
    """
    df_with_temporal = df.copy()
    
    if 'YearBuilt' in df.columns:
        # Âge du bâtiment
        df_with_temporal['Building_age'] = reference_year - df['YearBuilt']
        
        # Remplacer les âges négatifs (erreurs de données) par 0
        df_with_temporal['Building_age'] = df_with_temporal['Building_age'].clip(lower=0)
        
        # Âge au carré (pour capturer effets non-linéaires)
        df_with_temporal['Building_age_squared'] = df_with_temporal['Building_age'] ** 2
        
        # Indicateur bâtiment ancien (>50 ans)
        df_with_temporal['Is_old_building'] = (df_with_temporal['Building_age'] > 50).astype(int)
    
    return df_with_temporal


# ============================================================================
# FEATURES POLYNOMIALES
# ============================================================================

def create_polynomial_features(
    df: pd.DataFrame,
    polynomial_definitions: Dict[str, Tuple[str, int]]
) -> pd.DataFrame:
    """
    Crée des features polynomiales (puissances de variables).
    
    Exemple : PropertyGFATotal_squared = PropertyGFATotal^2
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset (train ou test)
    polynomial_definitions : dict
        Dictionnaire {nom_nouvelle_feature: (variable_source, puissance)}
    
    Retourne
    --------
    df_with_poly : pd.DataFrame
        Dataset avec les nouvelles features polynomiales
    
    Exemple
    -------
    >>> polys = {
    ...     'GFA_squared': ('PropertyGFATotal', 2),
    ...     'GFA_sqrt': ('PropertyGFATotal', 0.5)
    ... }
    >>> df_new = create_polynomial_features(df, polys)
    """
    df_with_poly = df.copy()
    
    for feature_name, (source_var, power) in polynomial_definitions.items():
        if source_var in df.columns:
            # Pour les racines, s'assurer que les valeurs sont positives
            if power < 1:
                df_with_poly[feature_name] = np.power(df[source_var].clip(lower=0), power)
            else:
                df_with_poly[feature_name] = np.power(df[source_var], power)
    
    return df_with_poly


# ============================================================================
# FEATURES AGRÉGÉES (avec fit/transform pour éviter leakage)
# ============================================================================

def fit_aggregated_features(
    df_train: pd.DataFrame,
    target_col: str,
    groupby_cols: List[str],
    agg_functions: List[str] = ['mean', 'median', 'std']
) -> Dict:
    """
    Calcule les statistiques agrégées par groupe sur le train set.
    
    Exemple : Émission moyenne par quartier, émission médiane par type de bâtiment
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    target_col : str
        Colonne cible à agréger
    groupby_cols : list
        Colonnes pour le groupby
    agg_functions : list, optional
        Fonctions d'agrégation (défaut: ['mean', 'median', 'std'])
    
    Retourne
    --------
    params : dict
        Dictionnaire contenant les statistiques agrégées par groupe
    
    Exemple
    -------
    >>> params = fit_aggregated_features(
    ...     train_df, 
    ...     target_col='TotalGHGEmissions_log',
    ...     groupby_cols=['Neighborhood', 'PrimaryPropertyType']
    ... )
    """
    params = {
        'target_col': target_col,
        'groupby_cols': groupby_cols,
        'agg_functions': agg_functions,
        'aggregations': {}
    }
    
    for groupby_col in groupby_cols:
        if groupby_col in df_train.columns:
            params['aggregations'][groupby_col] = {}
            
            for agg_func in agg_functions:
                # Calculer l'agrégation
                if agg_func == 'mean':
                    agg_values = df_train.groupby(groupby_col)[target_col].mean()
                elif agg_func == 'median':
                    agg_values = df_train.groupby(groupby_col)[target_col].median()
                elif agg_func == 'std':
                    agg_values = df_train.groupby(groupby_col)[target_col].std()
                elif agg_func == 'count':
                    agg_values = df_train.groupby(groupby_col)[target_col].count()
                else:
                    continue
                
                params['aggregations'][groupby_col][agg_func] = agg_values.to_dict()
    
    # Calculer la moyenne globale pour les groupes inconnus
    params['global_mean'] = df_train[target_col].mean()
    
    return params


def transform_aggregated_features(
    df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Applique les statistiques agrégées calculées sur le train.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset (train ou test)
    params : dict
        Paramètres calculés avec fit_aggregated_features()
    
    Retourne
    --------
    df_with_agg : pd.DataFrame
        Dataset avec les nouvelles features agrégées
    
    Exemple
    -------
    >>> params = fit_aggregated_features(train_df, 'target', ['Neighborhood'])
    >>> train_new = transform_aggregated_features(train_df, params)
    >>> test_new = transform_aggregated_features(test_df, params)
    """
    df_with_agg = df.copy()
    
    for groupby_col, agg_dict in params['aggregations'].items():
        if groupby_col in df.columns:
            for agg_func, mapping in agg_dict.items():
                feature_name = f"{groupby_col}_{agg_func}"
                
                # Mapper les valeurs
                df_with_agg[feature_name] = df[groupby_col].map(mapping)
                
                # Remplacer les groupes inconnus par la moyenne globale
                df_with_agg[feature_name].fillna(params['global_mean'], inplace=True)
    
    return df_with_agg


# ============================================================================
# FONCTION COMPLÈTE : CRÉER TOUTES LES FEATURES
# ============================================================================

def create_all_features(
    df: pd.DataFrame,
    ratio_defs: Optional[Dict] = None,
    interaction_defs: Optional[Dict] = None,
    polynomial_defs: Optional[Dict] = None,
    reference_year: int = 2016
) -> pd.DataFrame:
    """
    Crée toutes les features en une seule fois.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset (train ou test)
    ratio_defs : dict, optional
        Définitions des ratios
    interaction_defs : dict, optional
        Définitions des interactions
    polynomial_defs : dict, optional
        Définitions des polynômes
    reference_year : int, optional
        Année de référence pour features temporelles
    
    Retourne
    --------
    df_enriched : pd.DataFrame
        Dataset avec toutes les nouvelles features
    
    Exemple
    -------
    >>> df_new = create_all_features(
    ...     df,
    ...     ratio_defs={'GFA_per_floor': ('PropertyGFATotal', 'NumberofFloors')},
    ...     reference_year=2016
    ... )
    """
    df_enriched = df.copy()
    
    # 1. Ratios
    if ratio_defs:
        df_enriched = create_ratio_features(df_enriched, ratio_defs)
    
    # 2. Features temporelles
    df_enriched = create_temporal_features(df_enriched, reference_year)
    
    # 3. Interactions
    if interaction_defs:
        df_enriched = create_interaction_features(df_enriched, interaction_defs)
    
    # 4. Polynomiales
    if polynomial_defs:
        df_enriched = create_polynomial_features(df_enriched, polynomial_defs)
    
    return df_enriched


# ============================================================================
# FONCTION UTILITAIRE : AFFICHAGE
# ============================================================================

def print_feature_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> None:
    """
    Affiche un résumé des features créées.
    
    Paramètres
    ----------
    df_before : pd.DataFrame
        Dataset avant feature engineering
    df_after : pd.DataFrame
        Dataset après feature engineering
    """
    n_features_before = df_before.shape[1]
    n_features_after = df_after.shape[1]
    n_new_features = n_features_after - n_features_before
    
    new_features = [col for col in df_after.columns if col not in df_before.columns]
    
    print("=" * 80)
    print("RÉSUMÉ FEATURE ENGINEERING")
    print("=" * 80)
    print(f"\nFeatures avant : {n_features_before}")
    print(f"Features après : {n_features_after}")
    print(f"Nouvelles features créées : {n_new_features}")
    
    if new_features:
        print(f"\nListe des nouvelles features :")
        for i, feature in enumerate(new_features, 1):
            print(f"  {i}. {feature}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    """
    Exemple d'utilisation du module
    """
    print(__doc__)
    print("\nCe module est conçu pour être importé dans les notebooks.")
    print("\nExemple d'utilisation :")
    print("""
    from src.feature_engineering import (
        create_ratio_features,
        create_temporal_features,
        fit_aggregated_features,
        transform_aggregated_features
    )
    
    # Créer ratios
    ratios = {'GFA_per_floor': ('PropertyGFATotal', 'NumberofFloors')}
    df_new = create_ratio_features(df, ratios)
    
    # Créer features temporelles
    df_new = create_temporal_features(df_new, reference_year=2016)
    
    # Créer agrégations (FIT/TRANSFORM)
    agg_params = fit_aggregated_features(train_df, 'target', ['Neighborhood'])
    train_new = transform_aggregated_features(train_df, agg_params)
    test_new = transform_aggregated_features(test_df, agg_params)
    """)

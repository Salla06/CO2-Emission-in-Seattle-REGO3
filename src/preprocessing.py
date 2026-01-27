"""
REGO3 - Module de Prétraitement des Données
============================================

Ce module contient tous les pipelines de traitement des données pour le projet REGO3.

Principe fondamental : FIT sur train, TRANSFORM sur train et test
- Les statistiques (moyennes, modes, limites) sont calculées sur le train set uniquement
- Ces mêmes statistiques sont appliquées au test set
- Cela évite le data leakage

Auteur : Équipe REGO3
Date : Janvier 2026
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from typing import Dict, List, Tuple, Optional, Union


# ============================================================================
# PIPELINE VALEURS MANQUANTES
# ============================================================================

def fit_missing_values_pipeline(
    df_train: pd.DataFrame,
    threshold_missing: float = 0.70,
    threshold_unique: int = 1,
    threshold_impute: float = 0.10
) -> Dict:
    """
    Apprend les règles de traitement des valeurs manquantes sur le train set.
    
    Règles appliquées :
    1. Supprimer colonnes avec plus de threshold_missing% de valeurs manquantes
    2. Supprimer colonnes avec threshold_unique valeur(s) unique(s)
    3. Pour colonnes avec moins de threshold_impute% de valeurs manquantes :
       - Variables quantitatives : imputation par la moyenne
       - Variables qualitatives : imputation par le mode
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    threshold_missing : float, optional
        Seuil de suppression pour valeurs manquantes (défaut: 0.70)
    threshold_unique : int, optional
        Nombre minimum de valeurs uniques (défaut: 1)
    threshold_impute : float, optional
        Seuil pour imputation (défaut: 0.10)
    
    Retourne
    --------
    params : dict
        Dictionnaire contenant :
        - cols_to_drop_missing : liste des colonnes à supprimer (>70% missing)
        - cols_to_drop_unique : liste des colonnes à supprimer (valeur unique)
        - impute_mean : dict {colonne: valeur_moyenne}
        - impute_mode : dict {colonne: valeur_mode}
    
    Exemple
    -------
    >>> params = fit_missing_values_pipeline(train_df)
    >>> print(params['impute_mean'])
    {'NumberofFloors': 5.3, 'PropertyGFATotal': 45000.2}
    """
    params = {
        'cols_to_drop_missing': [],
        'cols_to_drop_unique': [],
        'impute_mean': {},
        'impute_mode': {}
    }
    
    # Règle 1 : Colonnes avec >threshold_missing% de valeurs manquantes
    missing_pct = df_train.isnull().sum() / len(df_train)
    params['cols_to_drop_missing'] = missing_pct[missing_pct > threshold_missing].index.tolist()
    
    # Supprimer ces colonnes temporairement pour les analyses suivantes
    df_temp = df_train.drop(columns=params['cols_to_drop_missing'], errors='ignore')
    
    # Règle 2 : Colonnes avec valeur unique
    for col in df_temp.columns:
        if df_temp[col].nunique() <= threshold_unique:
            params['cols_to_drop_unique'].append(col)
    
    # Supprimer ces colonnes aussi
    df_temp = df_temp.drop(columns=params['cols_to_drop_unique'], errors='ignore')
    
    # Règle 3 : Imputation pour colonnes avec <threshold_impute% de valeurs manquantes
    for col in df_temp.columns:
        missing_pct_col = df_temp[col].isnull().sum() / len(df_temp)
        
        if 0 < missing_pct_col < threshold_impute:
            if df_temp[col].dtype in ['int64', 'float64']:
                # Variable quantitative : moyenne
                params['impute_mean'][col] = df_temp[col].mean()
            else:
                # Variable qualitative : mode
                mode_val = df_temp[col].mode()
                params['impute_mode'][col] = mode_val[0] if not mode_val.empty else 'Unknown'
    
    return params


def transform_missing_values_pipeline(
    df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Applique les règles de traitement des valeurs manquantes sur un dataset.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset à traiter (train ou test)
    params : dict
        Paramètres appris avec fit_missing_values_pipeline()
    
    Retourne
    --------
    df_processed : pd.DataFrame
        Dataset traité
    
    Exemple
    -------
    >>> params = fit_missing_values_pipeline(train_df)
    >>> train_clean = transform_missing_values_pipeline(train_df, params)
    >>> test_clean = transform_missing_values_pipeline(test_df, params)
    """
    df_processed = df.copy()
    
    # Supprimer colonnes avec trop de missing
    cols_to_drop = [col for col in params['cols_to_drop_missing'] if col in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
    
    # Supprimer colonnes avec valeur unique
    cols_to_drop = [col for col in params['cols_to_drop_unique'] if col in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
    
    # Imputation par la moyenne (valeurs du train)
    for col, mean_val in params['impute_mean'].items():
        if col in df_processed.columns:
            df_processed[col].fillna(mean_val, inplace=True)
    
    # Imputation par le mode (valeurs du train)
    for col, mode_val in params['impute_mode'].items():
        if col in df_processed.columns:
            df_processed[col].fillna(mode_val, inplace=True)
    
    return df_processed


# ============================================================================
# PIPELINE VALEURS EXTRÊMES
# ============================================================================

def fit_outliers_pipeline(
    df_train: pd.DataFrame,
    threshold: float = 3,
    method: str = 'zscore'
) -> Dict:
    """
    Apprend les limites pour le traitement des valeurs extrêmes sur le train set.
    
    Méthode Z-score :
    - Calcule moyenne et écart-type sur train
    - Définit limites : moyenne ± threshold * écart-type
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    threshold : float, optional
        Seuil du Z-score (défaut: 3 pour ±3 écarts-types)
    method : str, optional
        Méthode de détection (défaut: 'zscore')
    
    Retourne
    --------
    params : dict
        Dictionnaire contenant :
        - threshold : seuil utilisé
        - method : méthode utilisée
        - limits : dict {colonne: {'mean', 'std', 'lower', 'upper'}}
    
    Exemple
    -------
    >>> params = fit_outliers_pipeline(train_df, threshold=3)
    >>> print(params['limits']['PropertyGFATotal'])
    {'mean': 45000, 'std': 20000, 'lower': -15000, 'upper': 105000}
    """
    params = {
        'threshold': threshold,
        'method': method,
        'limits': {}
    }
    
    # Pour chaque colonne numérique
    numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        mean_val = df_train[col].mean()
        std_val = df_train[col].std()
        
        params['limits'][col] = {
            'mean': mean_val,
            'std': std_val,
            'lower': mean_val - (threshold * std_val),
            'upper': mean_val + (threshold * std_val)
        }
    
    return params


def transform_outliers_pipeline(
    df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Applique les limites pour traiter les valeurs extrêmes (capping).
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset à traiter (train ou test)
    params : dict
        Paramètres appris avec fit_outliers_pipeline()
    
    Retourne
    --------
    df_processed : pd.DataFrame
        Dataset avec valeurs extrêmes cappées
    
    Exemple
    -------
    >>> params = fit_outliers_pipeline(train_df)
    >>> train_clean = transform_outliers_pipeline(train_df, params)
    >>> test_clean = transform_outliers_pipeline(test_df, params)
    """
    df_processed = df.copy()
    
    for col, limits in params['limits'].items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].clip(
                lower=limits['lower'],
                upper=limits['upper']
            )
    
    return df_processed


# ============================================================================
# PIPELINE FEATURES BINAIRES
# ============================================================================

def fit_binary_features_pipeline(
    df_train: pd.DataFrame,
    binary_mapping: Dict[str, str]
) -> Dict:
    """
    Apprend les modes pour imputation des indicateurs binaires.
    
    Transforme les variables de consommation en indicateurs binaires :
    - HasElectricity : 1 si Electricity(kWh) > 0, sinon 0
    - HasNaturalGas : 1 si NaturalGas(therms) > 0, sinon 0
    - HasSteam : 1 si SteamUse(kBtu) > 0, sinon 0
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    binary_mapping : dict
        Mapping {nom_feature: nom_colonne_source}
        Exemple: {'HasElectricity': 'Electricity(kWh)'}
    
    Retourne
    --------
    params : dict
        Dictionnaire contenant :
        - binary_features : mapping des features
        - modes : dict {feature_name: mode_value}
    
    Exemple
    -------
    >>> mapping = {'HasElectricity': 'Electricity(kWh)'}
    >>> params = fit_binary_features_pipeline(train_df, mapping)
    >>> print(params['modes'])
    {'HasElectricity': 1}
    """
    params = {
        'binary_features': binary_mapping,
        'modes': {}
    }
    
    # Créer les features temporairement pour calculer le mode
    for feature_name, source_col in binary_mapping.items():
        if source_col in df_train.columns:
            temp_feature = (df_train[source_col] > 0).astype(float)
            temp_feature[df_train[source_col].isnull()] = np.nan
            
            mode_val = temp_feature.mode()
            params['modes'][feature_name] = int(mode_val[0]) if not mode_val.empty else 0
    
    return params


def transform_binary_features_pipeline(
    df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Crée les indicateurs binaires et impute avec les modes du train.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset à traiter (train ou test)
    params : dict
        Paramètres appris avec fit_binary_features_pipeline()
    
    Retourne
    --------
    binary_df : pd.DataFrame
        DataFrame contenant uniquement les features binaires
    
    Exemple
    -------
    >>> params = fit_binary_features_pipeline(train_df, mapping)
    >>> train_binary = transform_binary_features_pipeline(train_df, params)
    >>> test_binary = transform_binary_features_pipeline(test_df, params)
    """
    binary_df = pd.DataFrame(index=df.index)
    
    for feature_name, source_col in params['binary_features'].items():
        if source_col in df.columns:
            # Créer la feature binaire
            binary_df[feature_name] = (df[source_col] > 0).astype(float)
            binary_df.loc[df[source_col].isnull(), feature_name] = np.nan
            
            # Imputer avec le mode du train
            mode_val = params['modes'][feature_name]
            binary_df[feature_name].fillna(mode_val, inplace=True)
            binary_df[feature_name] = binary_df[feature_name].astype(int)
    
    return binary_df


# ============================================================================
# ENCODAGE VARIABLES CATÉGORIELLES
# ============================================================================

def fit_categorical_encoder(
    df_train: pd.DataFrame,
    categorical_cols: List[str],
    encoding_method: str = 'target',
    target_col: Optional[str] = None,
    high_cardinality_threshold: int = 10
) -> Dict:
    """
    Apprend l'encodage des variables catégorielles sur le train set.
    
    Deux méthodes disponibles :
    1. OneHot : Pour variables à faible cardinalité (<= threshold)
    2. Target : Pour variables à haute cardinalité (> threshold)
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    categorical_cols : list
        Liste des colonnes catégorielles à encoder
    encoding_method : str, optional
        Méthode d'encodage ('onehot' ou 'target', défaut: 'target')
    target_col : str, optional
        Nom de la colonne cible (requis pour target encoding)
    high_cardinality_threshold : int, optional
        Seuil pour haute cardinalité (défaut: 10)
    
    Retourne
    --------
    params : dict
        Dictionnaire contenant :
        - encoding_method : méthode utilisée
        - low_cardinality_cols : colonnes pour OneHot
        - high_cardinality_cols : colonnes pour Target
        - onehot_categories : catégories pour OneHot
        - target_mappings : mappings pour Target encoding
    
    Exemple
    -------
    >>> params = fit_categorical_encoder(
    ...     train_df, 
    ...     ['PrimaryPropertyType', 'Neighborhood'],
    ...     encoding_method='target',
    ...     target_col='TotalGHGEmissions_log'
    ... )
    """
    params = {
        'encoding_method': encoding_method,
        'low_cardinality_cols': [],
        'high_cardinality_cols': [],
        'onehot_categories': {},
        'target_mappings': {}
    }
    
    # Séparer colonnes selon cardinalité
    for col in categorical_cols:
        if col not in df_train.columns:
            continue
            
        n_unique = df_train[col].nunique()
        
        if n_unique <= high_cardinality_threshold:
            # Faible cardinalité → OneHot
            params['low_cardinality_cols'].append(col)
            params['onehot_categories'][col] = df_train[col].unique().tolist()
        else:
            # Haute cardinalité → Target encoding
            params['high_cardinality_cols'].append(col)
            
            if target_col and encoding_method == 'target':
                # Calculer la moyenne de la cible par catégorie
                target_means = df_train.groupby(col)[target_col].mean()
                params['target_mappings'][col] = target_means.to_dict()
    
    return params


def transform_categorical_encoder(
    df: pd.DataFrame,
    params: Dict,
    global_mean: Optional[float] = None
) -> pd.DataFrame:
    """
    Applique l'encodage des variables catégorielles.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset à traiter (train ou test)
    params : dict
        Paramètres appris avec fit_categorical_encoder()
    global_mean : float, optional
        Moyenne globale pour les catégories inconnues (target encoding)
    
    Retourne
    --------
    df_encoded : pd.DataFrame
        Dataset avec variables catégorielles encodées
    
    Exemple
    -------
    >>> params = fit_categorical_encoder(train_df, cols, target_col='y')
    >>> train_encoded = transform_categorical_encoder(train_df, params)
    >>> test_encoded = transform_categorical_encoder(test_df, params, global_mean=5.0)
    """
    df_encoded = df.copy()
    
    # OneHot encoding pour faible cardinalité
    for col in params['low_cardinality_cols']:
        if col not in df_encoded.columns:
            continue
            
        # Créer colonnes dummy
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
        
        # Ajouter colonnes manquantes (catégories du train non présentes dans test)
        expected_cols = [f"{col}_{cat}" for cat in params['onehot_categories'][col][1:]]
        for expected_col in expected_cols:
            if expected_col not in dummies.columns:
                dummies[expected_col] = 0
        
        # Supprimer colonnes en trop (catégories du test non présentes dans train)
        dummies = dummies[[col_name for col_name in dummies.columns if col_name in expected_cols]]
        
        # Remplacer la colonne originale
        df_encoded = df_encoded.drop(columns=[col])
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    # Target encoding pour haute cardinalité
    for col in params['high_cardinality_cols']:
        if col not in df_encoded.columns:
            continue
            
        mapping = params['target_mappings'].get(col, {})
        
        # Créer nouvelle colonne encodée
        encoded_col_name = f"{col}_encoded"
        df_encoded[encoded_col_name] = df_encoded[col].map(mapping)
        
        # Pour les catégories inconnues, utiliser la moyenne globale
        if global_mean is not None:
            df_encoded[encoded_col_name].fillna(global_mean, inplace=True)
        
        # Supprimer la colonne originale
        df_encoded = df_encoded.drop(columns=[col])
    
    return df_encoded


# ============================================================================
# FONCTION UTILITAIRE : PIPELINE COMPLET
# ============================================================================

def fit_complete_pipeline(
    df_train: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    binary_mapping: Dict[str, str]
) -> Dict:
    """
    Apprend tous les pipelines de traitement sur le train set.
    
    Pipeline complet :
    1. Valeurs manquantes
    2. Valeurs extrêmes
    3. Features binaires
    4. Encodage catégorielles
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    target_col : str
        Nom de la colonne cible
    categorical_cols : list
        Liste des colonnes catégorielles
    binary_mapping : dict
        Mapping pour features binaires
    
    Retourne
    --------
    all_params : dict
        Dictionnaire contenant tous les paramètres de tous les pipelines
    
    Exemple
    -------
    >>> all_params = fit_complete_pipeline(
    ...     train_df,
    ...     target_col='TotalGHGEmissions_log',
    ...     categorical_cols=['PrimaryPropertyType'],
    ...     binary_mapping={'HasElectricity': 'Electricity(kWh)'}
    ... )
    """
    all_params = {}
    
    # Pipeline 1 : Valeurs manquantes
    all_params['missing_values'] = fit_missing_values_pipeline(df_train)
    
    # Pipeline 2 : Valeurs extrêmes
    all_params['outliers'] = fit_outliers_pipeline(df_train)
    
    # Pipeline 3 : Features binaires
    all_params['binary_features'] = fit_binary_features_pipeline(df_train, binary_mapping)
    
    # Pipeline 4 : Encodage catégorielles
    all_params['categorical'] = fit_categorical_encoder(
        df_train,
        categorical_cols,
        encoding_method='target',
        target_col=target_col
    )
    
    # Moyenne globale pour target encoding
    all_params['global_mean'] = df_train[target_col].mean()
    
    return all_params


def transform_complete_pipeline(
    df: pd.DataFrame,
    all_params: Dict,
    binary_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Applique tous les pipelines de traitement sur un dataset.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset à traiter (train ou test)
    all_params : dict
        Tous les paramètres appris avec fit_complete_pipeline()
    binary_mapping : dict
        Mapping pour features binaires
    
    Retourne
    --------
    df_processed : pd.DataFrame
        Dataset complètement traité
    
    Exemple
    -------
    >>> all_params = fit_complete_pipeline(train_df, ...)
    >>> train_clean = transform_complete_pipeline(train_df, all_params, mapping)
    >>> test_clean = transform_complete_pipeline(test_df, all_params, mapping)
    """
    df_processed = df.copy()
    
    # Étape 1 : Valeurs manquantes
    df_processed = transform_missing_values_pipeline(df_processed, all_params['missing_values'])
    
    # Étape 2 : Valeurs extrêmes
    df_processed = transform_outliers_pipeline(df_processed, all_params['outliers'])
    
    # Étape 3 : Features binaires (à ajouter séparément)
    binary_df = transform_binary_features_pipeline(df, all_params['binary_features'])
    
    # Étape 4 : Encodage catégorielles
    df_processed = transform_categorical_encoder(
        df_processed,
        all_params['categorical'],
        global_mean=all_params.get('global_mean')
    )
    
    # Ajouter les features binaires
    df_processed = pd.concat([df_processed, binary_df], axis=1)
    
    return df_processed


# ============================================================================
# FONCTION UTILITAIRE : AFFICHAGE
# ============================================================================

def print_pipeline_summary(params: Dict) -> None:
    """
    Affiche un résumé des paramètres du pipeline.
    
    Paramètres
    ----------
    params : dict
        Paramètres du pipeline (retour de fit_XXX_pipeline)
    
    Exemple
    -------
    >>> params = fit_missing_values_pipeline(train_df)
    >>> print_pipeline_summary(params)
    """
    print("=" * 80)
    print("RÉSUMÉ DES PARAMÈTRES DU PIPELINE")
    print("=" * 80)
    
    if 'cols_to_drop_missing' in params:
        print(f"\nColonnes supprimées (>70% missing) : {len(params['cols_to_drop_missing'])}")
        if params['cols_to_drop_missing']:
            for col in params['cols_to_drop_missing'][:5]:
                print(f"  - {col}")
            if len(params['cols_to_drop_missing']) > 5:
                print(f"  ... et {len(params['cols_to_drop_missing']) - 5} autres")
    
    if 'impute_mean' in params:
        print(f"\nColonnes imputées par moyenne : {len(params['impute_mean'])}")
        if params['impute_mean']:
            for col, val in list(params['impute_mean'].items())[:5]:
                print(f"  - {col} : {val:.2f}")
    
    if 'impute_mode' in params:
        print(f"\nColonnes imputées par mode : {len(params['impute_mode'])}")
        if params['impute_mode']:
            for col, val in list(params['impute_mode'].items())[:5]:
                print(f"  - {col} : {val}")
    
    if 'limits' in params:
        print(f"\nColonnes avec limites (outliers) : {len(params['limits'])}")
    
    if 'modes' in params:
        print(f"\nFeatures binaires : {len(params['modes'])}")
        for feature, mode_val in params['modes'].items():
            print(f"  - {feature} : mode = {mode_val}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    """
    Exemple d'utilisation du module
    """
    print(__doc__)
    print("\nCe module est conçu pour être importé dans les notebooks.")
    print("\nExemple d'utilisation :")
    print("""
    from src.preprocessing import (
        fit_missing_values_pipeline,
        transform_missing_values_pipeline
    )
    
    # Sur le train set
    params = fit_missing_values_pipeline(train_df)
    train_clean = transform_missing_values_pipeline(train_df, params)
    
    # Sur le test set (mêmes paramètres)
    test_clean = transform_missing_values_pipeline(test_df, params)
    """)

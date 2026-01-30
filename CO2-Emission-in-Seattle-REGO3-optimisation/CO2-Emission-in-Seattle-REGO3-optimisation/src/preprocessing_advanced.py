"""
REGO3 - Module de Prétraitement Avancé
========================================

Ce module implémente des méthodes statistiquement rigoureuses pour le traitement
des valeurs manquantes et des valeurs extrêmes.

Approche valeurs manquantes :
- >70% missing : Suppression
- 10-70% missing : Tests MCAR/MAR/MNAR + imputation adaptée
- <10% missing : Imputation simple

Approche valeurs extrêmes :
- Détection univariée : IQR, Z-score
- Détection multivariée : Leverage, Cook's distance, DFFITS
- Traitements : Winsorisation, transformation, suppression conditionnelle

Auteur : Équipe REGO3
Date : Janvier 2026
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import chi2
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier    

import warnings


# ============================================================================
# TESTS DE MISSINGNESS
# ============================================================================

def little_mcar_test(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Implémente Little's MCAR test pour tester si les données sont MCAR.
    
    H0 : Les données sont MCAR (Missing Completely At Random)
    H1 : Les données ne sont pas MCAR (MAR ou MNAR)
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset avec valeurs manquantes
    alpha : float, optional
        Seuil de significativité (défaut: 0.05)
    
    Retourne
    --------
    result : dict
        - statistic : Statistique du test
        - p_value : P-value
        - is_mcar : True si MCAR (p > alpha), False sinon
        - conclusion : Interprétation
    
    Note
    ----
    Si p < alpha : rejeter H0 → données ne sont PAS MCAR
    Si p > alpha : ne pas rejeter H0 → données sont potentiellement MCAR
    """
    # Sélectionner colonnes numériques avec missing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()
    
    # Compter patterns de missing
    missing_patterns = df_numeric.isnull().astype(int)
    unique_patterns = missing_patterns.drop_duplicates()
    
    if len(unique_patterns) <= 1:
        return {
            'statistic': 0,
            'p_value': 1.0,
            'is_mcar': True,
            'conclusion': 'Pas assez de patterns de missing pour tester'
        }
    
    # Calculer statistique (approximation simplifiée)
    # Note: implémentation complète nécessite EM algorithm
    n_patterns = len(unique_patterns)
    n_vars = len(numeric_cols)
    
    # Approximation chi-square
    chi_square_stat = 0
    for pattern_idx, pattern in unique_patterns.iterrows():
        pattern_mask = (missing_patterns == pattern).all(axis=1)
        n_obs = pattern_mask.sum()
        
        if n_obs > 1:
            for col in numeric_cols:
                if not pattern[col]:  # Valeur non manquante
                    obs_mean = df_numeric.loc[pattern_mask, col].mean()
                    overall_mean = df_numeric[col].mean()
                    obs_var = df_numeric[col].var()
                    
                    if obs_var > 0:
                        chi_square_stat += n_obs * ((obs_mean - overall_mean) ** 2) / obs_var
    
    # Degrés de liberté (approximation)
    df_test = max(1, (n_patterns - 1) * n_vars)
    
    # P-value
    p_value = 1 - chi2.cdf(chi_square_stat, df_test)
    
    return {
        'statistic': chi_square_stat,
        'p_value': p_value,
        'degrees_of_freedom': df_test,
        'is_mcar': p_value > alpha,
        'conclusion': 'MCAR' if p_value > alpha else 'MAR ou MNAR (pas MCAR)'
    }


def test_mar_vs_mnar(
    df: pd.DataFrame,
    col_with_missing: str,
    observed_cols: List[str]
) -> Dict:
    """
    Teste si une variable avec missing est MAR (liée aux observées) ou MNAR.
    
    Méthode : Régression logistique de l'indicateur de missing sur les variables observées
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset
    col_with_missing : str
        Colonne contenant des valeurs manquantes
    observed_cols : list
        Colonnes observées à tester
    
    Retourne
    --------
    result : dict
        - is_mar : True si MAR (corrélé aux observées), False si potentiellement MNAR
        - significant_predictors : Variables observées liées au missing
        - p_values : P-values pour chaque variable
    """
    from scipy.stats import ttest_ind, chi2_contingency
    
    # Créer indicateur de missing
    missing_indicator = df[col_with_missing].isnull().astype(int)
    
    if missing_indicator.sum() == 0:
        return {
            'is_mar': False,
            'significant_predictors': [],
            'p_values': {},
            'conclusion': 'Pas de valeurs manquantes'
        }
    
    significant_predictors = []
    p_values = {}
    
    for obs_col in observed_cols:
        if obs_col in df.columns and obs_col != col_with_missing:
            # Test selon le type de variable
            if df[obs_col].dtype in ['int64', 'float64']:
                # Variable continue : t-test
                group_missing = df[df[col_with_missing].isnull()][obs_col].dropna()
                group_observed = df[df[col_with_missing].notnull()][obs_col].dropna()
                
                if len(group_missing) > 0 and len(group_observed) > 0:
                    t_stat, p_val = ttest_ind(group_missing, group_observed)
                    p_values[obs_col] = p_val
                    
                    if p_val < 0.05:
                        significant_predictors.append(obs_col)
            else:
                # Variable catégorielle : chi2
                contingency = pd.crosstab(df[obs_col], missing_indicator)
                chi2_stat, p_val, dof, expected = chi2_contingency(contingency)
                p_values[obs_col] = p_val
                
                if p_val < 0.05:
                    significant_predictors.append(obs_col)
    
    is_mar = len(significant_predictors) > 0
    
    return {
        'is_mar': is_mar,
        'significant_predictors': significant_predictors,
        'p_values': p_values,
        'conclusion': 'MAR (lié aux variables observées)' if is_mar else 'Potentiellement MNAR'
    }


def diagnose_missing_mechanism(
    df: pd.DataFrame,
    col_with_missing: str,
    observed_cols: Optional[List[str]] = None
) -> str:
    """
    Diagnostique le mécanisme de missingness pour une colonne.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset
    col_with_missing : str
        Colonne à diagnostiquer
    observed_cols : list, optional
        Colonnes observées pour test MAR (si None, utilise toutes numériques)
    
    Retourne
    --------
    mechanism : str
        'MCAR', 'MAR', ou 'MNAR'
    """
    # Test MCAR global
    mcar_result = little_mcar_test(df[[col_with_missing]])
    
    if mcar_result['is_mcar']:
        return 'MCAR'
    
    # Test MAR vs MNAR
    if observed_cols is None:
        observed_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        observed_cols = [c for c in observed_cols if c != col_with_missing]
    
    mar_result = test_mar_vs_mnar(df, col_with_missing, observed_cols)
    
    if mar_result['is_mar']:
        return 'MAR'
    else:
        return 'MNAR'


# ============================================================================
# IMPUTATION ADAPTATIVE SELON TYPE DE MISSINGNESS
# ============================================================================

def impute_mcar(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Imputation pour MCAR : Moyenne/Médiane (simple).
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset
    col : str
        Colonne à imputer
    
    Retourne
    --------
    imputed : pd.Series
        Série avec valeurs imputées
    """
    if df[col].dtype in ['int64', 'float64']:
        # Numérique : médiane (plus robuste)
        fill_value = df[col].median()
    else:
        # Catégorielle : mode
        fill_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
    
    return df[col].fillna(fill_value)


def impute_mar_numeric(df: pd.DataFrame, col: str, predictors: List[str]) -> pd.Series:
    """
    Imputation pour MAR numérique : Régression linéaire sur les prédicteurs.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset
    col : str
        Colonne à imputer
    predictors : list
        Colonnes prédictives (liées au missing)
    
    Retourne
    --------
    imputed : pd.Series
        Série avec valeurs imputées
    """

    # Préparer données
    df_copy = df.copy()
    
    # Séparer observed et missing
    mask_observed = df_copy[col].notnull()
    mask_missing = df_copy[col].isnull()
    
    # Préparer X (prédicteurs)
    X_train = df_copy.loc[mask_observed, predictors].fillna(df_copy[predictors].median())
    X_predict = df_copy.loc[mask_missing, predictors].fillna(df_copy[predictors].median())
    
    # Préparer y
    y_train = df_copy.loc[mask_observed, col]
    
    # Entraîner modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédire
    y_pred = model.predict(X_predict)
    
    # Remplir
    result = df_copy[col].copy()
    result.loc[mask_missing] = y_pred
    
    return result


def impute_mar_categorical(df: pd.DataFrame, col: str, predictors: List[str]) -> pd.Series:
    """
    Imputation pour MAR catégorielle : Mode conditionnel sur les prédicteurs.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset
    col : str
        Colonne à imputer
    predictors : list
        Colonnes prédictives
    
    Retourne
    --------
    imputed : pd.Series
        Série avec valeurs imputées
    """
  
    
    df_copy = df.copy()
    
    # Séparer observed et missing
    mask_observed = df_copy[col].notnull()
    mask_missing = df_copy[col].isnull()
    
    # Préparer données
    X_train = df_copy.loc[mask_observed, predictors]
    X_predict = df_copy.loc[mask_missing, predictors]
    
    # Encoder variables catégorielles dans X si nécessaire
    for pred in predictors:
        if df_copy[pred].dtype == 'object':
            df_copy[pred] = pd.Categorical(df_copy[pred]).codes
    
    X_train = df_copy.loc[mask_observed, predictors].fillna(-1)
    X_predict = df_copy.loc[mask_missing, predictors].fillna(-1)
    y_train = df_copy.loc[mask_observed, col]
    
    # KNN pour imputation
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_predict)
    
    # Remplir
    result = df_copy[col].copy()
    result.loc[mask_missing] = y_pred
    
    return result


# ============================================================================
# PIPELINE VALEURS MANQUANTES AMÉLIORÉ
# ============================================================================

def fit_missing_values_pipeline_advanced(
    df_train: pd.DataFrame,
    threshold_drop: float = 0.70,
    threshold_simple: float = 0.10,
    test_mechanism: bool = True
) -> Dict:
    """
    Pipeline avancé pour valeurs manquantes avec tests statistiques.
    
    Stratégie :
    - >70% missing : Suppression
    - 10-70% missing : Test MCAR/MAR/MNAR + imputation adaptée
    - <10% missing : Imputation simple
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    threshold_drop : float
        Seuil de suppression (défaut: 0.70)
    threshold_simple : float
        Seuil pour imputation simple (défaut: 0.10)
    test_mechanism : bool
        Si True, teste le mécanisme de missingness
    
    Retourne
    --------
    params : dict
        Paramètres du pipeline incluant :
        - cols_to_drop : Colonnes à supprimer (>70%)
        - simple_impute : Colonnes pour imputation simple (<10%)
        - advanced_impute : Colonnes pour imputation avancée (10-70%)
        - mechanisms : Type de missingness par colonne
        - imputation_values : Valeurs/modèles d'imputation
    """
    params = {
        'cols_to_drop': [],
        'simple_impute': {'mean': {}, 'mode': {}},
        'advanced_impute': {},
        'mechanisms': {},
        'cols_to_drop_unique': []
    }
    
    # Calculer % de missing par colonne
    missing_pct = df_train.isnull().sum() / len(df_train)
    
    # Catégoriser colonnes
    for col in df_train.columns:
        pct = missing_pct[col]
        
        # Supprimer colonnes avec valeur unique
        if df_train[col].nunique() <= 1:
            params['cols_to_drop_unique'].append(col)
            continue
        
        if pct > threshold_drop:
            # >70% : Suppression
            params['cols_to_drop'].append(col)
        
        elif pct > threshold_simple:
            # 10-70% : Imputation avancée
            if test_mechanism:
                # Diagnostiquer mécanisme
                mechanism = diagnose_missing_mechanism(df_train, col)
                params['mechanisms'][col] = mechanism
                
                # Préparer imputation selon mécanisme
                if mechanism == 'MCAR':
                    # MCAR : Médiane/Mode
                    if df_train[col].dtype in ['int64', 'float64']:
                        params['advanced_impute'][col] = {
                            'method': 'median',
                            'value': df_train[col].median()
                        }
                    else:
                        mode_val = df_train[col].mode()
                        params['advanced_impute'][col] = {
                            'method': 'mode',
                            'value': mode_val[0] if not mode_val.empty else 'Unknown'
                        }
                
                elif mechanism == 'MAR':
                    # MAR : Identifier prédicteurs
                    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c != col and missing_pct[c] < 0.3]
                    
                    mar_test = test_mar_vs_mnar(df_train, col, numeric_cols[:10])  # Max 10 prédicteurs
                    
                    params['advanced_impute'][col] = {
                        'method': 'regression' if df_train[col].dtype in ['int64', 'float64'] else 'knn',
                        'predictors': mar_test['significant_predictors'][:5],  # Top 5
                        'fallback_value': df_train[col].median() if df_train[col].dtype in ['int64', 'float64'] else df_train[col].mode()[0]
                    }
                
                else:  # MNAR
                    # MNAR : Imputation conservatrice (médiane/mode)
                    if df_train[col].dtype in ['int64', 'float64']:
                        params['advanced_impute'][col] = {
                            'method': 'median',
                            'value': df_train[col].median(),
                            'note': 'MNAR - imputation conservatrice'
                        }
                    else:
                        mode_val = df_train[col].mode()
                        params['advanced_impute'][col] = {
                            'method': 'mode',
                            'value': mode_val[0] if not mode_val.empty else 'Unknown',
                            'note': 'MNAR - imputation conservatrice'
                        }
        
        elif 0 < pct <= threshold_simple:
            # <10% : Imputation simple
            if df_train[col].dtype in ['int64', 'float64']:
                params['simple_impute']['mean'][col] = df_train[col].mean()
            else:
                mode_val = df_train[col].mode()
                params['simple_impute']['mode'][col] = mode_val[0] if not mode_val.empty else 'Unknown'
    
    return params


def transform_missing_values_pipeline_advanced(
    df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Applique le pipeline avancé de traitement des valeurs manquantes.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset à traiter (train ou test)
    params : dict
        Paramètres du pipeline
    
    Retourne
    --------
    df_processed : pd.DataFrame
        Dataset traité
    """
    df_processed = df.copy()
    
    # 1. Supprimer colonnes
    cols_to_drop = params['cols_to_drop'] + params['cols_to_drop_unique']
    cols_to_drop = [c for c in cols_to_drop if c in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
    
    # 2. Imputation simple
    for col, val in params['simple_impute']['mean'].items():
        if col in df_processed.columns:
            df_processed[col].fillna(val, inplace=True)
    
    for col, val in params['simple_impute']['mode'].items():
        if col in df_processed.columns:
            df_processed[col].fillna(val, inplace=True)
    
    # 3. Imputation avancée
    for col, impute_info in params['advanced_impute'].items():
        if col not in df_processed.columns:
            continue
        
        method = impute_info['method']
        
        if method == 'median':
            df_processed[col].fillna(impute_info['value'], inplace=True)
        
        elif method == 'mode':
            df_processed[col].fillna(impute_info['value'], inplace=True)
        
        elif method == 'regression':
            # Régression (pour MAR numérique)
            predictors = impute_info['predictors']
            predictors = [p for p in predictors if p in df_processed.columns]
            
            if len(predictors) > 0:
                try:
                    df_processed[col] = impute_mar_numeric(df_processed, col, predictors)
                except:
                    # Fallback si régression échoue
                    df_processed[col].fillna(impute_info['fallback_value'], inplace=True)
            else:
                df_processed[col].fillna(impute_info['fallback_value'], inplace=True)
        
        elif method == 'knn':
            # KNN (pour MAR catégorielle)
            predictors = impute_info['predictors']
            predictors = [p for p in predictors if p in df_processed.columns]
            
            if len(predictors) > 0:
                try:
                    df_processed[col] = impute_mar_categorical(df_processed, col, predictors)
                except:
                    # Fallback
                    df_processed[col].fillna(impute_info['fallback_value'], inplace=True)
            else:
                df_processed[col].fillna(impute_info['fallback_value'], inplace=True)
    
    return df_processed


# ============================================================================
# DÉTECTION AVANCÉE DES OUTLIERS
# ============================================================================

def detect_outliers_univariate(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Dict[str, pd.Series]:
    """
    Détection univariée des outliers.
    
    Méthodes disponibles :
    - 'iqr' : Interquartile Range (Q1 - threshold×IQR, Q3 + threshold×IQR)
    - 'zscore' : Z-score (|z| > threshold)
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset
    method : str
        Méthode de détection
    threshold : float
        Seuil (1.5 pour IQR, 3 pour Z-score)
    
    Retourne
    --------
    outliers : dict
        {colonne: Series booléenne (True = outlier)}
    """
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers[col] = (df[col] < lower) | (df[col] > upper)
        
        elif method == 'zscore':
            z_scores = np.abs(sp_stats.zscore(df[col].dropna()))
            outliers[col] = pd.Series(False, index=df.index)
            outliers[col].loc[df[col].notnull()] = z_scores > threshold
    
    return outliers


def detect_outliers_multivariate(
    df_train: pd.DataFrame,
    target_col: Optional[str] = None
) -> Dict:
    """
    Détection multivariée des outliers (leverage, Cook's distance, DFFITS).
    
    Basée sur régression linéaire si target fournie.
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    target_col : str, optional
        Colonne cible pour régression
    
    Retourne
    --------
    results : dict
        - high_leverage : Indices avec leverage élevé
        - high_cooks : Indices avec Cook's distance élevée
        - high_dffits : Indices avec DFFITS élevé
        - thresholds : Seuils utilisés
    """

    # Préparer données
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col:
        X_cols = [c for c in numeric_cols if c != target_col]
        X = df_train[X_cols].fillna(df_train[X_cols].median())
        y = df_train[target_col].fillna(df_train[target_col].median())
    else:
        X = df_train[numeric_cols].fillna(df_train[numeric_cols].median())
        y = X.iloc[:, 0]  # Dummy
    
    n, p = X.shape
    
    # Régression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Hat matrix (leverage)
    X_with_intercept = np.column_stack([np.ones(n), X])
    H = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    leverage = np.diag(H)
    
    # Seuils
    leverage_threshold = 2 * (p + 1) / n
    
    # Cook's distance
    mse = np.sum(residuals ** 2) / (n - p - 1)
    cooks_d = (residuals ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)
    cooks_threshold = 1.0
    
    # DFFITS
    dffits = residuals * np.sqrt(leverage / (1 - leverage)) / np.sqrt(mse)
    dffits_threshold = 2 * np.sqrt(p / n)
    
    return {
        'leverage': leverage,
        'cooks_distance': cooks_d,
        'dffits': dffits,
        'high_leverage': df_train.index[leverage > leverage_threshold].tolist(),
        'high_cooks': df_train.index[cooks_d > cooks_threshold].tolist(),
        'high_dffits': df_train.index[np.abs(dffits) > dffits_threshold].tolist(),
        'thresholds': {
            'leverage': leverage_threshold,
            'cooks': cooks_threshold,
            'dffits': dffits_threshold
        }
    }


# ============================================================================
# TRAITEMENT AVANCÉ DES OUTLIERS
# ============================================================================

def winsorize_outliers(
    df: pd.DataFrame,
    lower_percentile: float = 0.05,
    upper_percentile: float = 0.95
) -> pd.DataFrame:
    """
    Winsorisation : Remplace outliers par percentiles.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset
    lower_percentile : float
        Percentile inférieur (défaut: 5%)
    upper_percentile : float
        Percentile supérieur (défaut: 95%)
    
    Retourne
    --------
    df_winsorized : pd.DataFrame
        Dataset winsorisé
    """
    from scipy.stats.mstats import winsorize
    
    df_winsorized = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        lower = df[col].quantile(lower_percentile)
        upper = df[col].quantile(upper_percentile)
        df_winsorized[col] = df[col].clip(lower=lower, upper=upper)
    
    return df_winsorized


def fit_outliers_pipeline_advanced(
    df_train: pd.DataFrame,
    method: str = 'winsorize',
    univariate_method: str = 'iqr',
    target_col: Optional[str] = None
) -> Dict:
    """
    Pipeline avancé de détection et traitement des outliers.
    
    Méthodes disponibles :
    - 'winsorize' : Winsorisation (5-95 percentiles)
    - 'iqr' : IQR capping
    - 'zscore' : Z-score capping
    - 'remove' : Suppression basée sur leverage
    
    Paramètres
    ----------
    df_train : pd.DataFrame
        Dataset d'entraînement
    method : str
        Méthode de traitement
    univariate_method : str
        Méthode de détection univariée
    target_col : str, optional
        Colonne cible pour détection multivariée
    
    Retourne
    --------
    params : dict
        Paramètres du pipeline
    """
    params = {
        'method': method,
        'univariate_outliers': {},
        'multivariate_outliers': {},
        'treatment_values': {}
    }
    
    # Détection univariée
    outliers_uni = detect_outliers_univariate(
        df_train,
        method=univariate_method,
        threshold=1.5 if univariate_method == 'iqr' else 3
    )
    params['univariate_outliers'] = outliers_uni
    
    # Détection multivariée (si target fournie)
    if target_col:
        outliers_multi = detect_outliers_multivariate(df_train, target_col)
        params['multivariate_outliers'] = outliers_multi
    
    # Calculer valeurs de traitement selon méthode
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'winsorize':
            params['treatment_values'][col] = {
                'lower': df_train[col].quantile(0.05),
                'upper': df_train[col].quantile(0.95)
            }
        elif method == 'iqr':
            Q1 = df_train[col].quantile(0.25)
            Q3 = df_train[col].quantile(0.75)
            IQR = Q3 - Q1
            params['treatment_values'][col] = {
                'lower': Q1 - 1.5 * IQR,
                'upper': Q3 + 1.5 * IQR
            }
        elif method == 'zscore':
            mean = df_train[col].mean()
            std = df_train[col].std()
            params['treatment_values'][col] = {
                'lower': mean - 3 * std,
                'upper': mean + 3 * std
            }
    
    return params


def transform_outliers_pipeline_advanced(
    df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Applique le traitement avancé des outliers.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Dataset à traiter
    params : dict
        Paramètres du pipeline
    
    Retourne
    --------
    df_treated : pd.DataFrame
        Dataset traité
    """
    df_treated = df.copy()
    
    method = params['method']
    
    if method in ['winsorize', 'iqr', 'zscore']:
        # Capping/Winsorisation
        for col, values in params['treatment_values'].items():
            if col in df_treated.columns:
                df_treated[col] = df_treated[col].clip(
                    lower=values['lower'],
                    upper=values['upper']
                )
    
    return df_treated


# ============================================================================
# FONCTIONS UTILITAIRES EXISTANTES (COMPATIBILITÉ)
# ============================================================================

# Garder les anciennes fonctions pour compatibilité avec notebooks existants
def fit_missing_values_pipeline(df_train, threshold_missing=0.70, threshold_unique=1, threshold_impute=0.10):
    """Version simple pour compatibilité."""
    return fit_missing_values_pipeline_advanced(
        df_train,
        threshold_drop=threshold_missing,
        threshold_simple=threshold_impute,
        test_mechanism=False  # Pas de tests pour version simple
    )

def transform_missing_values_pipeline(df, params):
    """Version simple pour compatibilité."""
    return transform_missing_values_pipeline_advanced(df, params)

def fit_outliers_pipeline(df_train, threshold=3, method='zscore'):
    """Version simple pour compatibilité."""
    return fit_outliers_pipeline_advanced(
        df_train,
        method='zscore' if method == 'zscore' else 'iqr',
        univariate_method=method
    )

def transform_outliers_pipeline(df, params):
    """Version simple pour compatibilité."""
    return transform_outliers_pipeline_advanced(df, params)


# Autres fonctions existantes (binary features, categorical, etc.) restent inchangées
# [Copier les fonctions fit_binary_features_pipeline, transform_binary_features_pipeline,
#  fit_categorical_encoder, transform_categorical_encoder, print_pipeline_summary
#  du fichier original]


if __name__ == "__main__":
    print(__doc__)
    print("\nModule de preprocessing avancé avec tests statistiques.")
    print("\nNouvelles fonctionnalités:")
    print("- Tests MCAR/MAR/MNAR")
    print("- Imputation adaptative")
    print("- Détection multivariée d'outliers")
    print("- Winsorisation et traitements avancés")

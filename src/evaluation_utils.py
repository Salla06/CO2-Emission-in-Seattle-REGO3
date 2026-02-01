"""
Module d'Évaluation - Fonctions Utilitaires

Ce module contient les fonctions pour évaluer les performances des modèles ML,
incluant les métriques, la validation croisée et l'analyse des résidus.

Auteur: Équipe Projet Seattle Energy Benchmarking
Date: Janvier 2026
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from typing import Dict, Tuple, Any, Optional


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Évalue un modèle sur train et test avec métriques complètes.
    
    Cette fonction calcule un ensemble complet de métriques sur les datasets
    d'entraînement et de test, incluant les métriques sur échelle log et originale.
    
    Parameters:
    -----------
    model : estimator
        Modèle sklearn entraîné
    X_train, y_train : pd.DataFrame, pd.Series
        Données d'entraînement
    X_test, y_test : pd.DataFrame, pd.Series
        Données de test
    model_name : str, default="Model"
        Nom du modèle pour affichage
    
    Returns:
    --------
    dict : Dictionnaire des métriques contenant:
        - train_r2, test_r2 : R² scores
        - train_rmse_log, test_rmse_log : RMSE échelle log
        - train_mae_log, test_mae_log : MAE échelle log
        - test_rmse_original : RMSE échelle originale
        - test_mae_original : MAE échelle originale
        - test_mape : Mean Absolute Percentage Error
        - overfitting_r2 : Différence R² train-test
        - overfitting_rmse : Différence RMSE test-train
    
    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor()
    >>> model.fit(X_train, y_train)
    >>> metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    >>> print(f"R² Test: {metrics['test_r2']:.4f}")
    """
    
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # ========================================================================
    # MÉTRIQUES ÉCHELLE LOG
    # ========================================================================
    
    # R² Score
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # MAE
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # ========================================================================
    # MÉTRIQUES ÉCHELLE ORIGINALE
    # ========================================================================
    
    # Retransformation (log1p → expm1)
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)
    y_train_pred_orig = np.expm1(y_train_pred)
    y_test_pred_orig = np.expm1(y_test_pred)
    
    # RMSE et MAE échelle originale
    test_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))
    test_mae_orig = mean_absolute_error(y_test_orig, y_test_pred_orig)
    
    # MAPE
    test_mape = mean_absolute_percentage_error(y_test_orig, y_test_pred_orig) * 100
    
    # ========================================================================
    # ANALYSE OVERFITTING
    # ========================================================================
    
    overfitting_r2 = train_r2 - test_r2
    overfitting_rmse = test_rmse - train_rmse
    
    # ========================================================================
    # RETOUR
    # ========================================================================
    
    metrics = {
        'model': model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse_log': train_rmse,
        'test_rmse_log': test_rmse,
        'train_mae_log': train_mae,
        'test_mae_log': test_mae,
        'test_rmse_original': test_rmse_orig,
        'test_mae_original': test_mae_orig,
        'test_mape': test_mape,
        'overfitting_r2': overfitting_r2,
        'overfitting_rmse': overfitting_rmse
    }
    
    return metrics


def cv_evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    model_name: str = "Model",
    n_jobs: int = -1
) -> Dict[str, float]:
    """
    Évalue un modèle avec cross-validation.
    
    Parameters:
    -----------
    model : estimator
        Modèle sklearn
    X, y : pd.DataFrame, pd.Series
        Données
    cv : int, default=5
        Nombre de folds
    model_name : str, default="Model"
        Nom du modèle
    n_jobs : int, default=-1
        Nombre de jobs parallèles
    
    Returns:
    --------
    dict : Métriques moyennes CV contenant:
        - cv_r2_mean, cv_r2_std : R² moyen et écart-type
        - cv_rmse_mean, cv_rmse_std : RMSE moyen et écart-type
        - cv_mae_mean, cv_mae_std : MAE moyen et écart-type
        - train_r2_mean : R² moyen sur train
    
    Examples:
    ---------
    >>> cv_metrics = cv_evaluate_model(model, X_train, y_train, cv=5)
    >>> print(f"R² CV: {cv_metrics['cv_r2_mean']:.4f} ± {cv_metrics['cv_r2_std']:.4f}")
    """
    
    # Définir les scorers
    scoring = {
        'r2': 'r2',
        'neg_rmse': 'neg_root_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    
    # Cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=n_jobs
    )
    
    # Calculer moyennes et écarts-types
    metrics = {
        'model': model_name,
        'cv_r2_mean': cv_results['test_r2'].mean(),
        'cv_r2_std': cv_results['test_r2'].std(),
        'cv_rmse_mean': -cv_results['test_neg_rmse'].mean(),
        'cv_rmse_std': cv_results['test_neg_rmse'].std(),
        'cv_mae_mean': -cv_results['test_neg_mae'].mean(),
        'cv_mae_std': cv_results['test_neg_mae'].std(),
        'train_r2_mean': cv_results['train_r2'].mean(),
    }
    
    return metrics


def compute_residuals_stats(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calcule les statistiques des résidus.
    
    Parameters:
    -----------
    y_true : pd.Series
        Valeurs réelles
    y_pred : np.ndarray
        Prédictions
    
    Returns:
    --------
    dict : Statistiques des résidus
    
    Examples:
    ---------
    >>> residuals_stats = compute_residuals_stats(y_test, y_pred)
    >>> print(f"Moyenne résidus: {residuals_stats['mean']:.4f}")
    """
    
    residuals = y_true - y_pred
    
    # Statistiques de base
    stats_dict = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'min': residuals.min(),
        'max': residuals.max(),
        'median': np.median(residuals),
        'q25': np.percentile(residuals, 25),
        'q75': np.percentile(residuals, 75)
    }
    
    # Test de normalité (Shapiro-Wilk)
    # Limité à 5000 observations pour performances
    sample_size = min(5000, len(residuals))
    if sample_size < len(residuals):
        residuals_sample = np.random.choice(residuals, sample_size, replace=False)
    else:
        residuals_sample = residuals
    
    shapiro_stat, shapiro_pval = stats.shapiro(residuals_sample)
    stats_dict['shapiro_stat'] = shapiro_stat
    stats_dict['shapiro_pval'] = shapiro_pval
    stats_dict['is_normal'] = shapiro_pval > 0.05
    
    return stats_dict


def test_homoscedasticity(
    y_pred: np.ndarray,
    residuals: np.ndarray
) -> Tuple[float, float, bool]:
    """
    Test d'homoscédasticité des résidus (test de Breusch-Pagan simplifié).
    
    Parameters:
    -----------
    y_pred : np.ndarray
        Prédictions
    residuals : np.ndarray
        Résidus
    
    Returns:
    --------
    tuple : (correlation, pvalue, is_homoscedastic)
    
    Examples:
    ---------
    >>> corr, pval, is_homo = test_homoscedasticity(y_pred, residuals)
    >>> print(f"Homoscédasticité: {is_homo}")
    """
    
    # Corrélation entre prédictions et résidus au carré
    residuals_squared = residuals ** 2
    corr, pval = stats.pearsonr(y_pred, residuals_squared)
    
    # Si pas de corrélation significative → homoscédastique
    is_homoscedastic = pval > 0.05
    
    return corr, pval, is_homoscedastic


def compute_prediction_intervals(
    y_pred: np.ndarray,
    residuals_std: float,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les intervalles de prédiction.
    
    Parameters:
    -----------
    y_pred : np.ndarray
        Prédictions
    residuals_std : float
        Écart-type des résidus
    confidence : float, default=0.95
        Niveau de confiance
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    
    Examples:
    ---------
    >>> lower, upper = compute_prediction_intervals(y_pred, residuals_std)
    >>> coverage = np.mean((y_test >= lower) & (y_test <= upper))
    >>> print(f"Couverture: {coverage:.2%}")
    """
    
    # Z-score pour le niveau de confiance
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Intervalles
    margin = z_score * residuals_std
    lower_bound = y_pred - margin
    upper_bound = y_pred + margin
    
    return lower_bound, upper_bound


def evaluate_prediction_quality(
    y_true: pd.Series,
    y_pred: np.ndarray,
    quantiles: list = [0.25, 0.5, 0.75]
) -> pd.DataFrame:
    """
    Évalue la qualité des prédictions par quantile.
    
    Parameters:
    -----------
    y_true : pd.Series
        Valeurs réelles
    y_pred : np.ndarray
        Prédictions
    quantiles : list, default=[0.25, 0.5, 0.75]
        Quantiles à analyser
    
    Returns:
    --------
    pd.DataFrame : Métriques par quantile
    
    Examples:
    ---------
    >>> quality_df = evaluate_prediction_quality(y_test, y_pred)
    >>> print(quality_df)
    """
    
    # Créer DataFrame
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'error': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'pct_error': np.abs((y_true - y_pred) / y_true) * 100
    })
    
    # Définir les quantiles
    df['quantile'] = pd.qcut(df['y_true'], q=len(quantiles)+1, labels=False, duplicates='drop')
    
    # Calculer métriques par quantile
    quality = df.groupby('quantile').agg({
        'y_true': ['min', 'max', 'count'],
        'error': 'mean',
        'abs_error': 'mean',
        'pct_error': 'mean'
    }).round(4)
    
    quality.columns = ['_'.join(col).strip() for col in quality.columns.values]
    quality = quality.reset_index()
    
    return quality


def calculate_metrics_summary(
    metrics_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Crée un tableau récapitulatif des métriques.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionnaire des métriques
    
    Returns:
    --------
    pd.DataFrame : Tableau formaté
    
    Examples:
    ---------
    >>> summary = calculate_metrics_summary(metrics)
    >>> display(summary)
    """
    
    summary_data = {
        'Métrique': [],
        'Valeur': [],
        'Interprétation': []
    }
    
    # R² Score
    if 'test_r2' in metrics_dict:
        r2 = metrics_dict['test_r2']
        summary_data['Métrique'].append('R² Score (Test)')
        summary_data['Valeur'].append(f"{r2:.4f}")
        if r2 >= 0.90:
            interp = "Excellent"
        elif r2 >= 0.80:
            interp = "Très bon"
        elif r2 >= 0.70:
            interp = "Bon"
        elif r2 >= 0.60:
            interp = "Acceptable"
        else:
            interp = "Insuffisant"
        summary_data['Interprétation'].append(interp)
    
    # RMSE
    if 'test_rmse_original' in metrics_dict:
        rmse = metrics_dict['test_rmse_original']
        summary_data['Métrique'].append('RMSE (échelle originale)')
        summary_data['Valeur'].append(f"{rmse:.2f} tonnes CO₂")
        summary_data['Interprétation'].append("Erreur moyenne quadratique")
    
    # MAPE
    if 'test_mape' in metrics_dict:
        mape = metrics_dict['test_mape']
        summary_data['Métrique'].append('MAPE (%)')
        summary_data['Valeur'].append(f"{mape:.2f}%")
        if mape < 10:
            interp = "Excellent"
        elif mape < 15:
            interp = "Bon"
        elif mape < 20:
            interp = "Acceptable"
        else:
            interp = "À améliorer"
        summary_data['Interprétation'].append(interp)
    
    # Overfitting
    if 'overfitting_r2' in metrics_dict:
        overfitting = metrics_dict['overfitting_r2']
        summary_data['Métrique'].append('Overfitting (R²)')
        summary_data['Valeur'].append(f"{overfitting:.4f}")
        if overfitting < 0.05:
            interp = "Pas d'overfitting"
        elif overfitting < 0.10:
            interp = "Overfitting léger"
        else:
            interp = "Overfitting significatif"
        summary_data['Interprétation'].append(interp)
    
    return pd.DataFrame(summary_data)


def compare_model_performance(
    model1_metrics: Dict,
    model2_metrics: Dict,
    model1_name: str = "Modèle 1",
    model2_name: str = "Modèle 2"
) -> pd.DataFrame:
    """
    Compare les performances de deux modèles.
    
    Parameters:
    -----------
    model1_metrics, model2_metrics : dict
        Métriques des modèles
    model1_name, model2_name : str
        Noms des modèles
    
    Returns:
    --------
    pd.DataFrame : Comparaison détaillée
    
    Examples:
    ---------
    >>> comparison = compare_model_performance(metrics_m1, metrics_m2, "Sans ENERGY STAR", "Avec ENERGY STAR")
    >>> display(comparison)
    """
    
    metrics_to_compare = [
        ('test_r2', 'R² Test', 'higher'),
        ('test_rmse_log', 'RMSE Test (log)', 'lower'),
        ('test_mae_log', 'MAE Test (log)', 'lower'),
        ('test_rmse_original', 'RMSE Test (original)', 'lower'),
        ('test_mape', 'MAPE Test (%)', 'lower'),
        ('overfitting_r2', 'Overfitting R²', 'lower')
    ]
    
    comparison_data = []
    
    for metric_key, metric_name, direction in metrics_to_compare:
        if metric_key in model1_metrics and metric_key in model2_metrics:
            val1 = model1_metrics[metric_key]
            val2 = model2_metrics[metric_key]
            diff = val2 - val1
            
            if direction == 'higher':
                winner = model2_name if val2 > val1 else model1_name
                improvement = (val2 - val1) / abs(val1) * 100 if val1 != 0 else 0
            else:
                winner = model2_name if val2 < val1 else model1_name
                improvement = (val1 - val2) / abs(val1) * 100 if val1 != 0 else 0
            
            comparison_data.append({
                'Métrique': metric_name,
                model1_name: f"{val1:.4f}",
                model2_name: f"{val2:.4f}",
                'Différence': f"{diff:+.4f}",
                'Amélioration (%)': f"{improvement:+.2f}%",
                'Meilleur': winner
            })
    
    return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    print("Module evaluation_utils chargé avec succès !")
    print(f"\nFonctions disponibles :")
    print("  - evaluate_model()")
    print("  - cv_evaluate_model()")
    print("  - compute_residuals_stats()")
    print("  - test_homoscedasticity()")
    print("  - compute_prediction_intervals()")
    print("  - evaluate_prediction_quality()")
    print("  - calculate_metrics_summary()")
    print("  - compare_model_performance()")
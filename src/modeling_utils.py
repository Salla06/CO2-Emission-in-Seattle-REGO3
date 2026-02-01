"""
Module de Modélisation - Fonctions Utilitaires

Ce module contient les fonctions utilitaires pour la modélisation ML,
incluant l'entraînement, l'optimisation et la gestion des modèles.

Auteur: Équipe Projet Seattle Energy Benchmarking
Date: Janvier 2026
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import joblib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def get_model_param_grid(model_name: str) -> Dict[str, List]:
    """
    Retourne la grille de paramètres pour un modèle donné.
    
    Parameters:
    -----------
    model_name : str
        Nom du modèle ('Ridge', 'Random Forest', etc.)
    
    Returns:
    --------
    dict : Grille de paramètres
    
    Examples:
    ---------
    >>> grid = get_model_param_grid('Ridge')
    >>> print(grid)
    {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    """
    
    param_grids = {
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        
        'Lasso': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
            'max_iter': [2000, 5000]
        },
        
        'ElasticNet': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [2000, 5000]
        },
        
        'Random Forest': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5]
        },
        
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        },
        
        'XGBoost': {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.3]
        },
        
        'LightGBM': {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10, -1],
            'num_leaves': [31, 50, 70, 100],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        },
        
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'kernel': ['rbf', 'linear']
        }
    }
    
    if model_name not in param_grids:
        raise ValueError(f"Modèle '{model_name}' non supporté. "
                        f"Modèles disponibles: {list(param_grids.keys())}")
    
    return param_grids[model_name]


def optimize_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    n_iter: int = 200,
    cv: int = 5,
    scoring: str = 'r2',
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[Any, Dict, float]:
    """
    Optimise les hyperparamètres d'un modèle via RandomizedSearchCV.
    
    Parameters:
    -----------
    model : estimator
        Modèle sklearn à optimiser
    X_train : pd.DataFrame
        Features d'entraînement
    y_train : pd.Series
        Target d'entraînement
    param_grid : dict, optional
        Grille de paramètres. Si None, utilise get_model_param_grid()
    n_iter : int, default=200
        Nombre d'itérations RandomizedSearchCV
    cv : int, default=5
        Nombre de folds pour cross-validation
    scoring : str, default='r2'
        Métrique d'optimisation
    random_state : int, default=42
        Seed pour reproductibilité
    verbose : int, default=1
        Niveau de verbosité
    
    Returns:
    --------
    tuple : (best_model, best_params, optimization_time)
    
    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor(random_state=42)
    >>> best_model, params, time = optimize_model(model, X_train, y_train)
    >>> print(f"Meilleurs paramètres: {params}")
    """
    
    print(f"🔄 Optimisation des hyperparamètres en cours...")
    print(f"   Méthode : RandomizedSearchCV")
    print(f"   Itérations : {n_iter}")
    print(f"   Cross-validation : {cv}-fold")
    print(f"   Métrique : {scoring}")
    
    start_time = time.time()
    
    # Récupérer la grille si non fournie
    if param_grid is None:
        model_name = model.__class__.__name__
        # Mapper les noms sklearn vers nos clés
        name_mapping = {
            'RandomForestRegressor': 'Random Forest',
            'GradientBoostingRegressor': 'Gradient Boosting',
            'XGBRegressor': 'XGBoost',
            'LGBMRegressor': 'LightGBM',
            'SVR': 'SVR'
        }
        param_grid = get_model_param_grid(name_mapping.get(model_name, model_name))
    
    # Optimisation
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=verbose
    )
    
    random_search.fit(X_train, y_train)
    
    optimization_time = time.time() - start_time
    
    print(f"✓ Optimisation terminée en {optimization_time/60:.2f} minutes")
    print(f"\nMeilleur score CV ({scoring}): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, optimization_time


def train_multiple_models(
    models_dict: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Entraîne et évalue plusieurs modèles.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionnaire {nom: modèle} des modèles à entraîner
    X_train, y_train : pd.DataFrame, pd.Series
        Données d'entraînement
    X_test, y_test : pd.DataFrame, pd.Series
        Données de test
    cv : int, default=5
        Nombre de folds pour cross-validation
    verbose : bool, default=True
        Afficher les logs
    
    Returns:
    --------
    tuple : (results_df, trained_models)
        - results_df : DataFrame des résultats
        - trained_models : Dict des modèles entraînés
    
    Examples:
    ---------
    >>> models = {
    ...     'Ridge': Ridge(),
    ...     'Random Forest': RandomForestRegressor()
    ... }
    >>> results_df, trained = train_multiple_models(models, X_train, y_train, X_test, y_test)
    """
    
    from evaluation_utils import evaluate_model, cv_evaluate_model
    
    results = []
    trained_models = {}
    
    if verbose:
        print("="*80)
        print("ENTRAÎNEMENT MULTIPLE MODÈLES")
        print("="*80)
        print(f"\nNombre de modèles : {len(models_dict)}")
        print(f"Cross-validation : {cv}-fold")
    
    for name, model in models_dict.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Traitement : {name}")
            print(f"{'='*70}")
        
        try:
            start_time = time.time()
            
            # Cross-validation
            if verbose:
                print("   → Cross-validation...")
            cv_metrics = cv_evaluate_model(model, X_train, y_train, cv=cv, model_name=name)
            
            # Entraînement
            if verbose:
                print("   → Entraînement...")
            model.fit(X_train, y_train)
            
            # Évaluation
            if verbose:
                print("   → Évaluation...")
            eval_metrics = evaluate_model(
                model, X_train, y_train, X_test, y_test, model_name=name
            )
            
            train_time = time.time() - start_time
            
            # Combiner métriques
            metrics = {**cv_metrics, **eval_metrics, 'train_time': train_time}
            results.append(metrics)
            trained_models[name] = model
            
            if verbose:
                print(f"   ✓ Terminé en {train_time:.2f}s")
                print(f"   R² Test : {eval_metrics['test_r2']:.4f}")
                print(f"   RMSE Test : {eval_metrics['test_rmse_log']:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"   ✗ Erreur : {str(e)}")
            continue
    
    # Créer DataFrame et trier
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    if verbose:
        print("\n" + "="*80)
        print("✓ ENTRAÎNEMENT TERMINÉ")
        print("="*80)
    
    return results_df, trained_models


def save_model(
    model: Any,
    filepath: Path,
    metadata: Optional[Dict] = None,
    verbose: bool = True
) -> None:
    """
    Sauvegarde un modèle avec ses métadonnées.
    
    Parameters:
    -----------
    model : estimator
        Modèle à sauvegarder
    filepath : Path
        Chemin du fichier de sauvegarde
    metadata : dict, optional
        Métadonnées à sauvegarder
    verbose : bool, default=True
        Afficher confirmation
    
    Examples:
    ---------
    >>> model = RandomForestRegressor()
    >>> model.fit(X_train, y_train)
    >>> metadata = {'r2': 0.85, 'rmse': 0.32}
    >>> save_model(model, Path('models/best_model.pkl'), metadata)
    """
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder modèle
    joblib.dump(model, filepath)
    
    # Sauvegarder métadonnées
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        if verbose:
            print(f"✓ Modèle sauvegardé : {filepath}")
            print(f"✓ Métadonnées sauvegardées : {metadata_path}")
    else:
        if verbose:
            print(f"✓ Modèle sauvegardé : {filepath}")


def load_model(filepath: Path, verbose: bool = True) -> Tuple[Any, Optional[Dict]]:
    """
    Charge un modèle et ses métadonnées.
    
    Parameters:
    -----------
    filepath : Path
        Chemin du fichier modèle
    verbose : bool, default=True
        Afficher confirmation
    
    Returns:
    --------
    tuple : (model, metadata)
    
    Examples:
    ---------
    >>> model, metadata = load_model(Path('models/best_model.pkl'))
    >>> print(f"R² : {metadata['r2']}")
    """
    
    filepath = Path(filepath)
    
    # Charger modèle
    model = joblib.load(filepath)
    
    # Charger métadonnées
    metadata = None
    metadata_path = filepath.with_suffix('.json')
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    if verbose:
        print(f"✓ Modèle chargé : {filepath}")
        if metadata:
            print(f"✓ Métadonnées chargées : {metadata_path}")
    
    return model, metadata


def create_stacking_model(
    base_estimators: List[Tuple[str, Any]],
    final_estimator: Any,
    cv: int = 5,
    n_jobs: int = -1
):
    """
    Crée un modèle de stacking.
    
    Parameters:
    -----------
    base_estimators : list of tuples
        Liste de (nom, modèle) pour les modèles de base
    final_estimator : estimator
        Méta-modèle
    cv : int, default=5
        Cross-validation folds
    n_jobs : int, default=-1
        Nombre de jobs parallèles
    
    Returns:
    --------
    StackingRegressor
    
    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    >>> from sklearn.linear_model import Ridge
    >>> 
    >>> base = [
    ...     ('rf', RandomForestRegressor()),
    ...     ('gb', GradientBoostingRegressor())
    ... ]
    >>> stacking = create_stacking_model(base, Ridge())
    """
    
    from sklearn.ensemble import StackingRegressor
    
    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=n_jobs
    )
    
    return stacking


# ============================================================================
# FONCTIONS D'ANALYSE
# ============================================================================

def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extrait l'importance des features d'un modèle.
    
    Parameters:
    -----------
    model : estimator
        Modèle entraîné
    feature_names : list
        Noms des features
    top_n : int, optional
        Nombre de top features à retourner
    
    Returns:
    --------
    pd.DataFrame : DataFrame avec features et importances
    
    Examples:
    ---------
    >>> importance_df = get_feature_importance(model, X_train.columns, top_n=20)
    >>> print(importance_df.head())
    """
    
    # Extraire importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Modèle ne supporte pas feature importance")
    
    # Créer DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    if top_n:
        importance_df = importance_df.head(top_n)
    
    return importance_df


def compare_models(
    results_df1: pd.DataFrame,
    results_df2: pd.DataFrame,
    model_name_1: str = "Modèle 1",
    model_name_2: str = "Modèle 2"
) -> pd.DataFrame:
    """
    Compare les résultats de deux modèles.
    
    Parameters:
    -----------
    results_df1, results_df2 : pd.DataFrame
        DataFrames de résultats
    model_name_1, model_name_2 : str
        Noms des modèles
    
    Returns:
    --------
    pd.DataFrame : Tableau comparatif
    
    Examples:
    ---------
    >>> comparison = compare_models(results_m1, results_m2, "Sans ENERGY STAR", "Avec ENERGY STAR")
    >>> print(comparison)
    """
    
    # Prendre le meilleur de chaque
    best_1 = results_df1.iloc[0]
    best_2 = results_df2.iloc[0]
    
    comparison = pd.DataFrame({
        'Métrique': [
            'R² Test',
            'RMSE Test (log)',
            'MAE Test (log)',
            'RMSE Test (original)',
            'MAPE Test (%)',
            'Overfitting R²',
            'Temps entraînement (s)'
        ],
        model_name_1: [
            best_1['test_r2'],
            best_1['test_rmse_log'],
            best_1['test_mae_log'],
            best_1['test_rmse_original'],
            best_1['test_mape'],
            best_1['overfitting_r2'],
            best_1['train_time']
        ],
        model_name_2: [
            best_2['test_r2'],
            best_2['test_rmse_log'],
            best_2['test_mae_log'],
            best_2['test_rmse_original'],
            best_2['test_mape'],
            best_2['overfitting_r2'],
            best_2['train_time']
        ]
    })
    
    # Ajouter colonne gain
    comparison['Gain'] = comparison[model_name_2] - comparison[model_name_1]
    
    return comparison


if __name__ == "__main__":
    print("Module modeling_utils chargé avec succès !")
    print(f"\nFonctions disponibles :")
    print("  - get_model_param_grid()")
    print("  - optimize_model()")
    print("  - train_multiple_models()")
    print("  - save_model()")
    print("  - load_model()")
    print("  - create_stacking_model()")
    print("  - get_feature_importance()")
    print("  - compare_models()")
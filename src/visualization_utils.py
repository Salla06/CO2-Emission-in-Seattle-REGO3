"""
Module de Visualisation - Fonctions Utilitaires

Ce module contient les fonctions pour créer des visualisations professionnelles
pour l'analyse de modèles ML.

Auteur: Équipe Projet Seattle Energy Benchmarking
Date: Janvier 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Any
from scipy import stats


# Configuration par défaut
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Prédictions vs Valeurs Réelles",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5),
    show_metrics: bool = True
) -> None:
    """
    Visualise les prédictions vs valeurs réelles avec analyse des résidus.
    
    Parameters:
    -----------
    y_true : pd.Series
        Valeurs réelles
    y_pred : np.ndarray
        Prédictions
    title : str
        Titre du graphique
    save_path : Path, optional
        Chemin de sauvegarde
    figsize : tuple
        Taille de la figure
    show_metrics : bool
        Afficher les métriques sur le graphique
    
    Examples:
    ---------
    >>> plot_predictions(y_test, y_pred, title="Modèle Random Forest",
    ...                  save_path=Path('figures/predictions.png'))
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ========================================================================
    # SCATTER PLOT : Prédictions vs Réalité
    # ========================================================================
    
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    
    # Ligne diagonale parfaite
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 
                 'r--', lw=2, label='Prédiction parfaite')
    
    # Métriques
    if show_metrics:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        axes[0].text(0.05, 0.95, metrics_text,
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
    
    axes[0].set_xlabel('Valeurs Réelles (log)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Prédictions (log)', fontsize=12, fontweight='bold')
    axes[0].set_title('Prédictions vs Réalité', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # ========================================================================
    # RÉSIDUS PLOT
    # ========================================================================
    
    residuals = y_true - y_pred
    
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2, label='Résidu = 0')
    
    # Bandes de confiance (±2 std)
    residuals_std = residuals.std()
    axes[1].axhline(y=2*residuals_std, color='orange', linestyle=':', lw=2, alpha=0.7)
    axes[1].axhline(y=-2*residuals_std, color='orange', linestyle=':', lw=2, alpha=0.7)
    
    axes[1].set_xlabel('Prédictions (log)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Résidus', fontsize=12, fontweight='bold')
    axes[1].set_title('Analyse des Résidus', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée : {save_path}")
    
    plt.show()


def plot_residuals_distribution(
    residuals: np.ndarray,
    title: str = "Distribution des Résidus",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualise la distribution des résidus avec tests statistiques.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Résidus du modèle
    title : str
        Titre du graphique
    save_path : Path, optional
        Chemin de sauvegarde
    figsize : tuple
        Taille de la figure
    
    Examples:
    ---------
    >>> residuals = y_test - y_pred
    >>> plot_residuals_distribution(residuals, save_path=Path('figures/residuals.png'))
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ========================================================================
    # HISTOGRAMME
    # ========================================================================
    
    axes[0, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Moyenne: {residuals.mean():.4f}')
    axes[0, 0].axvline(0, color='green', linestyle='--', linewidth=2, label='Zéro')
    axes[0, 0].set_xlabel('Résidus', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Fréquence', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Histogramme des Résidus', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ========================================================================
    # QQ-PLOT
    # ========================================================================
    
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('QQ-Plot (Normalité)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Test de normalité
    shapiro_stat, shapiro_pval = stats.shapiro(residuals[:5000])  # max 5000
    text = f'Test Shapiro-Wilk:\nStatistique = {shapiro_stat:.4f}\nP-value = {shapiro_pval:.4f}'
    if shapiro_pval > 0.05:
        text += '\n✓ Distribution normale'
    else:
        text += '\n✗ Distribution non-normale'
    axes[0, 1].text(0.05, 0.95, text,
                    transform=axes[0, 1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)
    
    # ========================================================================
    # BOXPLOT
    # ========================================================================
    
    axes[1, 0].boxplot(residuals, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 0].axhline(0, color='green', linestyle='--', linewidth=2, label='Zéro')
    axes[1, 0].set_ylabel('Résidus', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Boxplot des Résidus', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # STATISTIQUES
    # ========================================================================
    
    axes[1, 1].axis('off')
    
    stats_text = f"""
    STATISTIQUES DES RÉSIDUS
    
    Moyenne         : {residuals.mean():.6f}
    Médiane         : {np.median(residuals):.6f}
    Écart-type      : {residuals.std():.6f}
    
    Min             : {residuals.min():.6f}
    Max             : {residuals.max():.6f}
    Q1 (25%)        : {np.percentile(residuals, 25):.6f}
    Q3 (75%)        : {np.percentile(residuals, 75):.6f}
    
    Skewness        : {stats.skew(residuals):.6f}
    Kurtosis        : {stats.kurtosis(residuals):.6f}
    
    Test Shapiro-Wilk:
      Statistique   : {shapiro_stat:.6f}
      P-value       : {shapiro_pval:.6f}
      Normalité     : {'✓ Oui' if shapiro_pval > 0.05 else '✗ Non'}
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text,
                    transform=axes[1, 1].transAxes,
                    verticalalignment='top',
                    fontsize=11,
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée : {save_path}")
    
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> pd.DataFrame:
    """
    Visualise l'importance des features.
    
    Parameters:
    -----------
    model : estimator
        Modèle entraîné
    feature_names : list
        Noms des features
    top_n : int
        Nombre de top features à afficher
    title : str
        Titre du graphique
    save_path : Path, optional
        Chemin de sauvegarde
    figsize : tuple
        Taille de la figure
    
    Returns:
    --------
    pd.DataFrame : Importances triées
    
    Examples:
    ---------
    >>> importance_df = plot_feature_importance(model, X_train.columns, top_n=20)
    """
    
    # Extraire les importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_type = "Feature Importances"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        importance_type = "Coefficients (valeur absolue)"
    else:
        print("⚠ Modèle ne supporte pas feature importance")
        return None
    
    # Créer DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel(importance_type, fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f"{title}\n(Top {top_n} features)", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Ajouter valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars, importance_df['importance'])):
        plt.text(value, i, f' {value:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée : {save_path}")
    
    plt.show()
    
    return importance_df


def plot_model_comparison(
    results_df: pd.DataFrame,
    title: str = "Comparaison des Modèles",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Visualise la comparaison de plusieurs modèles.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame des résultats (issu de train_multiple_models)
    title : str
        Titre du graphique
    save_path : Path, optional
        Chemin de sauvegarde
    figsize : tuple
        Taille de la figure
    
    Examples:
    ---------
    >>> plot_model_comparison(results_df, save_path=Path('figures/comparison.png'))
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    models = results_df['model'].values
    x_pos = np.arange(len(models))
    width = 0.35
    
    # ========================================================================
    # R² SCORE (Train vs Test)
    # ========================================================================
    
    axes[0, 0].bar(x_pos - width/2, results_df['train_r2'], width, 
                   label='Train', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x_pos + width/2, results_df['test_r2'], width, 
                   label='Test', alpha=0.8, color='coral')
    axes[0, 0].set_xlabel('Modèle', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('R² Score - Train vs Test', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # RMSE TEST
    # ========================================================================
    
    colors_rmse = plt.cm.Reds(np.linspace(0.4, 0.9, len(models)))
    axes[0, 1].bar(models, results_df['test_rmse_log'], color=colors_rmse, alpha=0.8)
    axes[0, 1].set_xlabel('Modèle', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE (log)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('RMSE Test - Échelle Log', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # OVERFITTING
    # ========================================================================
    
    colors_over = ['green' if x < 0.05 else 'orange' if x < 0.10 else 'red' 
                   for x in results_df['overfitting_r2']]
    axes[1, 0].bar(models, results_df['overfitting_r2'], color=colors_over, alpha=0.8)
    axes[1, 0].axhline(y=0.05, color='orange', linestyle='--', linewidth=2, 
                       label='Seuil acceptable (0.05)')
    axes[1, 0].axhline(y=0.10, color='red', linestyle='--', linewidth=2, 
                       label='Seuil critique (0.10)')
    axes[1, 0].set_xlabel('Modèle', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Overfitting (R² Train - R² Test)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Analyse Overfitting', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # TEMPS D'ENTRAÎNEMENT
    # ========================================================================
    
    colors_time = plt.cm.Blues(np.linspace(0.4, 0.9, len(models)))
    axes[1, 1].bar(models, results_df['train_time'], color=colors_time, alpha=0.8)
    axes[1, 1].set_xlabel('Modèle', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Temps (secondes)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Temps d\'Entraînement', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée : {save_path}")
    
    plt.show()


def plot_learning_curves(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    title: str = "Learning Curves",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualise les courbes d'apprentissage.
    
    Parameters:
    -----------
    model : estimator
        Modèle sklearn
    X_train, y_train : pd.DataFrame, pd.Series
        Données d'entraînement
    cv : int
        Nombre de folds
    title : str
        Titre du graphique
    save_path : Path, optional
        Chemin de sauvegarde
    figsize : tuple
        Taille de la figure
    
    Examples:
    ---------
    >>> plot_learning_curves(model, X_train, y_train, cv=5)
    """
    
    from sklearn.model_selection import learning_curve
    
    print("🔄 Calcul des courbes d'apprentissage en cours...")
    print("   (Cela peut prendre quelques minutes)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=figsize)
    
    plt.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Score Train', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='steelblue')
    
    plt.plot(train_sizes, test_mean, 'o-', color='coral', label='Score Validation', linewidth=2)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                     alpha=0.2, color='coral')
    
    plt.xlabel('Taille du Dataset d\'Entraînement', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée : {save_path}")
    
    plt.show()
    
    print("✓ Courbes d'apprentissage générées")


def plot_comparison_two_models(
    results_m1: pd.DataFrame,
    results_m2: pd.DataFrame,
    model1_name: str = "Modèle 1",
    model2_name: str = "Modèle 2",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 6)
) -> None:
    """
    Compare visuellement deux modèles côte à côte.
    
    Parameters:
    -----------
    results_m1, results_m2 : pd.DataFrame
        Résultats des deux modèles
    model1_name, model2_name : str
        Noms des modèles
    save_path : Path, optional
        Chemin de sauvegarde
    figsize : tuple
        Taille de la figure
    
    Examples:
    ---------
    >>> plot_comparison_two_models(results_df_m1, results_df_m2, 
    ...                            "Sans ENERGY STAR", "Avec ENERGY STAR")
    """
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Meilleurs modèles de chaque
    best_m1 = results_m1.iloc[0]
    best_m2 = results_m2.iloc[0]
    
    metrics = ['test_r2', 'test_rmse_log', 'test_mape']
    labels = ['R² Test', 'RMSE Test (log)', 'MAPE Test (%)']
    colors = ['steelblue', 'coral']
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [best_m1[metric], best_m2[metric]]
        bars = axes[i].bar([model1_name, model2_name], values, color=colors, alpha=0.8)
        
        # Ajouter valeurs sur barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        axes[i].set_ylabel(label, fontsize=12, fontweight='bold')
        axes[i].set_title(label, fontsize=13, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Comparaison: {model1_name} vs {model2_name}', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée : {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Module visualization_utils chargé avec succès !")
    print(f"\nFonctions disponibles :")
    print("  - plot_predictions()")
    print("  - plot_residuals_distribution()")
    print("  - plot_feature_importance()")
    print("  - plot_model_comparison()")
    print("  - plot_learning_curves()")
    print("  - plot_comparison_two_models()")
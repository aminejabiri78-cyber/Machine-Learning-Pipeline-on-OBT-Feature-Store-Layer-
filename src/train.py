"""
src/train.py — Entraînement des modèles ML (régression & classification)
Cross-validation · RandomizedSearchCV · Export du meilleur modèle
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    CLASSIFICATION_MODELS,
    CV_CONFIG,
    EXPORT_CONFIG,
    REGRESSION_MODELS,
)
from src.utils import get_logger, save_json, save_pickle

logger = get_logger("train")

# Imports optionnels XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost non installé — modèles XGB ignorés")


# ─── Registry des modèles ───────────────────────────────────────────────────────
REGRESSOR_REGISTRY = {
    "LinearRegression":             LinearRegression,
    "Ridge":                        Ridge,
    "Lasso":                        Lasso,
    "RandomForestRegressor":        RandomForestRegressor,
    "GradientBoostingRegressor":    GradientBoostingRegressor,
}

CLASSIFIER_REGISTRY = {
    "LogisticRegression":           LogisticRegression,
    "RandomForestClassifier":       RandomForestClassifier,
    "GradientBoostingClassifier":   GradientBoostingClassifier,
    "SVC":                          SVC,
}

if HAS_XGB:
    REGRESSOR_REGISTRY["XGBRegressor"]   = XGBRegressor
    CLASSIFIER_REGISTRY["XGBClassifier"] = XGBClassifier


# ─── Helpers ────────────────────────────────────────────────────────────────────
def _instantiate(registry: dict, name: str) -> BaseEstimator:
    cls = registry.get(name)
    if cls is None:
        raise ValueError(f"Modèle inconnu : '{name}'. Disponibles : {list(registry)}")
    return cls()


def _param_grid_to_dist(param_grid: dict) -> dict:
    """Convertit les listes de valeurs en format compatible RandomizedSearchCV."""
    from scipy.stats import uniform  # noqa: F401  (utilisé dans des cas étendus)
    return {k: v for k, v in param_grid.items() if isinstance(v, list)}


# ─── Cross-validation simple ────────────────────────────────────────────────────
def cross_validate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    task: str = "regression",
) -> Dict[str, float]:
    """
    Évalue un modèle par cross-validation stratifiée.

    Returns:
        dict avec mean et std du scoring
    """
    scoring = CV_CONFIG["scoring_reg"] if task == "regression" else CV_CONFIG["scoring_clf"]
    cv      = CV_CONFIG["cv_folds"]

    scores = cross_val_score(
        model, X, y, cv=cv, scoring=scoring, n_jobs=CV_CONFIG["n_jobs"]
    )
    result = {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}
    logger.info(f"  CV {scoring} : {scores.mean():.4f} ± {scores.std():.4f}")
    return result


# ─── Hyperparameter tuning ──────────────────────────────────────────────────────
def tune_model(
    model: BaseEstimator,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
) -> Tuple[BaseEstimator, dict]:
    """
    Optimisation par RandomizedSearchCV.

    Returns:
        (best_model, best_params)
    """
    if not param_grid:
        logger.info("  Pas de grille d'hyperparamètres — modèle utilisé tel quel")
        model.fit(X_train, y_train)
        return model, {}

    scoring = CV_CONFIG["scoring_reg"] if task == "regression" else CV_CONFIG["scoring_clf"]

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=_param_grid_to_dist(param_grid),
        n_iter=CV_CONFIG["n_iter_random"],
        cv=CV_CONFIG["cv_folds"],
        scoring=scoring,
        n_jobs=CV_CONFIG["n_jobs"],
        random_state=CV_CONFIG["random_state"],
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(f"  Meilleurs params : {search.best_params_}")
    logger.info(f"  Meilleur score CV : {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


# ─── Entraînement d'une liste de modèles ───────────────────────────────────────
def train_all_models(
    model_configs: Dict[str, dict],
    registry: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
    tune: bool = True,
) -> List[Dict[str, Any]]:
    """
    Entraîne et évalue tous les modèles définis dans model_configs.

    Returns:
        Liste de résultats triés par score CV décroissant
    """
    results = []

    for model_name, param_grid in model_configs.items():
        if model_name not in registry:
            logger.warning(f"  '{model_name}' non disponible — ignoré")
            continue

        logger.info(f"\n▶ {model_name}")
        t0 = time.time()

        model = _instantiate(registry, model_name)

        # CV avant tuning pour baseline
        cv_baseline = cross_validate_model(model, X_train, y_train, task)

        # Tuning
        if tune and param_grid:
            best_model, best_params = tune_model(model, param_grid, X_train, y_train, task)
        else:
            model.fit(X_train, y_train)
            best_model  = model
            best_params = {}

        elapsed = time.time() - t0

        results.append({
            "name":        model_name,
            "model":       best_model,
            "cv_mean":     cv_baseline["cv_mean"],
            "cv_std":      cv_baseline["cv_std"],
            "best_params": best_params,
            "fit_time_s":  round(elapsed, 2),
        })

        logger.info(f"  Temps : {elapsed:.1f}s")

    # Tri par score décroissant
    results.sort(key=lambda x: x["cv_mean"], reverse=True)
    return results


# ─── Pipeline entraînement RÉGRESSION ──────────────────────────────────────────
def run_training_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Entraîne tous les modèles de régression, sélectionne le meilleur.

    Returns:
        dict avec best_model, all_results
    """
    logger.info("═" * 60)
    logger.info("ÉTAPE 4a — ENTRAÎNEMENT RÉGRESSION")
    logger.info("═" * 60)

    results = train_all_models(
        REGRESSION_MODELS, REGRESSOR_REGISTRY,
        X_train, y_train,
        task="regression", tune=tune,
    )

    best = results[0]
    logger.info(f"\n★ Meilleur modèle de régression : {best['name']}  (CV R²={best['cv_mean']:.4f})")

    if save:
        save_pickle(best["model"], EXPORT_CONFIG["best_reg_model"])

    # Résumé JSON (sans objet model)
    summary = [
        {k: v for k, v in r.items() if k != "model"}
        for r in results
    ]
    logger.info("\nClassement des modèles :")
    for i, r in enumerate(summary):
        logger.info(f"  {i+1}. {r['name']:35s} CV={r['cv_mean']:.4f} ± {r['cv_std']:.4f}")

    return {"best_model": best["model"], "best_name": best["name"], "all_results": results}


# ─── Pipeline entraînement CLASSIFICATION ──────────────────────────────────────
def run_training_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Entraîne tous les modèles de classification, sélectionne le meilleur.

    Returns:
        dict avec best_model, all_results
    """
    logger.info("═" * 60)
    logger.info("ÉTAPE 4b — ENTRAÎNEMENT CLASSIFICATION")
    logger.info("═" * 60)

    results = train_all_models(
        CLASSIFICATION_MODELS, CLASSIFIER_REGISTRY,
        X_train, y_train,
        task="classification", tune=tune,
    )

    best = results[0]
    logger.info(f"\n★ Meilleur modèle de classification : {best['name']}  (CV F1={best['cv_mean']:.4f})")

    if save:
        save_pickle(best["model"], EXPORT_CONFIG["best_clf_model"])

    summary = [
        {k: v for k, v in r.items() if k != "model"}
        for r in results
    ]
    logger.info("\nClassement des modèles :")
    for i, r in enumerate(summary):
        logger.info(f"  {i+1}. {r['name']:35s} CV={r['cv_mean']:.4f} ± {r['cv_std']:.4f}")

    return {"best_model": best["model"], "best_name": best["name"], "all_results": results}


if __name__ == "__main__":
    import numpy as np

    # Données synthétiques pour test rapide
    rng = np.random.default_rng(42)
    X  = rng.standard_normal((500, 20))
    yr = rng.standard_normal(500) * 100_000 + 250_000
    yc = rng.integers(0, 4, 500)

    reg = run_training_regression(X, yr, tune=False, save=False)
    clf = run_training_classification(X, yc, tune=False, save=False)
    print("Reg best :", reg["best_name"])
    print("Clf best :", clf["best_name"])
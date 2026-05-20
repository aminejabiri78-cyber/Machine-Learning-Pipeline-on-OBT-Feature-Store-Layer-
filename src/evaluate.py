"""
src/evaluate.py — Évaluation complète des modèles ML
Métriques · Feature Importance · Analyse des erreurs · Comparaison · Export
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # mode sans display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import EXPORT_CONFIG, MODELS_DIR
from src.utils import get_logger, save_json

logger = get_logger("evaluate")

PLOTS_DIR = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RÉGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_regression(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calcule et affiche les métriques de régression sur le jeu de test.

    Métriques : MAE, MSE, RMSE, R², MAPE
    """
    logger.info("═" * 60)
    logger.info(f"ÉVALUATION RÉGRESSION — {model_name}")
    logger.info("═" * 60)

    y_pred = model.predict(X_test)

    mae   = mean_absolute_error(y_test, y_pred)
    mse   = mean_squared_error(y_test, y_pred)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_test, y_pred)
    mape  = _mape(y_test, y_pred)

    metrics = {
        "model":   model_name,
        "MAE":     round(mae,  2),
        "MSE":     round(mse,  2),
        "RMSE":    round(rmse, 2),
        "R2":      round(r2,   6),
        "MAPE_%":  round(mape, 4),
    }

    logger.info(f"  MAE    : {mae:>15,.2f}")
    logger.info(f"  RMSE   : {rmse:>15,.2f}")
    logger.info(f"  R²     : {r2:>15.4f}")
    logger.info(f"  MAPE   : {mape:>14.2f}%")

    # Graphiques
    _plot_predictions_vs_actual(y_test, y_pred, model_name)
    _plot_residuals(y_test, y_pred, model_name)

    # Feature importance
    if feature_names:
        _plot_feature_importance(model, X_test, y_test, feature_names, model_name, task="regression")

    return metrics


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _plot_predictions_vs_actual(y_true, y_pred, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#2563EB")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Idéal")
    ax.set_xlabel("Valeur réelle")
    ax.set_ylabel("Valeur prédite")
    ax.set_title(f"{name} — Prédictions vs Réel")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name}_pred_vs_actual.png", dpi=120)
    plt.close(fig)
    logger.info(f"  Graphique sauvegardé : pred_vs_actual")


def _plot_residuals(y_true, y_pred, name: str) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="#7C3AED")
    axes[0].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Valeur prédite")
    axes[0].set_ylabel("Résidu")
    axes[0].set_title("Résidus vs Prédictions")

    axes[1].hist(residuals, bins=40, color="#7C3AED", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Résidu")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution des résidus")

    plt.suptitle(f"{name} — Analyse des résidus")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name}_residuals.png", dpi=120)
    plt.close(fig)
    logger.info(f"  Graphique sauvegardé : residuals")


def evaluate_classification(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:

    logger.info("═" * 60)
    logger.info(f"ÉVALUATION CLASSIFICATION — {model_name}")
    logger.info("═" * 60)

    # =====================================================
    # PREDICTIONS
    # =====================================================

    y_pred = model.predict(X_test)

    # =====================================================
    # METRICS
    # =====================================================

    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # =====================================================
    # ROC AUC
    # =====================================================

    roc_auc = None

    if hasattr(model, "predict_proba"):

        try:

            y_proba = model.predict_proba(X_test)

            # Multiclass
            if len(np.unique(y_test)) > 2:

                roc_auc = roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class="ovr",
                    average="weighted"
                )

            # Binary
            elif y_proba.shape[1] > 1:

                roc_auc = roc_auc_score(
                    y_test,
                    y_proba[:, 1]
                )

        except Exception as e:

            logger.warning(
                f"ROC-AUC non calculable : {e}"
            )

    # =====================================================
    # SAFE CLASSIFICATION REPORT
    # =====================================================

    labels = np.unique(
        np.concatenate([y_test, y_pred])
    )

    if class_names is not None:

        valid_class_names = [
            class_names[i]
            for i in labels
            if i < len(class_names)
        ]

    else:

        valid_class_names = [
            str(i)
            for i in labels
        ]

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=valid_class_names,
        zero_division=0
    )

    logger.info("\n" + report)

    # =====================================================
    # METRICS DICT
    # =====================================================

    metrics = {
        "model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_score": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else None,
    }

    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  F1 Score : {f1:.4f}")

    if roc_auc is not None:
        logger.info(f"  ROC-AUC  : {roc_auc:.4f}")
    else:
        logger.info("  ROC-AUC  : N/A")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    try:

        _plot_confusion_matrix(
            y_test,
            y_pred,
            model_name,
            valid_class_names
        )

    except Exception as e:

        logger.warning(
            f"Erreur confusion matrix : {e}"
        )

    # =====================================================
    # ROC CURVES
    # =====================================================

    try:

        if (
            hasattr(model, "predict_proba")
            and roc_auc is not None
            and len(labels) > 1
        ):

            _plot_roc_curves(
                y_test,
                model.predict_proba(X_test),
                model_name,
                valid_class_names
            )

    except Exception as e:

        logger.warning(
            f"Erreur ROC curves : {e}"
        )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    try:

        if feature_names:

            _plot_feature_importance(
                model,
                X_test,
                y_test,
                feature_names,
                model_name,
                task="classification",
            )

    except Exception as e:

        logger.warning(
            f"Erreur feature importance : {e}"
        )

    return metrics


def _plot_confusion_matrix(y_true, y_pred, name: str, class_names=None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{name} — Matrice de confusion")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name}_confusion_matrix.png", dpi=120)
    plt.close(fig)
    logger.info(f"  Graphique sauvegardé : confusion_matrix")


def _plot_roc_curves(y_true, y_proba, name: str, class_names=None) -> None:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    n_classes = y_proba.shape[1]
    classes   = list(range(n_classes))
    y_bin     = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, color in zip(classes, colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc_i   = auc(fpr, tpr)
        label = class_names[i] if class_names else f"Classe {i}"
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{label} (AUC={roc_auc_i:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title(f"{name} — Courbes ROC")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name}_roc_curves.png", dpi=120)
    plt.close(fig)
    logger.info(f"  Graphique sauvegardé : roc_curves")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_feature_importance(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    task: str = "regression",
    top_n: int = 20,
) -> None:
    """
    Affiche les features les plus importantes.
    Utilise feature_importances_ si disponible, sinon permutation importance.
    """
    scoring = "r2" if task == "regression" else "f1_weighted"

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1][:top_n]
        method      = "Native"
    else:
        logger.info(f"  Calcul permutation importance pour {model_name} …")
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, scoring=scoring
        )
        importances = perm.importances_mean
        indices     = np.argsort(importances)[::-1][:top_n]
        method      = "Permutation"

    top_features = [feature_names[i] if i < len(feature_names) else f"f_{i}" for i in indices]
    top_vals     = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    bars = ax.barh(range(len(top_features)), top_vals[::-1], color="#2563EB", alpha=0.85)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel(f"Importance ({method})")
    ax.set_title(f"{model_name} — Top {top_n} Features")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model_name}_feature_importance.png", dpi=120)
    plt.close(fig)
    logger.info(f"  Graphique sauvegardé : feature_importance ({method})")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARAISON DES MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

def compare_models(
    all_metrics: List[Dict[str, Any]],
    task: str = "regression",
    save: bool = True,
) -> pd.DataFrame:
    """
    Compare les métriques de tous les modèles testés.

    Returns:
        DataFrame de comparaison trié par métrique principale
    """
    df = pd.DataFrame(all_metrics)

    sort_col = "R2" if task == "regression" else "F1_score"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    logger.info("\n" + "═" * 60)
    logger.info(f"COMPARAISON DES MODÈLES ({task.upper()})")
    logger.info("═" * 60)
    logger.info("\n" + df.to_string(index=False))

    # Graphique de comparaison
    _plot_model_comparison(df, task)

    if save:
        metrics_dict = df.to_dict(orient="records")
        existing = {}
        try:
            import json
            with open(EXPORT_CONFIG["metrics_report"]) as f:
                existing = json.load(f)
        except Exception:
            pass
        existing[task] = metrics_dict
        save_json(existing, EXPORT_CONFIG["metrics_report"])

    return df


def _plot_model_comparison(df: pd.DataFrame, task: str) -> None:
    metric  = "R2" if task == "regression" else "F1_score"
    if metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#16A34A" if i == 0 else "#2563EB" for i in range(len(df))]
    ax.barh(df["model"], df[metric], color=colors, alpha=0.85)
    ax.set_xlabel(metric)
    ax.set_title(f"Comparaison des modèles ({task})")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"comparison_{task}.png", dpi=120)
    plt.close(fig)
    logger.info(f"  Graphique comparaison sauvegardé : comparison_{task}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE D'ÉVALUATION COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    reg_data: dict,
    clf_data: dict,
    reg_results: dict,
    clf_results: dict,
) -> Dict[str, Any]:
    """
    Évalue les meilleurs modèles de régression et de classification.
    Génère tous les graphiques et le rapport JSON.

    Returns:
        dict {"regression": metrics, "classification": metrics}
    """
    logger.info("═" * 60)
    logger.info("ÉTAPE 5 — ÉVALUATION FINALE")
    logger.info("═" * 60)

    # ── Régression ──
    reg_metrics_list = []
    for result in reg_results["all_results"]:
        m = evaluate_regression(
            result["model"],
            reg_data["X_test"], reg_data["y_test"],
            model_name=result["name"],
            feature_names=reg_data.get("feature_names"),
        )
        reg_metrics_list.append(m)

    reg_comparison = compare_models(reg_metrics_list, task="regression")

    # ── Classification ──
    clf_metrics_list = []
    label_encoder = clf_data.get("label_encoder")
    class_names   = list(label_encoder.classes_) if label_encoder else None

    for result in clf_results["all_results"]:
        m = evaluate_classification(
            result["model"],
            clf_data["X_test"], clf_data["y_test"],
            model_name=result["name"],
            class_names=class_names,
            feature_names=clf_data.get("feature_names"),
        )
        clf_metrics_list.append(m)

    clf_comparison = compare_models(clf_metrics_list, task="classification")

    logger.info("\n✅ Évaluation terminée. Graphiques dans : " + str(PLOTS_DIR))

    return {
        "regression":      reg_comparison.to_dict(orient="records"),
        "classification":  clf_comparison.to_dict(orient="records"),
    }


if __name__ == "__main__":
    # Test rapide avec données synthétiques
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    yr = rng.standard_normal(200) * 100_000 + 250_000
    yc = rng.integers(0, 3, 200)

    fn = [f"feature_{i}" for i in range(10)]

    reg = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, yr)
    evaluate_regression(reg, X, yr, "RF_test", fn)

    clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, yc)
    evaluate_classification(clf, X, yc, "RF_clf_test", None, fn)
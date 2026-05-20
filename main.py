"""
main.py — Orchestrateur principal du pipeline ML immobilier
Extraction OBT → Feature Engineering → Preprocessing → Training → Evaluation

Usage :
    python main.py                     # pipeline complet
    python main.py --task regression   # régression uniquement
    python main.py --task classification
    python main.py --no-tune           # sans hyperparameter tuning
    python main.py --limit 5000        # test rapide sur N lignes
    python main.py --demo              # données synthétiques (sans BDD)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import EXPORT_CONFIG, PROCESSED_DIR, RAW_DIR
from src.evaluate import run_evaluation
from src.extract import run_extraction
from src.features import run_feature_engineering
from src.preprocess import run_preprocessing_classification, run_preprocessing_regression
from src.train import run_training_classification, run_training_regression
from src.utils import get_logger, load_dataframe, save_json

logger = get_logger("main")


# ─── Demo mode (données synthétiques) ──────────────────────────────────────────
def _generate_demo_data(n: int = 2000):
    """Génère un DataFrame synthétique pour tester le pipeline sans BDD."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    cities    = ["Paris", "Lyon", "Marseille", "Bordeaux", "Nantes"]
    prop_types = ["apartment", "house", "studio", "villa"]
    conditions = ["new", "good", "average", "renovation"]
    heatings   = ["gas", "electric", "heat_pump"]
    exposures  = ["north", "south", "east", "west"]

    df = pd.DataFrame({
        "surface_m2":          rng.uniform(20, 300, n),
        "nb_rooms":            rng.integers(1, 10, n),
        "nb_bedrooms":         rng.integers(0, 6, n),
        "nb_bathrooms":        rng.integers(1, 4, n),
        "floor":               rng.integers(0, 20, n),
        "total_floors":        rng.integers(1, 25, n),
        "age_building":        rng.integers(0, 120, n),
        "distance_center_km":  rng.uniform(0, 60, n),
        "distance_transport_km": rng.uniform(0.1, 5, n),
        "energy_score":        rng.uniform(10, 500, n),
        "property_type":       rng.choice(prop_types, n),
        "city":                rng.choice(cities, n),
        "district":            rng.choice(["A", "B", "C", "D"], n),
        "condition":           rng.choice(conditions, n),
        "heating_type":        rng.choice(heatings, n),
        "exposure":            rng.choice(exposures, n),
        "has_parking":         rng.integers(0, 2, n).astype(bool),
        "has_garden":          rng.integers(0, 2, n).astype(bool),
        "has_terrace":         rng.integers(0, 2, n).astype(bool),
        "has_elevator":        rng.integers(0, 2, n).astype(bool),
        "listing_date":        pd.date_range("2020-01-01", periods=n, freq="6H"),
    })

    # Prix synthétique corrélé aux features
    df["price"] = (
        df["surface_m2"] * rng.uniform(2000, 8000, n)
        + df["nb_rooms"] * 5000
        - df["distance_center_km"] * 500
        + rng.normal(0, 15000, n)
    ).clip(lower=30_000)

    logger.info(f"Données synthétiques générées : {df.shape}")
    return df


# ─── Étapes du pipeline ─────────────────────────────────────────────────────────
def step_extract(args) -> object:
    if args.demo:
        logger.info("MODE DÉMO — données synthétiques (pas de BDD)")
        df = _generate_demo_data(n=args.limit or 2000)
        from src.utils import save_dataframe
        save_dataframe(df, RAW_DIR / "obt_extracted.parquet")
        return df

    if args.skip_extract and (RAW_DIR / "obt_extracted.parquet").exists():
        logger.info("Extraction ignorée — chargement depuis disque")
        return load_dataframe(RAW_DIR / "obt_extracted.parquet")

    return run_extraction(limit=args.limit, save=True)


def step_features(df) -> object:
    feat_path = PROCESSED_DIR / "obt_features.parquet"
    return run_feature_engineering(df, save=True)


def step_preprocess_reg(df) -> dict:
    return run_preprocessing_regression(df, save=True)


def step_preprocess_clf(df) -> dict:
    return run_preprocessing_classification(df, save=True)


def step_train_reg(reg_data: dict, tune: bool) -> dict:
    return run_training_regression(
        reg_data["X_train"], reg_data["y_train"],
        tune=tune, save=True,
    )


def step_train_clf(clf_data: dict, tune: bool) -> dict:
    return run_training_classification(
        clf_data["X_train"], clf_data["y_train"],
        tune=tune, save=True,
    )


# ─── Pipeline principal ─────────────────────────────────────────────────────────
def run_pipeline(args) -> None:
    t_start = time.time()
    logger.info("╔" + "═" * 62 + "╗")
    logger.info("║       PIPELINE ML IMMOBILIER — DÉMARRAGE               ║")
    logger.info("╚" + "═" * 62 + "╝")
    logger.info(f"  Task       : {args.task}")
    logger.info(f"  Tuning     : {'oui' if args.tune else 'non'}")
    logger.info(f"  Demo mode  : {'oui' if args.demo else 'non'}")
    logger.info(f"  Limit      : {args.limit or 'toutes les lignes'}")

    # ── 1. Extraction ──
    df_raw = step_extract(args)

    # ── 2. Feature Engineering ──
    df_feat = step_features(df_raw)

    # ── 3-5. Régression ──
    reg_data = reg_results = None
    if args.task in ("regression", "all"):
        reg_data    = step_preprocess_reg(df_feat.copy())
        reg_results = step_train_reg(reg_data, tune=args.tune)

    # ── 3-5. Classification ──
    clf_data = clf_results = None
    if args.task in ("classification", "all"):
        clf_data    = step_preprocess_clf(df_feat.copy())
        clf_results = step_train_clf(clf_data, tune=args.tune)

    # ── 6. Évaluation ──
    if reg_data and clf_data:
        eval_report = run_evaluation(reg_data, clf_data, reg_results, clf_results)
        save_json(eval_report, EXPORT_CONFIG["metrics_report"])
    elif reg_data:
        from src.evaluate import evaluate_regression, compare_models
        metrics_list = []
        for r in reg_results["all_results"]:
            m = evaluate_regression(
                r["model"], reg_data["X_test"], reg_data["y_test"],
                r["name"], reg_data.get("feature_names"),
            )
            metrics_list.append(m)
        compare_models(metrics_list, "regression", save=True)
    elif clf_data:
        from src.evaluate import evaluate_classification, compare_models
        label_enc   = clf_data.get("label_encoder")
        class_names = list(label_enc.classes_) if label_enc else None
        metrics_list = []
        for r in clf_results["all_results"]:
            m = evaluate_classification(
                r["model"], clf_data["X_test"], clf_data["y_test"],
                r["name"], class_names, clf_data.get("feature_names"),
            )
            metrics_list.append(m)
        compare_models(metrics_list, "classification", save=True)

    elapsed = time.time() - t_start
    logger.info("╔" + "═" * 62 + "╗")
    logger.info(f"║  ✅ PIPELINE TERMINÉ EN {elapsed:.0f}s" + " " * (36 - len(f"{elapsed:.0f}")) + "║")
    logger.info("╚" + "═" * 62 + "╝")
    logger.info(f"  Modèles exportés dans  : {EXPORT_CONFIG['best_reg_model'].parent}")
    logger.info(f"  Rapport métriques      : {EXPORT_CONFIG['metrics_report']}")


# ─── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline ML Immobilier — One Big Table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification", "all"],
        default="all",
        help="Tâche ML à exécuter (défaut: all)",
    )
    parser.add_argument(
        "--no-tune", dest="tune", action="store_false",
        help="Désactive le hyperparameter tuning (plus rapide)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limite le nombre de lignes extraites (test rapide)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Utilise des données synthétiques (sans connexion BDD)",
    )
    parser.add_argument(
        "--skip-extract", dest="skip_extract", action="store_true",
        help="Réutilise l'extraction précédente (si disponible)",
    )
    parser.set_defaults(tune=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
"""
config.py — Configuration centrale du pipeline ML immobilier
"""

import os
from pathlib import Path

# ─── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models"
LOGS_DIR      = BASE_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Base de données ────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 5433)),
    "database": os.getenv("DB_NAME",     "avito_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "user"),
    "schema":   "ml_schema",
    "table":    "ml_annonces",
}

# ─── Split ──────────────────────────────────────────────────────────────────────
SPLIT_CONFIG = {
    "test_size":       0.20,
    "val_size":        0.10,
    "random_state":    42,
    "stratify_column": None,
}

TEST_SIZE  = 0.20
CV_FOLDS   = 5
SMOTE_THRESHOLD = 0.20

# ─── Features réelles de ml_annonces ───────────────────────────────────────────
TARGET_REGRESSION     = "prix"
TARGET_CLASSIFICATION = "property_category"   # créée automatiquement

NUMERIC_FEATURES = [
    "surface", "chambres", "salles_bain", "etage",
    "annee", "prix_m2", "nb_pieces", "surface_par_piece",
]

CATEGORICAL_FEATURES = [
    "ville", "quartier",
]

DATE_FEATURES = []
DROP_COLUMNS  = ["titre"]   # texte brut inutile pour ML

# ─── Preprocessing ─────────────────────────────────────────────────────────────
PREPROCESS_CONFIG = {
    "imputer_strategy_num": "median",
    "imputer_strategy_cat": "most_frequent",
    "scaler":               "standard",
    "encoder":              "onehot",
    "use_smote":            True,
    "smote_strategy":       "auto",
    "smote_random_state":   42,
}

# ─── Feature Engineering ───────────────────────────────────────────────────────
FEATURE_ENG_CONFIG = {
    "log_transform_cols":   ["prix", "surface"],
    "ratio_cols":           [("prix", "surface"), ("prix", "nb_pieces")],
    "interaction_cols":     [("surface", "nb_pieces"), ("chambres", "surface")],
    "price_bins":           [0, 100_000, 250_000, 500_000, 1_000_000, float("inf")],
    "price_bin_labels":     ["very_low", "low", "medium", "high", "luxury"],
}

# ─── Modèles ────────────────────────────────────────────────────────────────────
REGRESSION_MODELS = {
    "LinearRegression":          {},
    "Ridge":                     {"alpha": [0.1, 1.0, 10.0]},
    "Lasso":                     {"alpha": [0.01, 0.1, 1.0]},
    "RandomForestRegressor":     {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "GradientBoostingRegressor": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "XGBRegressor":              {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]},
}

CLASSIFICATION_MODELS = {
    "LogisticRegression":         {"C": [0.1, 1.0, 10.0], "max_iter": [1000]},
    "RandomForestClassifier":     {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "GradientBoostingClassifier": {"n_estimators": [100], "learning_rate": [0.05, 0.1]},
    "XGBClassifier":              {"n_estimators": [100, 200], "max_depth": [4, 6]},
    "SVC":                        {"C": [1.0, 10.0], "kernel": ["rbf"]},
}

# ─── Cross-validation ──────────────────────────────────────────────────────────
CV_CONFIG = {
    "cv_folds":      5,
    "scoring_reg":   "r2",
    "scoring_clf":   "f1_weighted",
    "n_iter_random": 20,
    "random_state":  42,
    "n_jobs":        -1,
}

# ─── Export ────────────────────────────────────────────────────────────────────
EXPORT_CONFIG = {
    "best_reg_model": MODELS_DIR / "best_regressor.pkl",
    "best_clf_model": MODELS_DIR / "best_classifier.pkl",
    "preprocessor":   MODELS_DIR / "preprocessor.pkl",
    "feature_names":  MODELS_DIR / "feature_names.json",
    "metrics_report": MODELS_DIR / "metrics_report.json",
}

# Chemins pour app.py
BEST_REG_MODEL_PATH = EXPORT_CONFIG["best_reg_model"]
BEST_CLF_MODEL_PATH = EXPORT_CONFIG["best_clf_model"]
FEATURE_NAMES_PATH  = EXPORT_CONFIG["feature_names"]
RESULTS_REG_PATH    = MODELS_DIR / "results_regression.csv"
RESULTS_CLF_PATH    = MODELS_DIR / "results_classification.csv"
OBT_CSV_PATH        = RAW_DIR / "obt_extracted.parquet"

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_CONFIG = {
    "level":   "INFO",
    "format":  "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "file":    LOGS_DIR / "pipeline.log",
}
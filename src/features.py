<<<<<<< HEAD
import numpy as np

def add_features(df):

    df["prix_m2"] = df["prix"] / df["surface"]
    df["surface_chambres"] = df["surface"] * df["chambres"]
    df["log_prix"] = np.log1p(df["prix"])
=======
"""
src/features.py — Feature Engineering basé sur les colonnes réelles de ml_annonces
Colonnes : titre, ville, quartier, prix, surface, chambres, salles_bain,
           etage, annee, prix_m2, nb_pieces, surface_par_piece
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import FEATURE_ENG_CONFIG, PROCESSED_DIR, TARGET_CLASSIFICATION, TARGET_REGRESSION
from src.utils import describe_dataframe, get_logger, save_dataframe

logger = get_logger("features")


def log_transform(df):
    cols = [c for c in FEATURE_ENG_CONFIG.get("log_transform_cols", []) if c in df.columns]
    for col in cols:
        new_col = f"log_{col}"
        df[new_col] = np.log1p(df[col].clip(lower=0))
        logger.info(f"Log-transform : {col} → {new_col}")
    return df


def create_ratios(df):
    for (num, den) in FEATURE_ENG_CONFIG.get("ratio_cols", []):
        if num in df.columns and den in df.columns:
            col_name = f"{num}_per_{den}"
            df[col_name] = df[num] / df[den].replace(0, np.nan)
            logger.info(f"Ratio : {col_name}")
    return df


def create_interactions(df):
    for (a, b) in FEATURE_ENG_CONFIG.get("interaction_cols", []):
        if a in df.columns and b in df.columns:
            col_name = f"{a}_x_{b}"
            df[col_name] = df[a] * df[b]
            logger.info(f"Interaction : {col_name}")
    return df


def create_geo_features(df):
    for col in ["ville", "quartier"]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq)
            logger.info(f"Fréquence géo : {col}_freq")

            # Prix médian par zone
            if TARGET_REGRESSION in df.columns:
                median_prix = df.groupby(col)[TARGET_REGRESSION].transform("median")
                df[f"prix_median_{col}"] = median_prix
                logger.info(f"Prix médian par {col}")
    return df


def create_quality_features(df):
    # Ancienneté du bien
    if "annee" in df.columns:
        df["age_bien"] = 2024 - df["annee"]
        df["age_bien"] = df["age_bien"].clip(lower=0)
        logger.info("Feature 'age_bien' créée")

    # Ratio chambres / surface
    if "chambres" in df.columns and "surface" in df.columns:
        df["chambres_par_m2"] = df["chambres"] / df["surface"].replace(0, np.nan)
        logger.info("Feature 'chambres_par_m2' créée")

    # Ratio salles de bain / chambres
    if "salles_bain" in df.columns and "chambres" in df.columns:
        df["ratio_sdb_ch"] = df["salles_bain"] / df["chambres"].replace(0, np.nan)
        logger.info("Feature 'ratio_sdb_ch' créée")

    # Étage normalisé (si etage présent)
    if "etage" in df.columns:
        df["etage_clean"] = df["etage"].fillna(0).clip(lower=0)

    return df


def create_classification_target(df):
    target = TARGET_REGRESSION
    if target not in df.columns:
        logger.warning(f"'{target}' absent — target classification non créée")
        return df

    bins   = FEATURE_ENG_CONFIG["price_bins"]
    labels = FEATURE_ENG_CONFIG["price_bin_labels"]
    df[TARGET_CLASSIFICATION] = pd.cut(
        df[target], bins=bins, labels=labels, right=True
    ).astype(str)

    distrib = df[TARGET_CLASSIFICATION].value_counts()
    logger.info(f"Target '{TARGET_CLASSIFICATION}' :\n{distrib.to_string()}")
    return df


def run_feature_engineering(df, save=True):
    logger.info("═" * 60)
    logger.info("ÉTAPE 2 — FEATURE ENGINEERING")
    logger.info("═" * 60)

    # Supprimer colonnes texte inutiles pour ML
    drop_cols = [c for c in ["titre"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Colonnes texte supprimées : {drop_cols}")

    n_before = df.shape[1]
    df = create_geo_features(df)
    df = create_quality_features(df)
    df = log_transform(df)
    df = create_ratios(df)
    df = create_interactions(df)
    df = create_classification_target(df)

    logger.info(f"Features créées : {df.shape[1] - n_before} nouvelles colonnes")
    describe_dataframe(df, logger)

    if save:
        save_dataframe(df, PROCESSED_DIR / "obt_features.parquet")
>>>>>>> 4548eec (last v)

    return df
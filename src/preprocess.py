<<<<<<< HEAD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor(num_cols, cat_cols):

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    return preprocessor
=======
"""
src/preprocess.py — Preprocessing post-feature-engineering
Split · Imputation · Encodage · Scaling · SMOTE
Version corrigée et sécurisée
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    CATEGORICAL_FEATURES,
    EXPORT_CONFIG,
    NUMERIC_FEATURES,
    PREPROCESS_CONFIG,
    PROCESSED_DIR,
    SPLIT_CONFIG,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)

from src.utils import (
    describe_dataframe,
    get_logger,
    save_dataframe,
    save_pickle,
)

logger = get_logger("preprocess")


# =========================================================
# SCALER
# =========================================================

def _get_scaler():

    scaler_name = PREPROCESS_CONFIG.get("scaler", "standard")

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    return scalers.get(scaler_name, StandardScaler())


# =========================================================
# DETECT COLUMNS
# =========================================================

def _detect_columns(X: pd.DataFrame):

    exclude = [TARGET_REGRESSION, TARGET_CLASSIFICATION]

    # Numeric
    if NUMERIC_FEATURES:
        num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    else:
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if c not in exclude]

    # Categorical
    if CATEGORICAL_FEATURES:
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    else:
        cat_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        cat_cols = [c for c in cat_cols if c not in exclude]

    logger.info(f"Colonnes numériques ({len(num_cols)}): {num_cols[:10]}")
    logger.info(f"Colonnes catégorielles ({len(cat_cols)}): {cat_cols[:10]}")

    return num_cols, cat_cols


# =========================================================
# SPLIT DATA
# =========================================================

def split_data(df, target):

    cfg = SPLIT_CONFIG

    X = df.drop(columns=[target])
    y = df[target]

    # =====================================================
    # CHECK STRATIFY
    # =====================================================

    stratify_value = None

    # classification فقط
    if y.dtype == "object" or y.nunique() < 20:

        class_counts = y.value_counts()

        logger.info(
            f"Distribution classes:\n{class_counts.to_string()}"
        )

        # stratify فقط إذا كل class فيها >= 2
        if class_counts.min() >= 2:

            stratify_value = y

        else:

            logger.warning(
                "Stratify ignoré: une classe contient moins de 2 samples."
            )

    # =====================================================
    # TRAIN / TEST
    # =====================================================

    X_tv, X_test, y_tv, y_test = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=stratify_value,
    )

    # =====================================================
    # VALIDATION SPLIT
    # =====================================================

    val_ratio = cfg["val_size"] / (1 - cfg["test_size"])

    stratify_tv = None

    if stratify_value is not None:

        tv_counts = y_tv.value_counts()

        if tv_counts.min() >= 2:

            stratify_tv = y_tv

    X_train, X_val, y_train, y_val = train_test_split(
        X_tv,
        y_tv,
        test_size=val_ratio,
        random_state=cfg["random_state"],
        stratify=stratify_tv,
    )

    logger.info(
        f"Split → "
        f"train:{len(X_train):,} | "
        f"val:{len(X_val):,} | "
        f"test:{len(X_test):,}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# =========================================================
# BUILD PREPROCESSOR
# =========================================================

def build_preprocessor(X_train):

    num_cols, cat_cols = _detect_columns(X_train)

    # Numeric pipeline
    num_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=PREPROCESS_CONFIG["imputer_strategy_num"]
            )
        ),
        ("scaler", _get_scaler()),
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=PREPROCESS_CONFIG["imputer_strategy_cat"]
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        ),
    ])

    transformers = []

    if num_cols:
        transformers.append(
            ("num", num_pipeline, num_cols)
        )

    if cat_cols:
        transformers.append(
            ("cat", cat_pipeline, cat_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# =========================================================
# APPLY PREPROCESSOR
# =========================================================

def apply_preprocessor(
    preprocessor,
    X_train,
    X_val,
    X_test
):

    X_train_t = preprocessor.fit_transform(X_train)

    X_val_t = preprocessor.transform(X_val)

    X_test_t = preprocessor.transform(X_test)

    try:
        feature_names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

    except Exception:

        feature_names = [
            f"f_{i}"
            for i in range(X_train_t.shape[1])
        ]

    logger.info(
        f"Préprocesseur appliqué → "
        f"{X_train_t.shape[1]} features finales"
    )

    return (
        X_train_t,
        X_val_t,
        X_test_t,
        feature_names,
    )


# =========================================================
# SAFE SMOTE
# =========================================================

def apply_smote(X_train, y_train):

    if not PREPROCESS_CONFIG.get("use_smote", False):
        return X_train, np.array(y_train)

    try:
        from imblearn.over_sampling import SMOTE

    except ImportError:
        logger.warning(
            "imbalanced-learn non installé. SMOTE ignoré."
        )
        return X_train, np.array(y_train)

    counts = pd.Series(y_train).value_counts()

    logger.info(
        f"Distribution avant SMOTE:\n"
        f"{counts.to_string()}"
    )

    # Protection contre classes trop petites
    min_count = counts.min()

    if min_count < 2:
        logger.warning(
            "SMOTE ignoré: une classe contient moins de 2 samples."
        )
        return X_train, np.array(y_train)

    # k_neighbors dynamique
    k_neighbors = min(5, min_count - 1)

    logger.info(f"SMOTE k_neighbors = {k_neighbors}")

    smote = SMOTE(
        sampling_strategy=PREPROCESS_CONFIG["smote_strategy"],
        random_state=PREPROCESS_CONFIG["smote_random_state"],
        k_neighbors=k_neighbors,
    )

    X_res, y_res = smote.fit_resample(
        X_train,
        y_train
    )

    logger.info(
        f"Après SMOTE → {X_res.shape[0]:,} lignes"
    )

    return X_res, y_res


# =========================================================
# ENCODE TARGET
# =========================================================

def encode_classification_target(
    y_train,
    y_val,
    y_test
):

    le = LabelEncoder()

    y_train_enc = le.fit_transform(y_train)

    y_val_enc = le.transform(y_val)

    y_test_enc = le.transform(y_test)

    logger.info(
        f"Classes encodées : {list(le.classes_)}"
    )

    return (
        y_train_enc,
        y_val_enc,
        y_test_enc,
        le,
    )


# =========================================================
# CLEAN DF
# =========================================================

def _clean_df(df, target):

    dt_cols = df.select_dtypes(
        include="datetime64[ns]"
    ).columns.tolist()

    if dt_cols:

        df = df.drop(columns=dt_cols)

        logger.info(
            f"Colonnes datetime supprimées : {dt_cols}"
        )

    return df


# =========================================================
# REGRESSION
# =========================================================

def run_preprocessing_regression(
    df,
    save=True
):

    logger.info("═" * 60)
    logger.info("ÉTAPE 3a — PREPROCESSING RÉGRESSION")
    logger.info("═" * 60)

    target = TARGET_REGRESSION

    if target not in df.columns:

        candidates = [
            c for c in df.columns
            if "prix" in c.lower()
            or "price" in c.lower()
        ]

        if candidates:

            target = candidates[0]

            logger.warning(
                f"Target absente → utilisation de '{target}'"
            )

        else:
            raise ValueError(
                f"Target '{target}' introuvable."
            )

    extra = [
        c for c in [TARGET_CLASSIFICATION]
        if c in df.columns
    ]

    df_reg = _clean_df(
        df.drop(columns=extra),
        target,
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df_reg, target)

    preprocessor = build_preprocessor(X_train)

    (
        X_train_t,
        X_val_t,
        X_test_t,
        feature_names,
    ) = apply_preprocessor(
        preprocessor,
        X_train,
        X_val,
        X_test,
    )

    if save:

        save_pickle(
            preprocessor,
            EXPORT_CONFIG["preprocessor"]
        )

    logger.info("Preprocessing régression terminé.")

    return {
        "X_train": X_train_t,
        "X_val": X_val_t,
        "X_test": X_test_t,
        "y_train": y_train.values,
        "y_val": y_val.values,
        "y_test": y_test.values,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "task": "regression",
    }


# =========================================================
# CLASSIFICATION
# =========================================================

def run_preprocessing_classification(
    df,
    preprocessor=None,
    save=True
):

    logger.info("═" * 60)
    logger.info("ÉTAPE 3b — PREPROCESSING CLASSIFICATION")
    logger.info("═" * 60)

    target = TARGET_CLASSIFICATION

    if target not in df.columns:
        raise ValueError(
            f"Target classification '{target}' absente."
        )

    extra = [
        c for c in [TARGET_REGRESSION]
        if c in df.columns
    ]

    df_clf = _clean_df(
        df.drop(columns=extra),
        target,
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df_clf, target)

    if preprocessor is None:
        preprocessor = build_preprocessor(X_train)

    (
        X_train_t,
        X_val_t,
        X_test_t,
        feature_names,
    ) = apply_preprocessor(
        preprocessor,
        X_train,
        X_val,
        X_test,
    )

    (
        y_train_enc,
        y_val_enc,
        y_test_enc,
        label_encoder,
    ) = encode_classification_target(
        y_train,
        y_val,
        y_test,
    )

    # SMOTE sécurisé
    X_train_t, y_train_enc = apply_smote(
        X_train_t,
        y_train_enc,
    )

    logger.info(
        "Preprocessing classification terminé."
    )

    return {
        "X_train": X_train_t,
        "X_val": X_val_t,
        "X_test": X_test_t,
        "y_train": y_train_enc,
        "y_val": y_val_enc,
        "y_test": y_test_enc,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "task": "classification",
    }
>>>>>>> 4548eec (last v)

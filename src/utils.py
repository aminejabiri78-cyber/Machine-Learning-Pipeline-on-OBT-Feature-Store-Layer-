"""
src/utils.py — Utilitaires partagés (logging, sauvegarde, chargement)
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import LOG_CONFIG


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(LOG_CONFIG["level"])
    formatter = logging.Formatter(LOG_CONFIG["format"], datefmt=LOG_CONFIG["datefmt"])
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOG_CONFIG["file"], encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def save_pickle(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    get_logger("utils").info(f"Saved pickle → {path}")


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    get_logger("utils").info(f"Loaded pickle ← {path}")
    return obj


# Alias utilisé dans app.py
def load_object(path: Path) -> Any:
    return load_pickle(path)


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    get_logger("utils").info(f"Saved JSON → {path}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: Path, fmt: str = "parquet") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    get_logger("utils").info(f"Saved DataFrame ({fmt}) → {path}  shape={df.shape}")


def load_dataframe(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    get_logger("utils").info(f"Loaded DataFrame ← {path}  shape={df.shape}")
    return df


def memory_usage(df: pd.DataFrame) -> str:
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    return f"{mem:.2f} MB"


def describe_dataframe(df: pd.DataFrame, logger: logging.Logger) -> None:
    logger.info(f"Shape       : {df.shape}")
    logger.info(f"Memory      : {memory_usage(df)}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info(f"Missing values:\n{missing.to_string()}")
    else:
        logger.info("Missing values: none")
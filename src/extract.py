<<<<<<< HEAD
import pandas as pd
import os
from config.db_config import get_engine
from config.settings import OBT_TABLE

def load_data():

    engine = get_engine()

    query = f"SELECT * FROM {OBT_TABLE}"
    df = pd.read_sql(query, engine)

    print("Data loaded:")
    df.info()

    path_dir = r"C:\Users\user\Documents\ML_Project\data\raw"
    os.makedirs(path_dir, exist_ok=True)

    file_path = os.path.join(path_dir, "ml_schema_annonces.csv")

    df.to_csv(file_path, index=False, encoding="utf-8")

    print("File saved at:", file_path)

    return df
=======
"""
src/extract.py — Extraction OBT depuis ml_schema.ml_annonces (PostgreSQL)
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DB_CONFIG, DROP_COLUMNS, RAW_DIR
from src.utils import describe_dataframe, get_logger, save_dataframe

logger = get_logger("extract")


def build_engine() -> sqlalchemy.engine.Engine:
    cfg = DB_CONFIG
    url = (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    engine = create_engine(url, connect_args={"options": f"-csearch_path={cfg['schema']}"})
    logger.info(f"Engine → {cfg['host']}:{cfg['port']}/{cfg['database']} schema={cfg['schema']}")
    return engine


def test_connection(engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connexion PostgreSQL : OK")
        return True
    except Exception as exc:
        logger.error(f"Connexion échouée : {exc}")
        return False


def extract_obt(engine, table=None, limit=None, filters=None) -> pd.DataFrame:
    table  = table or DB_CONFIG["table"]
    schema = DB_CONFIG["schema"]
    where  = f"WHERE {filters}" if filters else ""
    lim    = f"LIMIT {limit}"   if limit   else ""
    query  = f"SELECT * FROM {schema}.{table} {where} {lim}"

    logger.info(f"Extraction depuis {schema}.{table} …")
    df = pd.read_sql(text(query), con=engine)
    logger.info(f"Extrait : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    describe_dataframe(df, logger)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.warning(f"{removed} doublons supprimés")
    return df


def run_extraction(limit=None, filters=None, save=True) -> pd.DataFrame:
    logger.info("═" * 60)
    logger.info("ÉTAPE 1 — EXTRACTION OBT")
    logger.info("═" * 60)

    engine = build_engine()
    if not test_connection(engine):
        raise ConnectionError("Impossible de se connecter à la base de données.")

    df = extract_obt(engine, limit=limit, filters=filters)

    # Supprimer colonnes inutiles définies dans config
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Colonnes supprimées : {cols_to_drop}")

    df = remove_duplicates(df)

    if save:
        save_dataframe(df, RAW_DIR / "obt_extracted.parquet")

    engine.dispose()
    return df


# Alias pour compatibilité app.py
def load_obt(limit=None) -> pd.DataFrame:
    return run_extraction(limit=limit, save=False)
>>>>>>> 4548eec (last v)

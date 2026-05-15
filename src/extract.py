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
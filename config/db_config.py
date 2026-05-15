from sqlalchemy import engine,create_engine
import os 
from dotenv import load_dotenv
load_dotenv()
def get_engine():
    USER = os.getenv("DB_USER")
    PASS = os.getenv("DB_PASS")
    HOST = os.getenv("DB_HOST")
    PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    try:
        url = f"postgresql+psycopg2://{USER}:{PASS}@{HOST}:{PORT}/{DB_NAME}"
        engine = create_engine(url)
        print(f"connection creer !")
    except Exception as e :
        print(f"erreur connection a database {e}")
    return engine
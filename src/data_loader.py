#  Data Loader
import pandas as pd
from src.logger import setup_logger

logger = setup_logger()

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
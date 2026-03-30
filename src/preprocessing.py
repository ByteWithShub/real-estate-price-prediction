# Preprocessing module for the house price prediction project.
from src.logger import setup_logger

logger = setup_logger()

def split_features_target(df):
    try:
        X = df.drop("price", axis=1)
        y = df["price"]
        logger.info("Split features and target")
        return X, y
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise
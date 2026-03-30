# Model Training
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.logger import setup_logger

logger = setup_logger()

def train_models(X_train, y_train):
    try:
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        rf = RandomForestRegressor(
            n_estimators=200,
            criterion="absolute_error",
            random_state=42
        )
        rf.fit(X_train, y_train)

        logger.info("Models trained successfully")
        return lr, rf

    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise


def save_model(model, path):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
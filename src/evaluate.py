# Model Evaluation
from sklearn.metrics import mean_absolute_error
from src.logger import setup_logger

logger = setup_logger()

def evaluate(model, X_test, y_test):
    try:
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        logger.info(f"MAE: {mae}")
        return mae
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise
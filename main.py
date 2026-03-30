# Main script for the house price prediction project.
import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import split_features_target
from src.train_model import train_models, save_model
from src.evaluate import evaluate

def main():
    df = load_data("final.csv")

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    lr, rf = train_models(X_train, y_train)

    lr_mae = evaluate(lr, X_test, y_test)
    rf_mae = evaluate(rf, X_test, y_test)

    print("Linear Regression MAE:", lr_mae)
    print("Random Forest MAE:", rf_mae)

    # Save best model (RF)
    save_model(rf, "models/real_estate_model.pkl")
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    print("Model saved successfully.")

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Define paths
DATA_PATH = os.path.join('data', 'crop_production.csv')
MODEL_PATH = os.path.join('models', 'RandomForestRegressor.pkl')
FEATURE_COLUMNS_PATH = os.path.join('models', 'yield_feature_columns.pkl')

def train_yield_model():
    """
    Trains a RandomForestRegressor on the crop production dataset,
    saves the trained model, and the list of feature columns.
    """
    print("Starting yield model training with District_Name...")

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Drop rows where 'Production' is NaN
    data = df.dropna(subset=["Production"])

    # --- MODIFICATION: Only drop Crop_Year, keep District_Name ---
    # Also remove the calculated 'percent_of_production' if it exists from previous steps
    if 'percent_of_production' in data.columns:
        data = data.drop("percent_of_production", axis=1)
    data1 = data.drop(["Crop_Year"], axis=1)

    # Create dummy variables for all categorical features, including District_Name
    data_dum = pd.get_dummies(data1)

    # Separate features (X) and target (y)
    x = data_dum.drop("Production", axis=1)
    y = data_dum[["Production"]]

    # Store feature column names (this list will now be much longer)
    feature_columns = x.columns.tolist()

    # Initialize and train the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y.values.ravel())

    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save the trained model
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    print(f"Yield model saved to {MODEL_PATH}")

    # Save the feature columns
    with open(FEATURE_COLUMNS_PATH, 'wb') as file:
        pickle.dump(feature_columns, file)
    print(f"Feature columns saved to {FEATURE_COLUMNS_PATH}")

    print("Yield model training complete.")

if __name__ == '__main__':
    train_yield_model()
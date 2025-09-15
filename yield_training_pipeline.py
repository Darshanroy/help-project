import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# --- MODIFICATION: Use the new augmented data file ---
DATA_PATH = os.path.join('data', 'crop_production.csv') 
MODEL_PATH = os.path.join('models', 'RandomForestRegressor.pkl')
FEATURE_COLUMNS_PATH = os.path.join('models', 'yield_feature_columns.pkl')

def train_yield_model():
    """
    Trains a RandomForestRegressor on the AUGMENTED crop production dataset,
    now including Rainfall and Soil Type.
    """
    print("Starting yield model training with AUGMENTED data...")

    # Load the new dataset
    df = pd.read_csv(DATA_PATH)
    data = df.dropna(subset=["Production"])

    # --- MODIFICATION: Only drop Crop_Year ---
    data1 = data.drop(["Crop_Year"], axis=1)

    # Create dummy variables for all categorical features
    data_dum = pd.get_dummies(data1)

    # Separate features (X) and target (y)
    x = data_dum.drop("Production", axis=1)
    y = data_dum[["Production"]]

    # Store feature column names
    feature_columns = x.columns.tolist()

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y.values.ravel())

    # Save the model and columns
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    print(f"Yield model saved to {MODEL_PATH}")
    with open(FEATURE_COLUMNS_PATH, 'wb') as file:
        pickle.dump(feature_columns, file)
    print(f"Feature columns saved to {FEATURE_COLUMNS_PATH}")
    print("Yield model training complete.")

if __name__ == '__main__':
    train_yield_model()
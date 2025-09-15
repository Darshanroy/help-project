import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Define paths
DATA_PATH = "data\Crop_recommendation.csv"
MODEL_PATH = "models\RandomForest.pkl"

# --- Model Training ---
def train_model():
    """
    Trains a RandomForestClassifier on the crop recommendation dataset
    and saves the trained model.
    """
    print("Starting model training...")

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Define features (X) and target (y)
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']

    # Initialize and train the Random Forest model
    # Using parameters from the notebook for consistency
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(features, target)

    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save the trained model using pickle
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model training complete. Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train_model()
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

# Define paths
DATA_PATH = os.path.join('data', 'Fertilizer Prediction.csv')
MODEL_PATH = os.path.join('models', 'fertilizer_model.pkl')
ENCODERS_PATH = os.path.join('models', 'fertilizer_encoders.pkl')

def train_fertilizer_model():
    """
    Trains a RandomForestClassifier for fertilizer recommendation,
    handles label encoding and SMOTE, and saves the pipeline and encoders.
    """
    print("Starting fertilizer model training...")

    # Load the dataset
    data = pd.read_csv(DATA_PATH)

    # --- Preprocessing ---
    # Rename columns to remove trailing spaces and ensure consistency
    data.rename(columns={
        "Temparature": "Temperature",
        "Humidity ": "Humidity"
    }, inplace=True)
    
    # Initialize label encoders
    soil_type_encoder = LabelEncoder()
    crop_type_encoder = LabelEncoder()
    fertilizer_name_encoder = LabelEncoder()

    # Fit and transform the categorical columns
    data["Soil Type"] = soil_type_encoder.fit_transform(data["Soil Type"])
    data["Crop Type"] = crop_type_encoder.fit_transform(data["Crop Type"])
    data["Fertilizer Name"] = fertilizer_name_encoder.fit_transform(data["Fertilizer Name"])
    
    # Separate features (X) and target (y)
    X = data.drop("Fertilizer Name", axis=1)
    y = data["Fertilizer Name"]

    # Apply SMOTE for oversampling
    print("Applying SMOTE to balance the dataset...")
    upsample = SMOTE(random_state=42)
    X_resampled, y_resampled = upsample.fit_resample(X, y)
    
    # Define the model pipeline
    # The pipeline will first scale the data, then apply the classifier
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=18))
    
    # Train the pipeline on the resampled data
    pipeline.fit(X_resampled, y_resampled)
    
    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save the trained pipeline
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Fertilizer model pipeline saved to {MODEL_PATH}")

    # Save the encoders
    encoders = {
        'soil_type': soil_type_encoder,
        'crop_type': crop_type_encoder,
        'fertilizer_name': fertilizer_name_encoder
    }
    with open(ENCODERS_PATH, 'wb') as file:
        pickle.dump(encoders, file)
    print(f"Encoders saved to {ENCODERS_PATH}")

    print("Fertilizer model training complete.")

if __name__ == '__main__':
    train_fertilizer_model()
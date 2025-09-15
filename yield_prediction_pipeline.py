import pickle
import numpy as np
import pandas as pd
import os

# Define paths
MODEL_PATH = os.path.join('models', 'RandomForestRegressor.pkl')
FEATURE_COLUMNS_PATH = os.path.join('models', 'yield_feature_columns.pkl')

# Load the trained model and feature columns
try:
    with open(MODEL_PATH, 'rb') as file:
        yield_model = pickle.load(file)
    with open(FEATURE_COLUMNS_PATH, 'rb') as file:
        yield_feature_columns = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model or feature columns file not found. Please run yield_training_pipeline.py first.")
    yield_model = None
    yield_feature_columns = []

def predict_crop_yield(data_dict):
    """
    Takes a dictionary of features and returns the predicted crop yield.
    
    Args:
        data_dict (dict): A dictionary with keys matching the input features
                          (e.g., 'Area', 'State_Name', 'Season', 'Crop').
        
    Returns:
        float: The predicted crop yield.
    """
    if yield_model is None:
        return "Yield prediction model not loaded. Please train the model first."
        
    try:
        # Create a DataFrame from the input dictionary
        input_df = pd.DataFrame([data_dict])
        
        # Apply one-hot encoding to categorical features
        # Ensure that categories are consistent with the training data
        input_df_encoded = pd.get_dummies(input_df)

        # Reindex the input_df_encoded to match the training feature columns
        # Fill missing columns (categories not present in this specific input) with 0
        final_input = input_df_encoded.reindex(columns=yield_feature_columns, fill_value=0)
        
        # Make a prediction
        prediction = yield_model.predict(final_input)
        
        # Return the first (and only) prediction
        return round(float(prediction[0]), 2)
    except Exception as e:
        print(f"An error occurred during yield prediction: {e}")
        return f"Error in yield prediction: {e}"
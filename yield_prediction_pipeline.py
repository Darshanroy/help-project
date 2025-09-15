import pickle
import numpy as np
import pandas as pd
import os

# Paths remain the same, but the loaded model and columns will be the new ones
MODEL_PATH = os.path.join('models', 'RandomForestRegressor.pkl')
FEATURE_COLUMNS_PATH = os.path.join('models', 'yield_feature_columns.pkl')

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
    Takes a dictionary of features (including Rainfall and Soil Type) 
    and returns the predicted crop yield.
    """
    if yield_model is None:
        return "Yield prediction model not loaded."
        
    try:
        # Create a DataFrame from the input dictionary
        input_df = pd.DataFrame([data_dict])
        
        # --- MODIFICATION: It will now create dummies for Soil Type as well ---
        input_df_encoded = pd.get_dummies(input_df)

        # Reindex to match the training feature columns
        final_input = input_df_encoded.reindex(columns=yield_feature_columns, fill_value=0)
        
        prediction = yield_model.predict(final_input)
        
        return round(float(prediction[0]), 2)
    except Exception as e:
        print(f"An error occurred during yield prediction: {e}")
        return f"Error in yield prediction: {e}"
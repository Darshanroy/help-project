import pickle
import numpy as np
import os

# Define paths
MODEL_PATH = os.path.join('models', 'fertilizer_model.pkl')
ENCODERS_PATH = os.path.join('models', 'fertilizer_encoders.pkl')

# Load the trained model and encoders
try:
    with open(MODEL_PATH, 'rb') as file:
        fertilizer_model = pickle.load(file)
    with open(ENCODERS_PATH, 'rb') as file:
        encoders = pickle.load(file)
        soil_type_encoder = encoders['soil_type']
        crop_type_encoder = encoders['crop_type']
        fertilizer_name_encoder = encoders['fertilizer_name']
except FileNotFoundError:
    print(f"Error: Model or encoders file not found. Please run fertilizer_training_pipeline.py first.")
    fertilizer_model = None
    encoders = None

def predict_fertilizer(data_dict):
    """
    Takes a dictionary of features and returns the predicted fertilizer name.
    
    Args:
        data_dict (dict): A dictionary with keys matching the input features.
        
    Returns:
        str: The predicted fertilizer name.
    """
    if fertilizer_model is None or encoders is None:
        return "Fertilizer prediction model not loaded. Please train the model first."
        
    try:
        # Create a copy to avoid modifying the original dict
        processed_data = data_dict.copy()

        # Transform categorical features using the loaded encoders
        processed_data['Soil Type'] = soil_type_encoder.transform([processed_data['Soil Type']])[0]
        processed_data['Crop Type'] = crop_type_encoder.transform([processed_data['Crop Type']])[0]
        
        # Convert dictionary to a numpy array in the correct feature order
        feature_array = np.array([
            processed_data['Temperature'],
            processed_data['Humidity'],
            processed_data['Moisture'],
            processed_data['Soil Type'],
            processed_data['Crop Type'],
            processed_data['Nitrogen'],
            processed_data['Potassium'],
            processed_data['Phosphorous']
        ]).reshape(1, -1)
        
        # Make a prediction (pipeline handles scaling)
        prediction_encoded = fertilizer_model.predict(feature_array)
        
        # Inverse transform the prediction to get the fertilizer name
        prediction_name = fertilizer_name_encoder.inverse_transform(prediction_encoded)
        
        return prediction_name[0]
    except Exception as e:
        print(f"An error occurred during fertilizer prediction: {e}")
        return f"Error: Could not make a prediction. Ensure all inputs are correct."
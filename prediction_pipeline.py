import pickle
import numpy as np
import os

# Define the path to the trained model
MODEL_PATH = "models\RandomForest.pkl"

# Load the trained model
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run training_pipeline.py first.")
    model = None

def predict_crop(data):
    """
    Takes a numpy array of features and returns the predicted crop.
    
    Args:
        data (np.array): A 2D numpy array with the feature values.
        
    Returns:
        str: The predicted crop name.
    """
    if model is None:
        return "Model not loaded. Please train the model first."
        
    try:
        # Convert data to a numpy array for prediction
        input_data = np.array(data).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Return the first (and only) prediction, capitalized
        return prediction[0].capitalize()
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Error in prediction"
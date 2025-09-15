from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

from prediction_pipeline import predict_crop # Crop recommendation
from yield_prediction_pipeline import predict_crop_yield # Yield prediction
from fertilizer_prediction_pipeline import predict_fertilizer # Fertilizer recommendation

app = Flask(__name__)

# --- Load data for dropdowns ---
YIELD_DATA_PATH = os.path.join('data', 'crop_production.csv')
FERTILIZER_DATA_PATH = os.path.join('data', 'Fertilizer Prediction.csv')

yield_df = pd.read_csv(YIELD_DATA_PATH).dropna(subset=["Production"])
fertilizer_df = pd.read_csv(FERTILIZER_DATA_PATH)

# Get unique values for yield dropdowns
unique_states = sorted(yield_df['State_Name'].unique())
unique_yield_seasons = sorted(yield_df['Season'].unique())
unique_yield_crops = sorted(yield_df['Crop'].unique())

# Get unique values for fertilizer dropdowns
unique_soil_types = sorted(fertilizer_df['Soil Type'].unique())
unique_crop_types = sorted(fertilizer_df['Crop Type'].unique())


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/crop-recommend')
def crop_recommend():
    """Renders the crop recommendation page."""
    return render_template('crop.html')

@app.route('/yield-predict')
def yield_predict():
    """Renders the yield prediction page."""
    return render_template(
        'yield.html',
        states=unique_states,
        seasons=unique_yield_seasons,
        crops=unique_yield_crops
    )

@app.route('/get_districts/<state_name>')
def get_districts(state_name):
    """Returns a list of districts for a given state."""
    districts = sorted(yield_df[yield_df['State_Name'] == state_name]['District_Name'].unique())
    return jsonify({'districts': districts})

# --- MODIFICATION: Renamed route and updated template ---
@app.route('/fertilizer-recommend')
def fertilizer_recommend():
    """Renders the fertilizer recommendation page."""
    return render_template(
        'fertilizer.html',
        soil_types=unique_soil_types,
        crop_types=unique_crop_types
    )
    
@app.route('/predict', methods=['POST'])
def predict():
    """Handles crop recommendation requests."""
    # ... (no changes needed here) ...
    try:
        data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        crop_prediction = predict_crop(data)
        return jsonify({'prediction': crop_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_yield', methods=['POST'])
def predict_yield_endpoint():
    """Handles yield prediction requests."""
    # ... (no changes needed here) ...
    try:
        data_for_prediction = {
            'Area': float(request.form['Area']),
            'State_Name': request.form['State_Name'],
            'District_Name': request.form['District_Name'],
            'Season': request.form['Season'],
            'Crop': request.form['Crop']
        }
        yield_prediction = predict_crop_yield(data_for_prediction)
        return jsonify({'prediction': yield_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# --- NEW ENDPOINT for fertilizer prediction ---
@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer_endpoint():
    """Handles fertilizer recommendation requests."""
    if request.method == 'POST':
        try:
            # Get form data, ensuring correct types
            data_for_prediction = {
                'Temperature': float(request.form['Temperature']),
                'Humidity': float(request.form['Humidity']),
                'Moisture': float(request.form['Moisture']),
                'Soil Type': request.form['Soil_Type'],
                'Crop Type': request.form['Crop_Type'],
                'Nitrogen': int(request.form['Nitrogen']),
                'Potassium': int(request.form['Potassium']),
                'Phosphorous': int(request.form['Phosphorous'])
            }
            
            fertilizer_prediction = predict_fertilizer(data_for_prediction)
            
            return jsonify({'prediction': fertilizer_prediction})

        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
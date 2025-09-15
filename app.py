from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import database

from prediction_pipeline import predict_crop
from yield_prediction_pipeline import predict_crop_yield
from fertilizer_prediction_pipeline import predict_fertilizer

app = Flask(__name__)
database.init_db()

# --- Load data for dropdowns ---
YIELD_DATA_PATH = os.path.join('data', 'crop_production.csv')
FERTILIZER_DATA_PATH = os.path.join('data', 'Fertilizer Prediction.csv')

yield_df = pd.read_csv(YIELD_DATA_PATH).dropna(subset=["Production"])
fertilizer_df = pd.read_csv(FERTILIZER_DATA_PATH)

# Get unique values for yield dropdowns
unique_states = sorted(yield_df['State_Name'].unique())
unique_yield_seasons = sorted(yield_df['Season'].unique())
unique_yield_crops = sorted(yield_df['Crop'].unique())

# --- MODIFICATION: Get unique soil types for the yield form ---
unique_soil_types = sorted(fertilizer_df['Soil Type'].unique())


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html')

@app.route('/yield-predict')
def yield_predict():
    """Renders the yield prediction page."""
    # --- MODIFICATION: Pass unique_soil_types to the template ---
    return render_template(
        'yield.html',
        states=unique_states,
        seasons=unique_yield_seasons,
        crops=unique_yield_crops,
        soil_types=unique_soil_types
    )

@app.route('/get_districts/<state_name>')
def get_districts(state_name):
    districts = sorted(yield_df[yield_df['State_Name'] == state_name]['District_Name'].unique())
    return jsonify({'districts': districts})

@app.route('/fertilizer-recommend')
def fertilizer_recommend():
    return render_template(
        'fertilizer.html',
        soil_types=unique_soil_types,
        crop_types=unique_crop_types
    )
    
@app.route('/predict', methods=['POST'])
def predict():
    # ... (no changes needed in this function) ...
    try:
        data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        crop_prediction = predict_crop(data)
        database.log_crop_prediction(data, crop_prediction)
        return jsonify({'prediction': crop_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_yield', methods=['POST'])
def predict_yield_endpoint():
    """Handles yield prediction requests."""
    if request.method == 'POST':
        try:
            # --- MODIFICATION: Accept new fields but only use the ones the model needs ---
            # The form will send 'Rainfall' and 'Soil_Type', but we ignore them here
            # because the current yield model was not trained on them.
            data_for_prediction = {
                'Area': float(request.form['Area']),
                'State_Name': request.form['State_Name'],
                'District_Name': request.form['District_Name'],
                'Season': request.form['Season'],
                'Crop': request.form['Crop']
            }
            
            yield_prediction = predict_crop_yield(data_for_prediction)

            # Log the prediction with the data the model used
            database.log_yield_prediction(data_for_prediction, yield_prediction)
            
            return jsonify({'prediction': yield_prediction})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer_endpoint():
    # ... (no changes needed in this function) ...
    try:
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
        database.log_fertilizer_prediction(data_for_prediction, fertilizer_prediction)
        return jsonify({'prediction': fertilizer_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history')
def history():
    # ... (no changes needed in this function) ...
    crop_logs = database.fetch_crop_logs()
    yield_logs = database.fetch_yield_logs()
    fertilizer_logs = database.fetch_fertilizer_logs()
    return render_template(
        'history.html', 
        crop_history=crop_logs,
        yield_history=yield_logs,
        fertilizer_history=fertilizer_logs
    )

if __name__ == '__main__':
    app.run(debug=True)
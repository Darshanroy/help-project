import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np # Needed for sqrt
import pickle
import os
import json

# --- MODIFICATION: Updated file paths ---
DATA_PATH = os.path.join('data', 'crop_production.csv') 
MODEL_PATH = os.path.join('models', 'RandomForestRegressor.pkl')
FEATURE_COLUMNS_PATH = os.path.join('models', 'yield_feature_columns.pkl')
PERFORMANCE_FILE = os.path.join('models', 'model_performance.json')

def train_yield_model():
    print("Starting YIELD model training with hyperparameter tuning...")
    
    df = pd.read_csv(DATA_PATH).dropna(subset=["Production"])
    data_dum = pd.get_dummies(df.drop(["Crop_Year"], axis=1))

    x = data_dum.drop("Production", axis=1)
    y = data_dum["Production"]
    feature_columns = x.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_leaf': [2, 4]}
    
    rfr = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')
    
    print("Running GridSearchCV for Yield model...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    # --- MODIFICATION: Calculate detailed evaluation metrics ---
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Best Yield Model R² Score: {r2:.4f}")

    # Save model and columns
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_model, file)
    with open(FEATURE_COLUMNS_PATH, 'wb') as file:
        pickle.dump(feature_columns, file)

    # --- MODIFICATION: Save expanded performance metrics ---
    performance_data = {}
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, 'r') as f:
            performance_data = json.load(f)

    performance_data['yield_prediction'] = {
        'model_type': 'Random Forest Regressor',
        'r2_score': f"{r2:.4f}",
        'mean_absolute_error': f"{mae:.2f}",
        'mean_squared_error': f"{mse:.2f}",
        'root_mean_squared_error': f"{rmse:.2f}",
        'best_parameters': grid_search.best_params_,
        'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(performance_data, f, indent=4)
    print(f"Yield model performance saved to {PERFORMANCE_FILE}")

    return performance_data['yield_prediction']

if __name__ == '__main__':
    train_yield_model()
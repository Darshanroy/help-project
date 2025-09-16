import pandas as pd
import pickle
import os
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- MODIFICATION: Updated file paths ---
DATA_PATH = os.path.join('data', 'Fertilizer Prediction.csv')
MODEL_PATH = os.path.join('models', 'fertilizer_model.pkl')
ENCODERS_PATH = os.path.join('models', 'fertilizer_encoders.pkl')
PERFORMANCE_FILE = os.path.join('models', 'model_performance.json')

def train_fertilizer_model():
    print("Starting FERTILIZER model training with hyperparameter tuning...")
    
    data = pd.read_csv(DATA_PATH)
    data.rename(columns={"Humidity ": "Humidity"}, inplace=True)
    
    fertilizer_name_encoder = LabelEncoder()
    data["Fertilizer Name"] = fertilizer_name_encoder.fit_transform(data["Fertilizer Name"])

    X_categorical = pd.get_dummies(data[['Soil Type', 'Crop Type']])
    X_continuous = data.drop(['Fertilizer Name', 'Soil Type', 'Crop Type'], axis=1)
    X = pd.concat([X_continuous, X_categorical], axis=1)
    y = data["Fertilizer Name"]
    
    soil_crop_columns = X_categorical.columns.tolist()
    encoders = {'fertilizer_name': fertilizer_name_encoder, 'soil_crop_columns': soil_crop_columns}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = ImbPipeline(steps=[('smote', SMOTE(random_state=42)), ('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42))])
    
    param_grid = {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [10, 20]}
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    print("Running GridSearchCV for Fertilizer model...")
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_

    # --- MODIFICATION: Calculate detailed evaluation metrics ---
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Best Fertilizer Model Accuracy: {accuracy:.4f}")

    # Save pipeline and encoders
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_pipeline, file)
    with open(ENCODERS_PATH, 'wb') as file:
        pickle.dump(encoders, file)
    
    # --- MODIFICATION: Save expanded performance metrics ---
    performance_data = {}
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, 'r') as f:
            performance_data = json.load(f)
            
    performance_data['fertilizer_recommendation'] = {
        'model_type': 'Random Forest Classifier with SMOTE',
        'accuracy': f"{accuracy:.4f}",
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'best_parameters': grid_search.best_params_,
        'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(performance_data, f, indent=4)
    print(f"Fertilizer model performance saved to {PERFORMANCE_FILE}")

    return performance_data['fertilizer_recommendation']

if __name__ == '__main__':
    train_fertilizer_model()
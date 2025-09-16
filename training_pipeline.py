import pandas as pd
import pickle
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- MODIFICATION: Updated file paths ---
DATA_PATH = os.path.join('data', 'Crop_recommendation.csv')
MODEL_PATH = os.path.join('models', 'RandomForest.pkl')
PERFORMANCE_FILE = os.path.join('models', 'model_performance.json')

def train_crop_model():
    print("Starting CROP model training with hyperparameter tuning...")
    
    df = pd.read_csv(DATA_PATH)
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    print("Running GridSearchCV for Crop model...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # --- MODIFICATION: Calculate detailed evaluation metrics ---
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Best Crop Model Accuracy: {accuracy:.4f}")

    # Save the best model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_model, file)
    
    # --- MODIFICATION: Save expanded performance metrics ---
    performance_data = {}
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, 'r') as f:
            performance_data = json.load(f)
            
    performance_data['crop_recommendation'] = {
        'model_type': 'Random Forest Classifier',
        'accuracy': f"{accuracy:.4f}",
        'classification_report': report,
        'confusion_matrix': cm.tolist(), # convert numpy array to list for JSON
        'best_parameters': grid_search.best_params_,
        'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(performance_data, f, indent=4)
    print(f"Crop model performance saved to {PERFORMANCE_FILE}")
    
    return performance_data['crop_recommendation']

if __name__ == '__main__':
    train_crop_model()
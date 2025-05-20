from flask import Flask, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import time
import threading

app = Flask(__name__)

# Load the raw dataset
print("Loading raw dataset...")
start_time = time.time()
data_path = "./healthcare_data.csv"  # Path relative to flask_api.py
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Raw dataset at '{data_path}' not found. Please ensure 'healthcare_data.csv' is in the same directory as flask_api.py.")
data = pd.read_csv(data_path, low_memory=False)  # Address DtypeWarning
print(f"Dataset loading took {time.time() - start_time:.2f} seconds.")
print(f"Dataset loaded with {len(data)} rows and {len(data.columns)} columns.")

# Reduce dataset size for testing
print("Reducing dataset size for testing...")
data = data.head(10000)  # Use only the first 10,000 rows
print(f"Reduced dataset to {len(data)} rows.")

# Preprocess the data
print("Starting preprocessing...")
start_time = time.time()
try:
    data['START'] = pd.to_datetime(data['START'], errors='coerce')
    birth_dates = pd.to_datetime(data['BIRTHDATE'], errors='coerce')
    time_diff = (datetime.now() - birth_dates).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    data['AGE'] = time_diff.astype(float).round(0).astype(int)
except KeyError as e:
    print(f"Error processing data: {e}. Check column names.")
    data['AGE'] = 0

encounter_class_map = {
    'ambulatory': 'Ambulatory', 'outpatient': 'Outpatient', 'inpatient': 'Inpatient',
    'telehealth': 'Telehealth', 'home': 'Home', 'virtual': 'Virtual',
    'emergency': 'Emergency', 'other': 'Other', 'urgentcare': 'Urgent Care',
    'observation': 'Observation'
}
data['ENCOUNTERCLASS'] = data['ENCOUNTERCLASS'].map(encounter_class_map).fillna('Other')
print(f"Preprocessing took {time.time() - start_time:.2f} seconds.")

# Load the preprocessor and model from the same directory
print("Loading model and preprocessor...")
prep_path = "./preprocessor.pkl"  # Path relative to flask_api.py
model_path = "./xgboost_model.pkl"  # Path relative to flask_api.py

if not os.path.exists(prep_path):
    raise FileNotFoundError(f"Preprocessor file at '{prep_path}' not found. Please ensure 'preprocessor.pkl' is in the same directory as flask_api.py.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file at '{model_path}' not found. Please ensure 'xgboost_model.pkl' is in the same directory as flask_api.py.")

preprocessor = joblib.load(prep_path)
model = joblib.load(model_path)
print("Preprocessor and model loaded successfully.")

# Define columns used in training
numerical_cols = [
    'BASE_ENCOUNTER_COST', 'TOTAL_CLAIM_COST', 'PAYER_COVERAGE', 'LAT', 'LON',
    'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 'AGE'
]
categorical_cols = ['ENCOUNTERCLASS', 'MARITAL', 'RACE', 'ETHNICITY']

total_patients = len(data)
current_index = 0

@app.route('/new_patients', methods=['GET'])
def get_new_patients():
    global current_index
    batch_size = 10
    end_index = min(current_index + batch_size, total_patients)
    if end_index <= current_index:
        return jsonify({"error": "No more patients to serve"}), 400
    batch = data.iloc[current_index:end_index].copy()

    try:
        for col in numerical_cols + categorical_cols:
            if col not in batch.columns:
                batch[col] = np.nan
        for col in categorical_cols:
            batch[col] = batch[col].astype(str)
        for col in numerical_cols:
            batch[col] = pd.to_numeric(batch[col], errors='coerce')
        batch[numerical_cols] = batch[numerical_cols].fillna(batch[numerical_cols].mean())
        for col in categorical_cols:
            batch[col] = batch[col].fillna(batch[col].mode()[0])
        batch_transformed = preprocessor.transform(batch)
        risk_scores = model.predict_proba(batch_transformed)[:, 1]
        batch['readmission_risk'] = risk_scores
        batch['DATE'] = datetime.now().isoformat()
        new_patients = batch[['AGE', 'ENCOUNTERCLASS', 'readmission_risk', 'DATE']].to_dict(orient='records')
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    current_index = end_index
    if current_index >= total_patients:
        current_index = 0
    return jsonify(new_patients)

if __name__ == '__main__':
    print("Starting Flask app on port 5001 without reloader...")
    def run_flask():
        try:
            print("Binding to port 5001...")
            app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
            print("Flask app is running on http://0.0.0.0:5001")
        except Exception as e:
            print(f"Failed to start Flask app: {e}")

    # Run Flask in a background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask app running in background. Use http://127.0.0.1:5001/new_patients to test.")
    time.sleep(5)  # Wait briefly to ensure Flask starts
    print("Testing Flask server...")
    try:
        import requests
        response = requests.get('http://127.0.0.1:5001/new_patients', timeout=10)
        print("Flask server responded successfully:")
        print(response.json())
    except Exception as e:
        print(f"Failed to connect to Flask server: {e}")
    # Keep the script running
    flask_thread.join()
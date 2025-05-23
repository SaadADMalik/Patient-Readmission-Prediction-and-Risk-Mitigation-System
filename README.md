# Healthcare Readmission Prediction and Risk Mitigation
End-to-End Machine Learning Solution with Dash, Flask, and XGBoost
## Tech Stack

Python 3.11+  
Dash (Plotly)  
Flask  
XGBoost  
Pandas  
SQLite  
Synthea (for synthetic data generation)


## Project Overview
This project predicts patient readmission risks within 30 days using a 1-million-row synthetic dataset generated by Synthea. It integrates:  

- Data Preprocessing and EDA for insights  
- Time Series Analysis for temporal trends  
- An XGBoost model for risk prediction  
- A Dash dashboard for interactive visualizations  
- A Flask API for real-time updates

🔹 All data is synthetic, designed for demonstration and learning purposes.  

![REPORT](https://github.com/user-attachments/assets/f7025422-fa5e-4e1c-a21e-0e22913e3029)


## Business Challenge
Healthcare providers face:  

- High readmission rates driving up costs  
- Difficulty identifying high-risk patients early  
- Limited real-time data access

The Solution Offers:
- Automated risk predictions via machine learning
- Interactive dashboards for monitoring
- Real-time API updates for decision-making  

## Table of Contents

Dataset  
Installation  
Usage  
Data Loading and Cleaning  
Exploratory Data Analysis  
Time Series Analysis  
Model Building  
Dashboard  
Flask API  
Contributing  
License


## Dataset
The dataset includes 1,000,000 patient encounter records from Synthea:  

- Rows: 1M initially  
- Columns: 30+ (including engineered features)  
## Key Features:  
- AGE: Patient age (from birthdate)  
- ENCOUNTERCLASS: Encounter type (e.g., inpatient, outpatient)  
- TOTAL_CLAIM_COST: Encounter cost  
- READMISSION_30D: Target (1 if readmitted within 30 days, 0 otherwise)  
- ENCOUNTER_FREQ: Encounters per patient


## Storage: healthcare_data.csv (raw), healthcare_data_final.csv.gz (processed)


## Installation
Prerequisites

- Python 3.11+  
J- ava (for Synthea, optional)  
- Libraries: dash, flask, xgboost, pandas, sqlite3

## Steps

Clone the Repository  git clone https://github.com/your-username/healthcare-readmission-prediction.git  
cd healthcare-readmission-prediction  


## Install Dependencies  pip install -r requirements.txt  


## Prepare the Dataset  
- Place healthcare_data.csv in the root or generate it using Synthea (see notebook).  
- Preprocessed files like healthcare_data_final.csv.gz are created during execution.

## Model Files  
Ensure preprocessor.pkl and xgboost_model.pkl are in the root for the API.




## Usage

Run the Flask API  python flask_api.py  


Starts at http://127.0.0.1:5001


Launch the Dash Dashboard  python app.py  


Opens at http://127.0.0.1:8052


Interact with the Dashboard  
Filter by age group and encounter class  
View KPIs, charts, and high-risk patient tables  
Download data as CSV




## Data Loading and Cleaning

- Duplicates: Removed ~50K rows (based on PATIENT and START)  
- Missing Values: Filled non-critical fields (e.g., SSN) with 'Unknown'; dropped rows missing START/STOP  
- Data Types: Converted START, STOP, BIRTHDATE, DEATHDATE to datetime  
- Feature Engineering:  
- DURATION_HOURS: Encounter duration  
- DAYS_SINCE_LAST: Days since last encounter  
- READMISSION_30D: Refined for inpatient focus  
- ENCOUNTER_FREQ: Total encounters per patient


## Exploratory Data Analysis

- Age Distribution: Mean age ~50, most patients 40-60 (histogram)  
- Readmission Rates: 25% for emergency encounters (bar chart)  
- Correlations: TOTAL_CLAIM_COST vs. READMISSION_30D = 0.45 (heatmap)  
- Purpose: Guides feature selection and trend identification


## Time Series Analysis

- Monthly Rates: 15% increase in winter  
- Encounter Frequency: 20% rise in Q4  
- Age Trends: 61-80 age group at ~30% readmission rate  
- Purpose: Highlights seasonal patterns for visualization


## Model Building
An XGBoost classifier predicts READMISSION_30D:  

- Data Split: 400K train, 200K test  
- Features: AGE, ENCOUNTERCLASS, TOTAL_CLAIM_COST, etc.  
- Tuning: GridSearchCV (best: learning_rate=0.05, max_depth=5, n_estimators=200)  
### Performance:  
- AUC-ROC: 0.85  
- Precision: 0.70 (threshold 0.39)  
- Recall: 0.65

## Calibration: 
- Sigmoid-adjusted probabilities  
- Feature Importance: ENCOUNTERCLASS and AGE lead (SHAP analysis)


## Dashboard
The Dash app (app.py) offers:  

- Filters: Age group, encounter class  
- KPIs: High-risk count, average risk, total patients  
### Visualizations:  
- Scatter: Risk vs. Age  
- Bar: Risk by encounter class  
- Pie: High-risk by age  
- Trend: Risk over time  
- Heatmap: Risk by age and encounter  
- Histogram: Risk score distribution
- Table: Top 10 high-risk patients  
- Updates: Refreshes every 10s via API


## Flask API
The Flask API (flask_api.py) provides:  

- Endpoint: /new_patients  
- Functionality: Serves 10-patient batches with risks  
- Integration: Dash queries every 10s  
- Purpose: Simulates real-time data feeds


- Contributing
Contributions are welcome! Fork the repo and submit pull requests with enhancements or fixes.  



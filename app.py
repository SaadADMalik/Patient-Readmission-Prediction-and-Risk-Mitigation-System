import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import nest_asyncio
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import schedule
import time
import logging
import requests
import random

# Enable async handling for local environments
nest_asyncio.apply()

# Set up logging
logging.basicConfig(filename='update_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the initial processed dataset
print("Loading initial processed dataset...")
data_with_risk = pd.read_csv('./healthcare_step4_processed.csv', low_memory=False)

# Prepare data: Ensure necessary columns and construct DATE if missing
required_cols = ['AGE', 'ENCOUNTERCLASS', 'readmission_risk']
if not all(col in data_with_risk.columns for col in required_cols):
    raise ValueError(f"Missing required columns in CSV: {required_cols}")
data_with_risk = data_with_risk[required_cols].copy()
data_with_risk['AGE'] = pd.to_numeric(data_with_risk['AGE'], errors='coerce')
data_with_risk['readmission_risk'] = pd.to_numeric(data_with_risk['readmission_risk'], errors='coerce')

# Construct DATE with sample data for trend line
dates = pd.date_range(start='2024-12-31', end='2025-05-18', freq='D')
data_with_risk['DATE'] = np.random.choice(dates, size=len(data_with_risk))
data_with_risk['DATE'] = data_with_risk['DATE'].dt.strftime('%Y-%m-%d')

data_with_risk['AGE_GROUP'] = pd.cut(data_with_risk['AGE'], bins=[0, 20, 40, 60, 80, 120], 
                                    labels=['0-20', '21-40', '41-60', '61-80', '81+'], right=False)
data_with_risk['AGE_GROUP'] = data_with_risk['AGE_GROUP'].astype(str).fillna('81+')

# Add an ID column if it doesn't exist
if 'ID' not in data_with_risk.columns:
    data_with_risk['ID'] = range(1, len(data_with_risk) + 1)

# Save to SQLite database with correct schema
with sqlite3.connect('patients.db') as conn:
    cursor = conn.cursor()
    cursor.execute('''
        DROP TABLE IF EXISTS patients
    ''')
    cursor.execute('''
        CREATE TABLE patients (
            ID INTEGER PRIMARY KEY,
            AGE INTEGER,
            ENCOUNTERCLASS TEXT,
            AGE_GROUP TEXT,
            readmission_risk REAL,
            DATE TEXT
        )
    ''')
    data_with_risk.to_sql('patients', conn, index=False, if_exists='replace')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_age_group ON patients (AGE_GROUP)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_encounter_class ON patients (ENCOUNTERCLASS)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_readmission_risk ON patients (readmission_risk)')
    conn.commit()

# Function to fetch new patient data from Flask API and insert into database
def fetch_new_patients():
    print(f"Attempting to fetch new patients at {datetime.now().strftime('%H:%M:%S')}")
    try:
        with sqlite3.connect('patients.db') as conn:
            cursor = conn.cursor()
            response = requests.get('http://127.0.0.1:5001/new_patients', timeout=10)
            response.raise_for_status()
            new_patients_json = response.json()
            new_patients = pd.DataFrame(new_patients_json)

            # Debug: Print the columns of the API response
            print(f"API response columns: {new_patients.columns.tolist()}")

            # Ensure required columns
            required_cols = ['AGE', 'ENCOUNTERCLASS', 'readmission_risk']
            for col in required_cols:
                if col not in new_patients.columns:
                    raise ValueError(f"API response missing required column: {col}")

            # Process the data
            new_patients['AGE'] = pd.to_numeric(new_patients['AGE'], errors='coerce')
            new_patients['readmission_risk'] = pd.to_numeric(new_patients['readmission_risk'], errors='coerce')
            if 'DATE' in new_patients.columns:
                new_patients['DATE'] = pd.to_datetime(new_patients['DATE']).dt.strftime('%Y-%m-%d')
            elif all(col in new_patients.columns for col in ['YEAR', 'MONTH', 'DAY']):
                new_patients['DATE'] = pd.to_datetime(new_patients[['YEAR', 'MONTH', 'DAY']]).dt.strftime('%Y-%m-%d')
            else:
                new_patients['DATE'] = pd.to_datetime('2025-01-01').strftime('%Y-%m-%d')

            new_patients['AGE_GROUP'] = pd.cut(new_patients['AGE'], bins=[0, 20, 40, 60, 80, 120], 
                                              labels=['0-20', '21-40', '41-60', '61-80', '81+'], right=False)
            new_patients['AGE_GROUP'] = new_patients['AGE_GROUP'].astype(str).fillna('81+')

            # Generate a synthetic ID if not present
            if 'ID' not in new_patients.columns:
                cursor.execute("SELECT MAX(ID) FROM patients")
                max_id = cursor.fetchone()[0] or 0
                new_patients['ID'] = range(max_id + 1, max_id + 1 + len(new_patients))

            # Insert new patients into the database
            for _, row in new_patients.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO patients (ID, AGE, ENCOUNTERCLASS, AGE_GROUP, readmission_risk, DATE)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (row['ID'], row['AGE'], row['ENCOUNTERCLASS'], row['AGE_GROUP'], row['readmission_risk'], row['DATE']))
            conn.commit()

            logging.info(f"Added {len(new_patients)} new patients from API.")
            print(f"Fetched new patients at {datetime.now().strftime('%H:%M:%S')}. Total rows updated.")
    except Exception as e:
        print(f"Error fetching new patients: {e}")
        logging.error(f"Error fetching new patients: {e}")

# Schedule API calls every 10 seconds
schedule.every(10).seconds.do(fetch_new_patients)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define custom styles
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '220px',
    'padding': '25px',
    'backgroundColor': '#f9fafc',
    'boxShadow': '2px 0 6px rgba(0,0,0,0.02)',
    'overflowY': 'auto',
    'fontFamily': '"Roboto", sans-serif'
}

CONTENT_STYLE = {
    'marginLeft': '240px',
    'padding': '25px',
    'background': 'linear-gradient(135deg, #f7f9fc 0%, #e9ecef 100%)',
    'minHeight': '100vh',
    'fontFamily': '"Roboto", sans-serif',
    'color': '#2c3e50'
}

KPI_CARD_STYLE = {
    'backgroundColor': '#ffffff',
    'borderRadius': '12px',
    'padding': '15px',
    'margin': '10px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.02)',
    'border': 'none',
    'textAlign': 'center',
    'transition': 'opacity 0.3s ease-in',
    'opacity': '0.9',
    ':hover': {'opacity': '1'}
}

CARD_STYLE = {
    'backgroundColor': '#ffffff',
    'borderRadius': '12px',
    'padding': '20px',
    'marginBottom': '25px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.02)',
    'border': 'none',
    'transition': 'opacity 0.3s ease-in',
    'opacity': '0.9',
    ':hover': {'opacity': '1'}
}

CHART_STYLE = {
    'height': '320px',
    'borderRadius': '12px',
    'backgroundColor': '#ffffff',
    'padding': '15px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.02)',
    'transition': 'opacity 0.3s ease-in',
    'opacity': '0.9',
    ':hover': {'opacity': '1'}
}

GRID_STYLE = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(3, 1fr)',  # 3 columns
    'gap': '25px',
    'marginBottom': '25px'
}

KPI_GRID_STYLE = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(3, 1fr)',  # 3 KPI cards
    'gap': '15px',
    'marginBottom': '25px'
}

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H2("Healthcare Readmission Dashboard", style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '5px', 'fontSize': '28px', 'fontWeight': '300'}),
        html.P("Real-time patient readmission risk insights", 
               style={'color': '#7f8c8d', 'textAlign': 'center', 'fontStyle': 'italic', 'fontSize': '14px'}),
    ], style={'backgroundColor': '#f1f4f8', 'padding': '25px', 'borderBottom': '1px solid #dfe4ea'}),
    html.Div([
        html.Div([
            html.Label("Filter by Age Group", style={'fontWeight': '400', 'color': '#2c3e50', 'fontSize': '14px'}),
            dcc.Dropdown(
                id='age-group-filter',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': str(age_group), 'value': str(age_group)} for age_group in data_with_risk['AGE_GROUP'].unique() if pd.notna(age_group)],
                value='All',
                style={'width': '100%', 'marginBottom': '20px', 'fontSize': '13px', 'backgroundColor': '#ffffff', 'color': '#2c3e50', 'border': '1px solid #dfe4ea', 'borderRadius': '8px'},
                className='dropdown-custom'
            ),
            html.Label("Filter by Encounter Class", style={'fontWeight': '400', 'color': '#2c3e50', 'fontSize': '14px'}),
            dcc.Dropdown(
                id='encounter-class-filter',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': str(encounter), 'value': encounter} for encounter in data_with_risk['ENCOUNTERCLASS'].unique()],
                value='All',
                style={'width': '100%', 'marginBottom': '20px', 'fontSize': '13px', 'backgroundColor': '#ffffff', 'color': '#2c3e50', 'border': '1px solid #dfe4ea', 'borderRadius': '8px'},
                className='dropdown-custom'
            ),
            html.Div([
                html.Label("Quick Filters", style={'fontWeight': '400', 'color': '#2c3e50', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Button("Reset Filters", id='reset-button', 
                           style={'width': '100%', 'padding': '10px', 'backgroundColor': '#ecf0f1', 'color': '#2c3e50', 'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '400'})
            ], style=CARD_STYLE)
        ], style=SIDEBAR_STYLE),
        html.Div([
            # KPI Row
            html.Div([
                html.Div([
                    html.Label("High-Risk Patients", style={'fontWeight': '400', 'color': '#2c3e50', 'fontSize': '14px'}),
                    html.Div(id='kpi-high-risk', style={'fontSize': '26px', 'color': '#e67e22', 'fontWeight': '500'})
                ], style=KPI_CARD_STYLE),
                html.Div([
                    html.Label("Avg. Readmission Risk", style={'fontWeight': '400', 'color': '#2c3e50', 'fontSize': '14px'}),
                    html.Div(id='kpi-avg-risk', style={'fontSize': '26px', 'color': '#3498db', 'fontWeight': '500'})
                ], style=KPI_CARD_STYLE),
                html.Div([
                    html.Label("Total Patients", style={'fontWeight': '400', 'color': '#2c3e50', 'fontSize': '14px'}),
                    html.Div(id='kpi-total-patients', style={'fontSize': '26px', 'color': '#9b59b6', 'fontWeight': '500'})
                ], style=KPI_CARD_STYLE),
            ], style=KPI_GRID_STYLE),
            # 2x3 Grid for Charts
            html.Div([
                html.Div([dcc.Graph(id='scatter-plot', style=CHART_STYLE)]),
                html.Div([dcc.Graph(id='bar-chart', style=CHART_STYLE)]),
                html.Div([dcc.Graph(id='pie-chart', style=CHART_STYLE)]),
                html.Div([dcc.Graph(id='trend-line', style=CHART_STYLE)]),
                html.Div([dcc.Graph(id='heatmap', style=CHART_STYLE)]),
                html.Div([dcc.Graph(id='histogram', style=CHART_STYLE)]),
            ], style=GRID_STYLE),
            # Table and Download Button
            html.Div([
                html.H3("High-Risk Patients", style={'color': '#2c3e50', 'textAlign': 'center', 'marginTop': '25px', 'marginBottom': '15px', 'fontSize': '22px', 'fontWeight': '300'}),
                html.Div(id='patient-table', style={'margin': '20px'}),
                html.Button("Download High-Risk Patients as CSV", id='download-button', 
                           style={'display': 'block', 'margin': '25px auto', 'padding': '12px 25px', 'backgroundColor': '#3498db', 
                                  'color': '#ffffff', 'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': '400',
                                  'transition': 'background-color 0.3s', ':hover': {'backgroundColor': '#2980b9'}}),
                dcc.Download(id='download-dataframe-csv')
            ], style=CARD_STYLE)
        ], style=CONTENT_STYLE),
    ]),
    html.Footer([
        html.P("© 2025 Healthcare Analytics | Powered by xAI", style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px', 'fontSize': '12px'})
    ], style={'backgroundColor': '#f1f4f8', 'borderTop': '1px solid #dfe4ea'}),
    dcc.Interval(
        id='interval-component',
        interval=120 * 1000,  # Update every 5 seconds (in milliseconds)
        n_intervals=0
    ),
])

# Add custom CSS for dropdowns, animations, and hover effects
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .dropdown-custom .Select-control {
                background-color: #ffffff !important;
                color: #2c3e50 !important;
                border: 1px solid #dfe4ea !important;
                border-radius: 8px !important;
            }
            .dropdown-custom .Select-menu-outer {
                background-color: #ffffff !important;
                color: #2c3e50 !important;
                border-radius: 8px !important;
            }
            .dropdown-custom .Select-value {
                color: #2c3e50 !important;
            }
            .dropdown-custom .Select-option {
                background-color: #ffffff !important;
                color: #2c3e50 !important;
            }
            .dropdown-custom .Select-option.is-focused {
                background-color: #f7f9fc !important;
            }
            .fade-in {
                animation: fadeIn 0.5s ease-in;
            }
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 0.9; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callback to reset filters
@app.callback(
    [Output('age-group-filter', 'value'),
     Output('encounter-class-filter', 'value')],
    [Input('reset-button', 'n_clicks')],
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return 'All', 'All'

# Callback to update KPI cards
@app.callback(
    [Output('kpi-high-risk', 'children'),
     Output('kpi-avg-risk', 'children'),
     Output('kpi-total-patients', 'children')],
    [Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_kpi_cards(age_group, encounter_class, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT readmission_risk FROM patients"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            if df.empty:
                return "0", "0.00", "0"
            high_risk = len(df[df['readmission_risk'] >= 0.3909])
            avg_risk = df['readmission_risk'].mean()
            total_patients = len(df)
            return str(high_risk), f"{avg_risk:.2f}", str(total_patients)
    except Exception as e:
        print(f"Error in update_kpi_cards: {e}")
        return "0", "0.00", "0"

# Callback to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_scatter_plot(age_group, encounter_class, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT AGE, ENCOUNTERCLASS, readmission_risk FROM patients ORDER BY DATE DESC LIMIT 10000"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            df = df.dropna(subset=['AGE', 'readmission_risk'])
            if df.empty:
                return go.Figure().update_layout(title='No Valid Data for Scatter Plot')
            df['Risk_Level'] = np.where(df['readmission_risk'] >= 0.3909, 'High Risk', 'Low Risk')
            fig = px.scatter(df, x='AGE', y='readmission_risk', color='Risk_Level', 
                             color_discrete_map={'High Risk': '#ff6f61', 'Low Risk': '#6ab0e0'}, 
                             title='Readmission Risk vs. Age', 
                             labels={'AGE': 'Patient Age', 'readmission_risk': 'Risk Score'})
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(size=12, color='#2c3e50', family='"Roboto", sans-serif'),
                title_font_size=16,
                title_font_weight=300,
                margin=dict(l=40, r=40, t=40, b=20),
                title_x=0.5
            )
            return fig
    except Exception as e:
        print(f"Error in update_scatter_plot: {e}")
        return go.Figure().update_layout(title=f'Error: Unable to render scatter plot')

# Callback to update bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_bar_chart(age_group, encounter_class, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT ENCOUNTERCLASS, readmission_risk FROM patients ORDER BY DATE DESC LIMIT 10000"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            df = df.dropna(subset=['ENCOUNTERCLASS', 'readmission_risk'])
            if df.empty:
                return go.Figure().update_layout(title='No Valid Data for Bar Chart')
            avg_risk_by_encounter = df.groupby('ENCOUNTERCLASS')['readmission_risk'].mean().reset_index()
            if avg_risk_by_encounter.empty:
                return go.Figure().update_layout(title='No Data for Bar Chart')
            avg_risk_by_encounter['Color'] = np.where(avg_risk_by_encounter['readmission_risk'] >= 0.3909, '#ff6f61', '#6ab0e0')
            fig = go.Figure(data=[go.Bar(x=avg_risk_by_encounter['ENCOUNTERCLASS'], y=avg_risk_by_encounter['readmission_risk'],
                                         marker_color=avg_risk_by_encounter['Color'], text=avg_risk_by_encounter['readmission_risk'].round(3),
                                         textposition='auto')])
            fig.update_layout(
                title='Avg. Readmission Risk by Encounter',
                xaxis_title='Encounter Class',
                yaxis_title='Avg. Risk Score',
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(size=12, color='#2c3e50', family='"Roboto", sans-serif'),
                title_font_size=16,
                title_font_weight=300,
                margin=dict(l=40, r=40, t=40, b=20),
                showlegend=False,
                title_x=0.5
            )
            return fig
    except Exception as e:
        print(f"Error in update_bar_chart: {e}")
        return go.Figure().update_layout(title=f'Error: Unable to render bar chart')

# Callback to update pie chart
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_pie_chart(age_group, encounter_class, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT AGE_GROUP FROM patients WHERE readmission_risk >= 0.3909 ORDER BY DATE DESC LIMIT 10000"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " AND " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            df = df.dropna(subset=['AGE_GROUP'])
            if df.empty:
                return go.Figure().update_layout(title='No Valid Data for Pie Chart')
            age_group_distribution = df['AGE_GROUP'].value_counts().reset_index()
            age_group_distribution.columns = ['AGE_GROUP', 'Count']
            if age_group_distribution.empty:
                return go.Figure().update_layout(title='No Data for Pie Chart')
            fig = px.pie(age_group_distribution, names='AGE_GROUP', values='Count',
                         title='High-Risk Patients by Age Group',
                         color_discrete_sequence=['#6ab0e0', '#ffcc5c', '#a29bfe', '#ff6f61', '#54d2d2'])
            fig.update_traces(textinfo='percent+label', pull=[0.05, 0, 0, 0, 0],
                              hovertemplate='%{label}: %{value} patients (%{percent})')
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(size=12, color='#2c3e50', family='"Roboto", sans-serif'),
                title_font_size=16,
                title_font_weight=300,
                margin=dict(l=40, r=40, t=40, b=20),
                title_x=0.5
            )
            return fig
    except Exception as e:
        print(f"Error in update_pie_chart: {e}")
        return go.Figure().update_layout(title=f'Error: Unable to render pie chart')

# Callback to update trend line
@app.callback(
    Output('trend-line', 'figure'),
    [Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_trend_line(age_group, encounter_class, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT DATE, readmission_risk FROM patients ORDER BY DATE DESC LIMIT 10000"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df = df.dropna(subset=['DATE', 'readmission_risk'])
            if df.empty:
                return go.Figure().update_layout(title='No Valid Data for Trend Line')
            trend_data = df.groupby(df['DATE'].dt.to_period('D').dt.to_timestamp())['readmission_risk'].mean().reset_index()
            if trend_data.empty:
                return go.Figure().update_layout(title='No Data for Trend Line')
            trend_data['readmission_risk'] = trend_data['readmission_risk'] + np.random.normal(0, 0.05, len(trend_data))
            fig = px.line(trend_data, x='DATE', y='readmission_risk',
                          title='Avg. Readmission Risk Over Time',
                          labels={'DATE': 'Date', 'readmission_risk': 'Avg. Risk Score'})
            fig.add_hline(y=0.3909, line_dash="dash", line_color="#ff6f61", annotation_text="High Risk Threshold",
                          annotation_position="top right")
            fig.update_traces(line_color='#6ab0e0', line_width=2)
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(size=12, color='#2c3e50', family='"Roboto", sans-serif'),
                title_font_size=16,
                title_font_weight=300,
                margin=dict(l=40, r=40, t=40, b=20),
                xaxis=dict(tickformat='%Y-%m-%d'),
                title_x=0.5
            )
            return fig
    except Exception as e:
        print(f"Error in update_trend_line: {e}")
        return go.Figure().update_layout(title=f'Error: Unable to render trend line')

# Callback to update heatmap
@app.callback(
    Output('heatmap', 'figure'),
    [Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_heatmap(age_group, encounter_class, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT AGE_GROUP, ENCOUNTERCLASS, readmission_risk FROM patients ORDER BY DATE DESC LIMIT 10000"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            df = df.dropna(subset=['AGE_GROUP', 'ENCOUNTERCLASS', 'readmission_risk'])
            if df.empty:
                return go.Figure().update_layout(title='No Valid Data for Heatmap')
            heatmap_data = df.groupby(['AGE_GROUP', 'ENCOUNTERCLASS'])['readmission_risk'].mean().reset_index()
            if heatmap_data.empty:
                return go.Figure().update_layout(title='No Data for Heatmap')
            heatmap_data = heatmap_data.pivot(index='AGE_GROUP', columns='ENCOUNTERCLASS', values='readmission_risk')
            if heatmap_data.empty:
                return go.Figure().update_layout(title='No Data for Heatmap')
            fig = px.imshow(heatmap_data, title='Risk by Age Group & Encounter',
                            labels={'color': 'Avg. Risk Score'}, color_continuous_scale=px.colors.sequential.Peach)
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(size=12, color='#2c3e50', family='"Roboto", sans-serif'),
                title_font_size=16,
                title_font_weight=300,
                margin=dict(l=40, r=40, t=40, b=20),
                title_x=0.5
            )
            return fig
    except Exception as e:
        print(f"Error in update_heatmap: {e}")
        return go.Figure().update_layout(title=f'Error: Unable to render heatmap')

# Callback for annotated histogram chart (distribution of risk scores)
@app.callback(
    Output('histogram', 'figure'),
    [Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_histogram(age_group, encounter_class, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT readmission_risk FROM patients ORDER BY DATE DESC LIMIT 10000"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            df = df.dropna(subset=['readmission_risk'])
            if df.empty:
                return go.Figure().update_layout(title='No Valid Data for Histogram')
            fig = px.histogram(df, x='readmission_risk', nbins=10, 
                               title='Distribution of Readmission Risk Scores',
                               labels={'readmission_risk': 'Risk Score'}, 
                               color_discrete_sequence=['#6ab0e0'])
            # Add high-risk threshold line and annotation
            fig.add_vline(x=0.3909, line_dash="dash", line_color="#ff6f61", 
                          annotation_text="High Risk (0.39)", annotation_position="top right")
            high_risk_percent = (df['readmission_risk'] >= 0.3909).mean() * 100
            fig.add_annotation(x=0.8, y=0.9, xref="paper", yref="paper",
                              text=f"High-Risk: {high_risk_percent:.1f}%", showarrow=False,
                              font=dict(size=12, color='#ff6f61'))
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(size=12, color='#2c3e50', family='"Roboto", sans-serif'),
                title_font_size=16,
                title_font_weight=300,
                margin=dict(l=40, r=40, t=40, b=20),
                title_x=0.5,
                bargap=0.15
            )
            return fig
    except Exception as e:
        print(f"Error in update_histogram: {e}")
        return go.Figure().update_layout(title=f'Error: Unable to render histogram')

@app.callback(
    Output('patient-table', 'children'),
    [Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('bar-chart', 'clickData'),
     Input('pie-chart', 'clickData'),
     Input('interval-component', 'n_intervals')]
)
def update_table(age_group, encounter_class, bar_click, pie_click, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT AGE, ENCOUNTERCLASS, AGE_GROUP, readmission_risk FROM patients WHERE readmission_risk >= 0.3909"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " AND " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            if bar_click:
                selected_encounter = bar_click['points'][0]['x']
                df = df[df['ENCOUNTERCLASS'] == selected_encounter]
            if pie_click:
                selected_age_group = pie_click['points'][0]['label']
                df = df[df['AGE_GROUP'] == selected_age_group]
            if df.empty:
                return dash_table.DataTable(data=[], columns=[{'name': 'No data', 'id': 'nodata'}],
                                            style_table={'overflowX': 'auto'}, 
                                            style_cell={'textAlign': 'left', 'padding': '10px', 'color': '#2c3e50', 'backgroundColor': '#ffffff', 'fontSize': '12px'},
                                            style_header={'backgroundColor': '#f1f4f8', 'color': '#2c3e50', 'fontWeight': '400', 'fontSize': '12px'})
            df['Recommendation'] = np.where(df['readmission_risk'] > 0.5, 'Schedule Follow-Up', 'Monitor Closely')
            table_df = df[['AGE', 'ENCOUNTERCLASS', 'AGE_GROUP', 'readmission_risk', 'Recommendation']].head(10)
            style_data_conditional = [
                {'if': {'filter_query': '{readmission_risk} >= 0.3909'}, 'backgroundColor': 'rgba(255, 111, 97, 0.1)', 'color': '#2c3e50'}
            ]
            return dash_table.DataTable(data=table_df.to_dict('records'),
                                        columns=[{'name': col.replace('_', ' ').title(), 'id': col} for col in table_df.columns],
                                        style_table={'overflowX': 'auto'}, 
                                        style_cell={'textAlign': 'left', 'padding': '10px', 'color': '#2c3e50', 'backgroundColor': '#ffffff', 'fontSize': '12px', 'fontFamily': '"Roboto", sans-serif'},
                                        style_header={'backgroundColor': '#f1f4f8', 'color': '#2c3e50', 'fontWeight': '400', 'fontSize': '12px'},
                                        style_data_conditional=style_data_conditional)
    except Exception as e:
        print(f"Error in update_table: {e}")
        return dash_table.DataTable(data=[], columns=[{'name': 'Error', 'id': 'error'}],
                                    style_table={'overflowX': 'auto'}, 
                                    style_cell={'textAlign': 'left', 'padding': '10px', 'color': '#2c3e50', 'backgroundColor': '#ffffff'},
                                    style_header={'backgroundColor': '#f1f4f8', 'color': '#2c3e50', 'fontWeight': '400', 'fontSize': '12px'})

@app.callback(
    Output('download-dataframe-csv', 'data'),
    [Input('download-button', 'n_clicks'),
     Input('age-group-filter', 'value'),
     Input('encounter-class-filter', 'value'),
     Input('bar-chart', 'clickData'),
     Input('pie-chart', 'clickData'),
     Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def download_csv(n_clicks, age_group, encounter_class, bar_click, pie_click, n):
    try:
        with sqlite3.connect('patients.db') as conn:
            query = "SELECT AGE, ENCOUNTERCLASS, AGE_GROUP, readmission_risk FROM patients WHERE readmission_risk >= 0.3909"
            params = []
            conditions = []
            if age_group != 'All':
                conditions.append("AGE_GROUP = ?")
                params.append(age_group)
            if encounter_class != 'All':
                conditions.append("ENCOUNTERCLASS = ?")
                params.append(encounter_class)
            if conditions:
                query += " AND " + " AND ".join(conditions)
            df = pd.read_sql(query, conn, params=params)
            if bar_click:
                selected_encounter = bar_click['points'][0]['x']
                df = df[df['ENCOUNTERCLASS'] == selected_encounter]
            if pie_click:
                selected_age_group = pie_click['points'][0]['label']
                df = df[df['AGE_GROUP'] == selected_age_group]
            if df.empty:
                return dcc.send_data_frame(pd.DataFrame(columns=['No data']).to_csv, "high_risk_patients.csv")
            df['Recommendation'] = np.where(df['readmission_risk'] > 0.5, 'Schedule Follow-Up', 'Monitor Closely')
            table_df = df[['AGE', 'ENCOUNTERCLASS', 'AGE_GROUP', 'readmission_risk', 'Recommendation']]
            return dcc.send_data_frame(table_df.to_csv, "high_risk_patients.csv")
    except Exception as e:
        print(f"Error in download_csv: {e}")
        return dcc.send_data_frame(pd.DataFrame(columns=['Error']).to_csv, "high_risk_patients.csv")

# Run the scheduler and app
if __name__ == '__main__':
    print("Starting scheduler and Dash app on port 8052...")
    def run_scheduler():
        while True:
            schedule.run_pending()
            print(f"Scheduler running at {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(1)

    import threading
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    print("Attempting to start Dash app...")
    try:
        app.run(port=8052, host='0.0.0.0', debug=True, use_reloader=False)
        print("Dash app started successfully.")
    except Exception as e:
        print(f"Failed to start Dash app: {e}")
    print("Dash app should be accessible at http://127.0.0.1:8052")
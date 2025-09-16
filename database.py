import sqlite3
import datetime
import os


# Define the database file name
DATABASE_FILE = os.path.join('database','harvest_helper.db')

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        
        # --- Table for Crop Recommendation Logs ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crop_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                nitrogen REAL,
                phosphorous REAL,
                potassium REAL,
                temperature REAL,
                humidity REAL,
                ph REAL,
                rainfall REAL,
                predicted_crop TEXT
            )
        ''')

        # --- Table for Yield Prediction Logs ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS yield_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                area REAL,
                state_name TEXT,
                district_name TEXT,
                season TEXT,
                crop TEXT,
                predicted_yield REAL
            )
        ''')

        # --- Table for Fertilizer Recommendation Logs ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fertilizer_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                temperature REAL,
                humidity REAL,
                moisture REAL,
                soil_type TEXT,
                crop_type TEXT,
                nitrogen INTEGER,
                potassium INTEGER,
                phosphorous INTEGER,
                predicted_fertilizer TEXT
            )
        ''')
        
        conn.commit()
    print("Database initialized successfully.")

def log_crop_prediction(inputs, prediction):
    """Logs the inputs and output of a crop recommendation."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        query = '''
            INSERT INTO crop_logs (nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, predicted_crop)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        # Combine inputs and the single prediction into one tuple
        log_data = tuple(inputs) + (prediction,)
        cursor.execute(query, log_data)
        conn.commit()

def log_yield_prediction(inputs, prediction):
    """Logs the inputs and output of a yield prediction."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        query = '''
            INSERT INTO yield_logs (area, state_name, district_name, season, crop, predicted_yield)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        # The order here must match the dictionary keys used in app.py
        log_data = (
            inputs['Area'], inputs['State_Name'], inputs['District_Name'],
            inputs['Season'], inputs['Crop'], prediction
        )
        cursor.execute(query, log_data)
        conn.commit()
        
def log_fertilizer_prediction(inputs, prediction):
    """Logs the inputs and output of a fertilizer recommendation."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        query = '''
            INSERT INTO fertilizer_logs (temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous, predicted_fertilizer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        log_data = (
            inputs['Temperature'], inputs['Humidity'], inputs['Moisture'],
            inputs['Soil Type'], inputs['Crop Type'], inputs['Nitrogen'],
            inputs['Potassium'], inputs['Phosphorous'], prediction
        )
        cursor.execute(query, log_data)
        conn.commit()

def fetch_crop_logs():
    """Fetches all logs from the crop_logs table."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        # This makes the cursor return dictionaries, which are easier to work with in templates
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM crop_logs ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        return [dict(row) for row in rows] # Convert sqlite3.Row objects to standard dicts

def fetch_yield_logs():
    """Fetches all logs from the yield_logs table."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM yield_logs ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def fetch_fertilizer_logs():
    """Fetches all logs from the fertilizer_logs table."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM fertilizer_logs ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_log_entry(log_type, log_id):
    """Deletes a single log entry from a specified table by its ID."""
    # Whitelist of table names to prevent SQL injection
    table_map = {
        'crop': 'crop_logs',
        'yield': 'yield_logs',
        'fertilizer': 'fertilizer_logs'
    }
    if log_type not in table_map:
        raise ValueError("Invalid log type specified.")
    
    table_name = table_map[log_type]

    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        # Use a parameterized query for safety
        cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (log_id,))
        conn.commit()
    print(f"Deleted log entry with ID {log_id} from {table_name}.")

def delete_all_logs_for_type(log_type):
    """Deletes all log entries from a specified table."""
    table_map = {
        'crop': 'crop_logs',
        'yield': 'yield_logs',
        'fertilizer': 'fertilizer_logs'
    }
    if log_type not in table_map:
        raise ValueError("Invalid log type specified.")
        
    table_name = table_map[log_type]

    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        conn.commit()
    print(f"Cleared all entries from {table_name}.")
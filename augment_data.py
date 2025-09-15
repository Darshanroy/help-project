import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Define the paths to your data files
YIELD_DATA_FILE = os.path.join('data', 'crop_production.csv')
FERTILIZER_DATA_FILE = os.path.join('data', 'Fertilizer Prediction.csv')
OUTPUT_FILE = os.path.join('data', 'crop_production.csv')

def augment_yield_data():
    """
    Adds 'Rainfall' and 'Soil Type' columns with random data to the
    crop production dataset and saves it as a new file.
    """
    print(f"Loading yield data from: {YIELD_DATA_FILE}")
    try:
        df_yield = pd.read_csv(YIELD_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {YIELD_DATA_FILE} not found. Please ensure the file is in the 'data' directory.")
        return

    print(f"Loading fertilizer data for soil types from: {FERTILIZER_DATA_FILE}")
    try:
        df_fertilizer = pd.read_csv(FERTILIZER_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {FERTILIZER_DATA_FILE} not found. This is needed for soil type categories.")
        return

    # Check if columns already exist
    if 'Rainfall' in df_yield.columns and 'Soil Type' in df_yield.columns:
        print("Columns 'Rainfall' and 'Soil Type' already exist. No changes made.")
        print(f"If you want to regenerate them, please use the existing file: {YIELD_DATA_FILE}")
        return

    num_rows = len(df_yield)
    print(f"Dataset has {num_rows} rows. Generating new data...")

    # 1. Generate random Rainfall data (numeric, between 18 and 43)
    print("Generating 'Rainfall' data...")
    rainfall_data = np.random.uniform(1, 433.0, num_rows)
    df_yield['Rainfall'] = np.round(rainfall_data, 2) # Round to 2 decimal places

    # 2. Generate random Soil Type data (categorical)
    print("Generating 'Soil Type' data...")
    # Get the unique soil types from the fertilizer dataset
    soil_type_categories = df_fertilizer['Soil Type'].unique()
    # Randomly assign a soil type to each row in the yield dataset
    soil_type_data = np.random.choice(soil_type_categories, num_rows)
    df_yield['Soil Type'] = soil_type_data
    
    # 3. Save the augmented DataFrame to a new CSV file
    try:
        df_yield.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccessfully created new augmented file at: {OUTPUT_FILE}")
        print("\nFirst 5 rows of the new dataset:")
        print(df_yield.head())
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")

if __name__ == '__main__':
    augment_yield_data()
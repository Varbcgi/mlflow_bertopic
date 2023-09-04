import pandas as pd
import os
import glob

def ingest_data(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    return df

def retrain_ingest_data(file_path):
    # Get the folder path where the CSV files are located
    #folder_path = os.path.dirname(file_path)

    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(file_path, '*.csv'))

    # Check if any CSV file exists
    if not csv_files:
        raise ValueError("No CSV files found in the specified folder.")

    # Sort the CSV files by modification time in descending order
    csv_files.sort(key=os.path.getmtime, reverse=True)

    # Read the latest CSV file into a pandas DataFrame
    latest_csv_file = csv_files[0]
    print(latest_csv_file)
    df = pd.read_csv(latest_csv_file)

    return df
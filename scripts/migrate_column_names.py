import pandas as pd
import os
from pathlib import Path
from src.config import RAW_DATA_DIR
from src.utils.naming import standardize_dataframe_columns

def migrate_column_names():
    """One-time migration to standardize all column names in data files."""
    print("Starting column name migration...")
    
    # Get all CSV files in raw data directory
    data_dir = Path(RAW_DATA_DIR)
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in data directory")
        return
    
    for file_path in csv_files:
        print(f"\nProcessing {file_path.name}...")
        
        # Load file
        df = pd.read_csv(file_path)
        
        # Show original columns
        print("Original columns:", df.columns.tolist())
        
        # Standardize columns
        df = standardize_dataframe_columns(df)
        
        # Show new columns
        print("Standardized columns:", df.columns.tolist())
        
        # Save back to file
        df.to_csv(file_path, index=False)
        print(f"Updated {file_path.name}")
    
    print("\nMigration complete!")

if __name__ == "__main__":
    migrate_column_names() 
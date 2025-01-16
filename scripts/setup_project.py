import os
import sys
from pathlib import Path
import shutil

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

def setup_directories():
    """Create project directory structure."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def move_existing_data():
    """Move existing data files to the correct location."""
    # Define source files from scrape_ufc_stats-main
    source_files = [
        'ufc_event_details.csv',
        'ufc_fighter_details.csv',
        'ufc_fighter_tott.csv',
        'ufc_fight_details.csv',
        'ufc_fight_results.csv',
        'ufc_fight_stats.csv'
    ]
    
    # Define target names in raw directory
    target_names = [
        'events.csv',
        'fighter_details.csv',
        'fighter_tott.csv',
        'fight_details.csv',
        'fight_results.csv',
        'fight_stats.csv'
    ]
    
    # Move and rename files
    for source, target in zip(source_files, target_names):
        source_path = os.path.join(project_root, 'scrape_ufc_stats-main', source)
        target_path = os.path.join(RAW_DATA_DIR, target)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"Moved {source} to {target_path}")

def main():
    setup_directories()
    move_existing_data()
    print("Project setup complete!")

if __name__ == "__main__":
    main() 
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import pandas as pd
from src.config import DATA_DIR, RAW_DATA_DIR
from src.data.scraping.event_scraper import EventScraper
from src.data.scraping.fight_scraper import FightScraper
from src.data.scraping.fighter_scraper import FighterScraper

def main():
    # Initialize scrapers
    event_scraper = EventScraper('http://ufcstats.com/statistics/events/completed?page=all')
    fight_scraper = FightScraper()
    fighter_scraper = FighterScraper()
    
    # Get all events
    events_df = event_scraper.get_all_events()
    
    # Get fight data for each event
    all_fight_details = []
    all_fight_stats = []
    
    for _, event in events_df.iterrows():
        details, stats = fight_scraper.get_fight_data(event['URL'])
        all_fight_details.append(details)
        all_fight_stats.append(stats)
    
    # Save data
    events_df.to_csv(os.path.join(RAW_DATA_DIR, 'events.csv'), index=False)
    pd.concat(all_fight_details).to_csv(os.path.join(RAW_DATA_DIR, 'fight_details.csv'), index=False)
    pd.concat(all_fight_stats).to_csv(os.path.join(RAW_DATA_DIR, 'fight_stats.csv'), index=False)

if __name__ == "__main__":
    main() 
import os
import sys
from pathlib import Path
import asyncio
import argparse

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.scraping.event_scraper import EventScraper
from src.data.scraping.fight_scraper import FightScraper
from src.data.scraping.fighter_scraper import FighterScraper
from src.data.preprocessing import process_training_data
from src.utils.naming import standardize_dataframe_columns
from src.data.scraping.base_scraper import BaseScraper
# Add argument parsing before main
def parse_args():
    parser = argparse.ArgumentParser(description='Update UFC fight data')
    parser.add_argument('--setup-tunnels', action='store_true',
                       help='Setup SSH tunnels for proxies')
    parser.add_argument('--cleanup-tunnels', action='store_true',
                       help='Cleanup existing SSH tunnels')
    return parser.parse_args()

async def update_event_data():
    """Update event and fight data."""
    print("\n=== Updating Event Data ===")
    
    try:
        # Load and sort existing events
        existing_events = pd.read_csv(os.path.join(RAW_DATA_DIR, 'events.csv'))
        existing_events = standardize_dataframe_columns(existing_events)
        existing_events['date'] = pd.to_datetime(existing_events['date'])
        existing_events = existing_events.sort_values('date', ascending=False)
    except FileNotFoundError:
        print("No existing events file found. Creating new one.")
        existing_events = pd.DataFrame()
    
    # Get current events
    event_scraper = EventScraper('http://ufcstats.com/statistics/events/completed?page=all')
    current_events = await event_scraper.get_all_events()
    current_events = standardize_dataframe_columns(current_events)
    current_events['date'] = pd.to_datetime(current_events['date'])
    current_events = current_events[current_events['date'] <= pd.Timestamp.now()]
    current_events = current_events.sort_values('date', ascending=False)
    
    # Find new events
    new_events = current_events[~current_events['event'].isin(existing_events['event'])]
    
    if new_events.empty:
        print("No new events to update")
        return False
    
    print(f"\nFound {len(new_events)} new events to add:")
    for _, event in new_events.iterrows():
        print(f"- {event['event']} ({event['date'].strftime('%B %d, %Y')})")
    
    # Get fight data for new events
    fight_scraper = FightScraper()
    new_fight_details = []
    new_fight_stats = []
    new_fight_results = []
    
    print("\nFetching fight data for new events...")
    for _, event in new_events.iterrows():
        print(f"Processing {event['event']}...")
        details, stats, results = await fight_scraper.get_fight_data(event['url'])
        new_fight_details.append(standardize_dataframe_columns(details))
        new_fight_stats.append(standardize_dataframe_columns(stats))
        new_fight_results.append(standardize_dataframe_columns(results))
    
    # Update data files
    pd.concat([existing_events, new_events]).to_csv(
        os.path.join(RAW_DATA_DIR, 'events.csv'), index=False
    )
    
    for data, filename in [
        (new_fight_details, 'fight_details.csv'),
        (new_fight_stats, 'fight_stats.csv'),
        (new_fight_results, 'fight_results.csv')
    ]:
        try:
            existing = pd.read_csv(os.path.join(RAW_DATA_DIR, filename))
            existing = standardize_dataframe_columns(existing)
        except FileNotFoundError:
            existing = pd.DataFrame()
            
        pd.concat([existing] + data).to_csv(
            os.path.join(RAW_DATA_DIR, filename), index=False
        )
    
    return True

async def update_fighter_data():
    """Update fighter details and stats asynchronously."""
    print("\n=== Updating Fighter Data ===")
    
    # Load fight details and check columns
    fight_details = standardize_dataframe_columns(
        pd.read_csv(os.path.join(RAW_DATA_DIR, 'fight_details.csv'))
    )
    print("\nFight details columns:", fight_details.columns.tolist())
    
    try:
        existing_fighters = standardize_dataframe_columns(
            pd.read_csv(os.path.join(RAW_DATA_DIR, 'fighter_details.csv'))
        )
        print("Fighter details columns:", existing_fighters.columns.tolist())
    except FileNotFoundError:
        print("No existing fighter details file found. Creating new one.")
        existing_fighters = pd.DataFrame()
    
    # Get all unique fighter URLs from available columns
    all_fighters = set()
    
    # Try different possible column names
    possible_fighter_cols = ['fighter', 'fighter_url', 'url', 'fighter_1', 'fighter1']
    possible_opponent_cols = ['opponent', 'opponent_url', 'fighter_2', 'fighter2']
    
    # Find fighter columns that exist in the DataFrame
    fighter_cols = [col for col in possible_fighter_cols if col in fight_details.columns]
    opponent_cols = [col for col in possible_opponent_cols if col in fight_details.columns]
    
    if not fighter_cols and not opponent_cols:
        print("Error: Could not find fighter or opponent columns in fight details")
        print("Available columns:", fight_details.columns.tolist())
        return False
    
    # Collect all fighter URLs
    for col in fighter_cols + opponent_cols:
        all_fighters.update(fight_details[col].unique())
    
    # Convert to list and remove None/NaN values
    all_fighters = [f for f in all_fighters if pd.notna(f)]
    
    # Find new fighters
    if not existing_fighters.empty:
        # Try different possible URL columns in existing_fighters
        url_cols = ['url', 'fighter', 'fighter_url']
        existing_urls = set()
        for col in url_cols:
            if col in existing_fighters.columns:
                existing_urls.update(existing_fighters[col].values)
                break
        
        if not existing_urls:
            print("Warning: Could not find URL column in existing fighters data")
            print("Available columns:", existing_fighters.columns.tolist())
            
        new_fighters = [f for f in all_fighters if f not in existing_urls]
    else:
        new_fighters = all_fighters
    
    if not new_fighters:
        print("No new fighters to update")
        return False
    
    print(f"\nFound {len(new_fighters)} new fighters to add")
    
    # Get fighter data asynchronously
    fighter_scraper = FighterScraper()
    details_df, tott_df = await fighter_scraper.get_fighters_data(new_fighters)
    
    # Update files
    for df, filename in [
        (details_df, 'fighter_details.csv'),
        (tott_df, 'fighter_tott.csv')
    ]:
        if not df.empty:
            try:
                existing = pd.read_csv(os.path.join(RAW_DATA_DIR, filename))
                combined_data = pd.concat([existing, df])
            except FileNotFoundError:
                combined_data = df
            combined_data.to_csv(os.path.join(RAW_DATA_DIR, filename), index=False)
            print(f"Updated {filename}")
    
    return True

async def main():
    args = parse_args()
    print("Starting data update process...")
    
    if args.cleanup_tunnels:
        scraper = BaseScraper()
        await scraper.cleanup_existing_tunnels()
        return
    
    if args.setup_tunnels:
        scraper = BaseScraper()
        await scraper.setup_ssh_tunnels()
        print("\nSSH tunnels are ready. You can now run the script without --setup-tunnels")
        return
    
    # events_updated = await update_event_data()
    # fighters_updated = await update_fighter_data()
    
    # if events_updated or fighters_updated:
    if True:
        process_training_data()
        print("\n✅ All data updated successfully!")
    else:
        print("\n✅ All data is already up to date!")

if __name__ == "__main__":
    asyncio.run(main()) 
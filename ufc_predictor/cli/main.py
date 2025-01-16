import click
import traceback
import sys
from ufc_predictor.cli.predict import predict_fights
from ufc_predictor.cli.setup import setup_project
from ufc_predictor.data.preprocessing import process_training_data
from ufc_predictor.data.scraping.event_scraper import EventScraper
from ufc_predictor.data.scraping.fighter_scraper import FighterScraper
import asyncio
import pandas as pd
from tqdm import tqdm
from ufc_predictor.data.scraping.new_fight_scraper import FightScraper

@click.group()
def cli():
    """UFC fight prediction tool."""
    pass

@click.command()
def scrape_fight_data():
    """Scrape fight data from UFC stats website."""
    try:
        scraper = FightScraper()
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(scraper.get_fight_data('http://ufcstats.com/fight-details/6f0f84075fa1a4b0'))
        
    except Exception as e:
        click.echo(f"Error during scrape: {str(e)}")
        click.echo(traceback.format_exc())
        sys.exit(1)

@click.command()
@click.option('--events-url', default="http://ufcstats.com/statistics/events/completed?page=all",
              help='URL to scrape events from')
def update(events_url):
    """Update fight data from UFC stats website."""
    try:
        click.echo("🔄 Starting UFC data update...")
        
        # Run the async scraping in the event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_update_data(events_url))
        
        click.echo("✅ Data update complete!")
        
    except Exception as e:
        click.echo(f"Error during update: {str(e)}")
        click.echo(traceback.format_exc())
        sys.exit(1)

async def _update_data(events_url):
    """Async function to update all fight data."""
    # Phase 1: Fetch Events
    click.echo("\n📥 Phase 1: Fetching Events...")
    scraper = EventScraper(events_url)
    events = await scraper.get_event_details()
    events_df = scraper.to_dataframe(events)
    events_df.to_csv('data/raw/events.csv', index=False)
    click.echo(f"✅ Fetched {len(events_df)} events")
    
    # Phase 2: Fetch Fighter Data
    click.echo("\n📥 Phase 2: Fetching Fighter Data...")
    fighter_scraper = FighterScraper()
    all_fighter_details = []
    all_fighter_stats = []
    
    # Create a flat list of all fight URLs
    all_fights = [(event.name, url) for event in events for url in event.fight_urls]
    
    # Process fights with progress bar and more frequent updates
    with tqdm(total=len(all_fights), desc="Processing fights", mininterval=0.1) as pbar:
        for event_name, fight_url in all_fights:
            try:
                tqdm.write(f"Processing: {event_name} - {fight_url}")
                details_df, stats_df = await fighter_scraper.get_fighter_data(fight_url)
                all_fighter_details.append(details_df)
                all_fighter_stats.append(stats_df)
            except Exception as e:
                tqdm.write(f"⚠️  Error processing fight from {event_name} ({fight_url}): {str(e)}")
            finally:
                pbar.update(1)
                await asyncio.sleep(0.1)  # Small delay to prevent rate limiting
    
    # Phase 3: Save Fighter Data
    click.echo("\n💾 Phase 3: Saving Fighter Data...")
    if all_fighter_details:
        fighter_details_df = pd.concat(all_fighter_details, ignore_index=True)
        fighter_details_df.to_csv('data/raw/fighter_details.csv', index=False)
        click.echo(f"✅ Saved {len(fighter_details_df)} fighter details")
    
    if all_fighter_stats:
        fighter_stats_df = pd.concat(all_fighter_stats, ignore_index=True)
        fighter_stats_df.to_csv('data/raw/fighter_stats.csv', index=False)
        click.echo(f"✅ Saved {len(fighter_stats_df)} fighter stats")
    
    # Phase 4: Process Training Data
    click.echo("\n🔄 Phase 4: Processing Training Data...")
    process_training_data()
    click.echo("✅ Training data processing complete")

@click.command()
def reprocess():
    """Reprocess existing raw data without downloading new data."""
    try:
        process_training_data()
        click.echo("✅ Data reprocessing complete!")
    except FileNotFoundError:
        click.echo("Error: Raw data files not found. Run 'ufc setup' and update data first.")
    except Exception as e:
        click.echo(f"Error during reprocessing: {str(e)}")
        click.echo(traceback.format_exc())
        sys.exit(1)

cli.add_command(predict_fights, name='predict')
cli.add_command(setup_project, name='setup')
cli.add_command(reprocess, name='reprocess')
cli.add_command(update, name='update')
cli.add_command(scrape_fight_data, name='scrape')

if __name__ == "__main__":
    cli() 
import click
import traceback
import sys
from ufc_predictor.prediction.predictor import UFCPredictor
from ufc_predictor.cli.setup import setup_project
from ufc_predictor.data.preprocessing import UFCDataPreprocessor
from ufc_predictor.data.scraping.scraper import UFCScraper
from ufc_predictor.training.train import UFCModelTrainer
from ufc_predictor.prediction.predictor import UFCPredictor
from typing import Optional

import asyncio
import pandas as pd
from tqdm import tqdm
import anyio

@click.group()
def cli():
    """UFC fight prediction tool."""
    pass

@click.command()
@click.option('--events-url', default="http://ufcstats.com/statistics/events/completed?page=all",
              help='URL to scrape events from')
def update(events_url):
    """Update fight data from UFC stats website."""
    try:
        click.echo("🔄 Starting UFC data update...")
        
        # Run the async scraping in the event loop
        scraper = UFCScraper()
        anyio.run(scraper.run)
        
        click.echo("✅ Data update complete!")
        
    except Exception as e:
        click.echo(f"Error during update: {str(e)}")
        click.echo(traceback.format_exc())
        sys.exit(1)

@click.command()
def preprocess():
    """Reprocess existing raw data without downloading new data."""
    try:
        preprocessor = UFCDataPreprocessor()
        preprocessor.load_data(
            fights_path="data/raw/fight_data.csv", 
            events_path="data/raw/event_data.csv", 
            fighters_path="data/raw/fighter_data.csv",
            rounds_path="data/raw/round_data.csv"
        )
        preprocessor.preprocess()
        click.echo("✅ Data reprocessing complete!")
    except FileNotFoundError:
        click.echo("Error: Raw data files not found. Run 'ufc setup' and update data first.")
    except Exception as e:
        click.echo(f"Error during reprocessing: {str(e)}")
        click.echo(traceback.format_exc())
        sys.exit(1)

@click.command()
def train():
    """Train the model."""
    trainer = UFCModelTrainer()
    trainer.train()

@click.command()
@click.option('--fighter1', '-f1', help='Name of first fighter')
@click.option('--fighter2', '-f2', help='Name of second fighter')
def predict(fighter1: str, fighter2: str):
    """Predict the outcome of a UFC fight."""
    try:
        predictor = UFCPredictor()
        result = predictor.predict_single_fight(fighter1, fighter2)
        
        print(f"\n🥊 Prediction for {fighter1} vs {fighter2}:")
        print(f"Predicted winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        
        print(f"\nPhysical Differences:")
        
        print(f"Weight diff: {result['weight_diff']:.1f} lbs")
        print(f"Height diff: {result['height_diff']:.1f} cm")
        print(f"Reach diff: {result['reach_diff']:.1f} cm")
        
        print(f"\nForm Scores:")
        print(f"{fighter1}: {result['form_scores'][fighter1]:.2f}")
        print(f"{fighter2}: {result['form_scores'][fighter2]:.2f}")
        
        print(f"\nWin probabilities:")
        print(f"{fighter1}: {result['probability_fighter1_wins']*100:.1f}%")
        print(f"{fighter2}: {result['probability_fighter2_wins']*100:.1f}%")
        
    except ValueError as e:
        click.echo(f"Error making prediction: {str(e)}")
        if "Feature names seen at fit time" in str(e):
            click.echo("\nThere seems to be a mismatch between model features and available data.")
            click.echo("Try running 'ufc update' and 'ufc train' to refresh the model.")
        raise

cli.add_command(predict, name='predict')
cli.add_command(setup_project, name='setup')
cli.add_command(update, name='update')
cli.add_command(train, name='train')
cli.add_command(preprocess, name='preprocess')

if __name__ == "__main__":
    cli() 
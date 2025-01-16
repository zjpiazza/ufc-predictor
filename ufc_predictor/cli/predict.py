import click
from pathlib import Path
from ufc_predictor.models.predictor import UFCPredictor
from ufc_predictor.config import MODELS_DIR

@click.command()
@click.argument('fighter1')
@click.argument('fighter2')
@click.option('--model', '-m', default='model.h5', help='Name of the model file to use')
@click.option('--debug/--no-debug', default=False, help='Show debug information')
def predict_fights(fighter1, fighter2, model, debug):
    """Predict the outcome of a fight between FIGHTER1 and FIGHTER2."""
    try:
        predictor = UFCPredictor(model_path=MODELS_DIR / model)
        
        fight_data = {
            'fighter1': {'name': fighter1},
            'fighter2': {'name': fighter2}
        }
        
        if debug:
            click.echo("\nDebug: Fight Data")
            click.echo(fight_data)
        
        prediction = predictor.predict_fight(fight_data['fighter1'], fight_data['fighter2'])
        
        if debug:
            click.echo("\nDebug: Prediction Result")
            click.echo(prediction)
        
        click.echo("\nFight Prediction:")
        click.echo("-----------------")
        click.echo(f"\n{fighter1} vs {fighter2}")
        click.echo(f"{fighter1} win probability: {prediction['fighter1_win_probability']:.2%}")
        click.echo(f"{fighter2} win probability: {prediction['fighter2_win_probability']:.2%}")
        click.echo("\nStyle Matchup (positive favors fighter 1):")
        click.echo(f"Striking advantage: {prediction['style_matchup']['striking']:.2%}")
        click.echo(f"Grappling advantage: {prediction['style_matchup']['grappling']:.2%}")
        click.echo(f"Physical advantage: {prediction['style_matchup']['physical']:.2f}")
        click.echo(f"Experience advantage: {prediction['style_matchup']['experience']:.2%}")
        
    except IndexError:
        click.echo(f"Error: Could not find fighter data. Make sure both fighter names are exact matches.")
    except FileNotFoundError:
        click.echo("Error: Could not find fighter data files. Have you run 'ufc setup' and updated the data?")
    except Exception as e:
        click.echo(f"Error: {str(e)}")

if __name__ == "__main__":
    predict_fights()
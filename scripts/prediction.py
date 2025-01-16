import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.models.lstm import UFCPredictor

def main():
    # Initialize predictor with trained model
    model_path = Path(project_root) / 'models' / 'model.h5'
    predictor = UFCPredictor(model_path=model_path)
    
    # Example fight data
    example_fights = [
        {
            'fighter1': {
                'name': 'Islam Makhachev',
            },
            'fighter2': {
                'name': 'Arman Tsarukyan',
            }
        },
        # Add more example fights as needed
    ]
    
    # Make predictions
    predictions = predictor.predict_fights(example_fights)
    
    # Print results
    print("\nFight Predictions:")
    print("-----------------")
    for pred in predictions:
        print(f"\n{pred['fighter1']} vs {pred['fighter2']}")
        print(f"{pred['fighter1']} win probability: {pred['fighter1_win_probability']:.2%}")
        print(f"{pred['fighter2']} win probability: {pred['fighter2_win_probability']:.2%}")
        print("\nStyle Matchup (positive favors fighter 1):")
        print(f"Striking advantage: {pred['style_matchup']['striking']:.2%}")
        print(f"Grappling advantage: {pred['style_matchup']['grappling']:.2%}")
        print(f"Physical advantage: {pred['style_matchup']['physical']:.2f}")
        print(f"Experience advantage: {pred['style_matchup']['experience']:.2%}")

if __name__ == "__main__":
    main() 
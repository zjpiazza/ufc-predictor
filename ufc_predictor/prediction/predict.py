import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from keras.models import load_model

class UFCPredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.nn_model = None
        self.log_reg = None
        self.scaler = None
        self.load_models()
        
        # Load a sample of features.csv to get the exact column names
        features_sample = pd.read_csv("data/processed/features.csv", nrows=1)
        self.feature_columns = [col for col in features_sample.columns 
                              if col not in ['win', 'EVENT', 'BOUT', 'FIGHTER', 'OPPONENT']]
    
    def load_models(self):
        """Load trained models and scaler"""
        self.nn_model = load_model(self.model_dir / "nn_model.h5")
        self.log_reg = joblib.load(self.model_dir / "logistic_regression.joblib")
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
    
    def predict_fight(self, fighter1_stats: dict, fighter2_stats: dict) -> dict:
        """Make fight prediction using both models"""
        try:
            # Create feature vector
            features = self._create_feature_vector(fighter1_stats, fighter2_stats)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from both models
            nn_prob = self.nn_model.predict(features_scaled)[0][0]
            lr_prob = self.log_reg.predict_proba(features_scaled)[0][1]
            
            return {
                'neural_network': {
                    'probability': float(nn_prob),
                    'prediction': 'fighter1' if nn_prob > 0.5 else 'fighter2'
                },
                'logistic_regression': {
                    'probability': float(lr_prob),
                    'prediction': 'fighter1' if lr_prob > 0.5 else 'fighter2'
                }
            }
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            print("\nFeature columns expected:")
            print(self.feature_columns)
            print("\nFeature columns provided:")
            print(list(features.columns if hasattr(features, 'columns') else []))
            raise
    
    def _create_feature_vector(self, fighter1_stats: dict, fighter2_stats: dict) -> np.ndarray:
        """Create feature vector in the same format as training data"""
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=self.feature_columns)
        
        # Map fighter stats to feature names
        features = {
            # Basic stats
            'avg_takedown_accuracy_1': fighter1_stats['td_acc'],
            'avg_takedown_accuracy_2': fighter2_stats['td_acc'],
            'control_time_diff': fighter1_stats['ctrl_time'] - fighter2_stats['ctrl_time'],
            'ko_ratio_1': fighter1_stats['ko_ratio'],
            'ko_ratio_2': fighter2_stats['ko_ratio'],
            'win_ratio_1': fighter1_stats['win_ratio'],
            'win_ratio_2': fighter2_stats['win_ratio'],
            'height_diff': fighter1_stats['height'] - fighter2_stats['height'],
            'reach_diff': fighter1_stats['reach'] - fighter2_stats['reach'],
            'age_diff': fighter1_stats['age'] - fighter2_stats['age'],
            'total_fights_1': fighter1_stats['total_fights'],
            'total_fights_2': fighter2_stats['total_fights'],
            
            # Add all other features from the training data
            # ... (we need to map all features exactly as they appear in features.csv)
        }
        
        # Create DataFrame with single row
        df = pd.DataFrame([features])
        
        # Ensure all columns from training are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            for col in missing_cols:
                df[col] = 0  # Fill missing columns with 0
                
        # Ensure columns are in same order as training
        df = df[self.feature_columns]
        
        return df.values 
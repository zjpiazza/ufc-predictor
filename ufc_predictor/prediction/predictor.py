import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import pickle
from typing import Dict, Any, Optional  
import joblib  # Add this import at the top

class UFCPredictor:
    def __init__(self, model_dir='model', fighter_stats_path='data/processed/fighter_stats.csv', 
                 fighter_data_path='data/raw/fighter_data.csv'):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.fighter_stats = self._load_fighter_stats(fighter_stats_path)
        self.fighter_data = self._load_fighter_data(fighter_data_path)
        self.load_model()
    
    def _load_fighter_stats(self, stats_path: str) -> pd.DataFrame:
        """Load fighter statistics from CSV"""
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"Fighter stats file not found: {stats_path}")
        return pd.read_csv(stats_path)
    
    def _load_fighter_data(self, data_path: str) -> pd.DataFrame:
        """Load raw fighter data containing DOB information"""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Fighter data file not found: {data_path}")
        return pd.read_csv(data_path)
    
    def find_fighter(self, fighter_name: str) -> Optional[pd.DataFrame]:
        """
        Find a fighter in the fighter stats dataframe
        
        Args:
            fighter_name (str): Name of the fighter to look up
            
        Returns:
            Optional[pd.DataFrame]: DataFrame row with fighter data or None if not found
        """
        if pd.isna(fighter_name) or fighter_name is None:
            return None
            
        fighter_name_str = str(fighter_name).lower()
        fighter = self.fighter_stats[self.fighter_stats['name'].str.lower() == fighter_name_str]
        
        return fighter if not fighter.empty else None

    def get_fighter_features(self, fighter_name: str) -> Optional[Dict[str, Any]]:
        """Get features for a specific fighter"""
        fighter_df = self.find_fighter(fighter_name)
        if fighter_df is None:
            raise ValueError(f"Fighter not found: {fighter_name}")
        
        try:
            # Get fighter ID to look up DOB
            fighter_id = fighter_df['fighter_id'].iloc[0]
            
            # Look up fighter's DOB from raw data
            fighter_raw = self.fighter_data[self.fighter_data['fighter_id'] == fighter_id]
            if not fighter_raw.empty and pd.notna(fighter_raw['fighter_dob'].iloc[0]):
                dob = pd.to_datetime(fighter_raw['fighter_dob'].iloc[0])
                age = (pd.Timestamp.now() - dob).days / 365.25
            else:
                age = None  # If DOB is not available
            
            fighter = {
                'fighter_id': fighter_id,
                'name': fighter_df['name'].iloc[0],
                'win_ratio': float(fighter_df['win_ratio'].iloc[0]),
                'recent_win_ratio': float(fighter_df['recent_win_ratio'].iloc[0]),
                'height': float(fighter_df['height'].iloc[0]),
                'weight': float(fighter_df['weight'].iloc[0]),
                'reach': float(fighter_df['reach'].iloc[0]),
                'age': age,  # Now calculated from DOB
                'stance': fighter_df['stance'].iloc[0],
                'total_fights': int(fighter_df['total_fights'].iloc[0]),
                'avg_strikes_landed': float(fighter_df['avg_strikes_landed'].iloc[0]),
                'avg_strikes_attempted': float(fighter_df['avg_strikes_attempted'].iloc[0]),
                'avg_strikes_accuracy': float(fighter_df['avg_strikes_accuracy'].iloc[0]),
                'avg_takedowns': float(fighter_df['avg_takedowns'].iloc[0]),
                'avg_takedown_accuracy': float(fighter_df['avg_takedown_accuracy'].iloc[0]),
                'avg_knockdowns': float(fighter_df['avg_knockdowns'].iloc[0]),
                'ko_ratio': float(fighter_df['ko_ratio'].iloc[0]),
                'avg_control_time': float(fighter_df['avg_control_time'].iloc[0]),
            }
            return fighter
        except (KeyError, ValueError, IndexError) as e:
            print(f"Error processing fighter data for {fighter_name}: {e}")
            return None
        
    
    def load_model(self):
        """Load the trained model and scaler"""
        model_path = self.model_dir / 'logistic_regression.joblib'
        scaler_path = self.model_dir / 'scaler.joblib'
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Model or scaler not found. Please train the model first.")
        
        # Load using joblib instead of XGBoost
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def get_model_features(self) -> list:
        """Get the feature names that the model was trained with"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # For scikit-learn models, get feature names from scaler
        return self.scaler.get_feature_names_out().tolist()

    def predict(self, features_df):
        """Make predictions for given features"""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get expected feature names from scaler
        expected_features = set(self.scaler.get_feature_names_out())
        provided_features = set(features_df.columns)
        
        # Check for mismatches
        missing_features = expected_features - provided_features
        extra_features = provided_features - expected_features
        
        if missing_features or extra_features:
            error_msg = "There seems to be a mismatch between model features and available data.\n"
            if missing_features:
                error_msg += "\nMissing features (required by model):\n- "
                error_msg += "\n- ".join(sorted(missing_features))
            if extra_features:
                error_msg += "\n\nExtra features (not used by model):\n- "
                error_msg += "\n- ".join(sorted(extra_features))
            error_msg += "\n\nTry running 'ufc update' and 'ufc train' to refresh the model."
            raise ValueError(error_msg)
        
        # Process features similar to training
        X = features_df.copy()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction using scikit-learn model
        probabilities = self.model.predict_proba(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        return [
            {
                'probability_fighter2_wins': float(prob[1]),
                'probability_fighter1_wins': float(prob[0]),
                'predicted_winner': 'fighter2' if pred == 1 else 'fighter1',
                'confidence': float(max(prob))
            }
            for prob, pred in zip(probabilities, predictions)
        ]

    def predict_single_fight(self, fighter_name: str, fighter_opponent_name: str):
        """Make prediction for a single fight"""
        fighter_features = self.get_fighter_features(fighter_name)
        fighter_opponent_features = self.get_fighter_features(fighter_opponent_name)
        
        if not fighter_features or not fighter_opponent_features:
            raise ValueError(f"Could not find features for one or both fighters")
        
        # Get expected feature names in correct order
        expected_features = self.scaler.get_feature_names_out()
        
        # Calculate differences
        height_diff = fighter_features['height'] - fighter_opponent_features['height']
        reach_diff = fighter_features['reach'] - fighter_opponent_features['reach']
        age_diff = (fighter_features['age'] or 0) - (fighter_opponent_features['age'] or 0)
        
        # Create feature dictionary matching the model's expected features
        feature_dict = {
            # Differences
            'height_diff': height_diff,
            'reach_diff': reach_diff,
            'age_diff': age_diff,
            
            # Fighter features
            'fighter_fight_no': fighter_features['total_fights'],
            'fighter_win_ratio': fighter_features['win_ratio'],
            'fighter_recent_win_ratio': fighter_features['recent_win_ratio'],
            'fighter_strikes_landed': fighter_features['avg_strikes_landed'],
            'fighter_strikes_attempted': fighter_features['avg_strikes_attempted'],
            'fighter_strikes_accuracy': fighter_features['avg_strikes_accuracy'],
            'fighter_takedowns': fighter_features['avg_takedowns'],
            'fighter_takedown_accuracy': fighter_features['avg_takedown_accuracy'],
            'fighter_knockdowns': fighter_features['avg_knockdowns'],
            'fighter_ko_ratio': fighter_features['ko_ratio'],
            'fighter_control_time': fighter_features['avg_control_time'],
            
            # Opponent features
            'opponent_fight_no': fighter_opponent_features['total_fights'],
            'opponent_win_ratio': fighter_opponent_features['win_ratio'],
            'opponent_recent_win_ratio': fighter_opponent_features['recent_win_ratio'],
            'opponent_strikes_landed': fighter_opponent_features['avg_strikes_landed'],
            'opponent_strikes_attempted': fighter_opponent_features['avg_strikes_attempted'],
            'opponent_strikes_accuracy': fighter_opponent_features['avg_strikes_accuracy'],
            'opponent_takedowns': fighter_opponent_features['avg_takedowns'],
            'opponent_takedown_accuracy': fighter_opponent_features['avg_takedown_accuracy'],
            'opponent_knockdowns': fighter_opponent_features['avg_knockdowns'],
            'opponent_ko_ratio': fighter_opponent_features['ko_ratio'],
            'opponent_control_time': fighter_opponent_features['avg_control_time'],
        }
        
        # Create DataFrame with features in correct order
        features = pd.DataFrame([{name: feature_dict.get(name, 0) for name in expected_features}])
        
        # Make prediction
        prediction = self.predict(features)[0]
        
        return {
            'fighter1': fighter_name,
            'fighter2': fighter_opponent_name,
            'probability_fighter1_wins': prediction['probability_fighter1_wins'],
            'probability_fighter2_wins': prediction['probability_fighter2_wins'],
            'predicted_winner': fighter_name if prediction['predicted_winner'] == 'fighter1' else fighter_opponent_name,
            'confidence': prediction['confidence']
        }
    
    def _determine_weight_class(self, weight_lbs):
        """Determine weight class based on weight"""
        if weight_lbs <= 125:
            return 'flyweight'
        elif weight_lbs <= 135:
            return 'bantamweight'
        elif weight_lbs <= 145:
            return 'featherweight'
        elif weight_lbs <= 155:
            return 'lightweight'
        elif weight_lbs <= 170:
            return 'welterweight'
        elif weight_lbs <= 185:
            return 'middleweight'
        elif weight_lbs <= 205:
            return 'light_heavyweight'
        else:
            return 'heavyweight' 
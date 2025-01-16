import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import load_model
from datetime import datetime

class UFCPredictor:
    """UFC fight predictor using LSTM model."""
    
    def __init__(self, model_path='models/model.h5'):
        """Initialize the predictor with a trained model."""
        self.model = load_model(model_path)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()
        
    def _clean_ratio(self, value):
        """Convert 'X of Y' strings to float ratios."""
        if isinstance(value, str) and 'of' in value:
            try:
                numerator, denominator = value.split(' of ')
                return float(numerator) / float(denominator)
            except:
                return 0.0
        return value
    
    def _convert_date_to_age(self, date_str):
        """Convert date string to age in years."""
        try:
            return (datetime.now() - pd.to_datetime(date_str)).days / 365.25
        except:
            return np.nan
    
    def _preprocess_fighter_data(self, fighter_data):
        """Preprocess a single fighter's data."""
        # Convert dates to age
        if 'dob' in fighter_data:
            fighter_data['age'] = self._convert_date_to_age(fighter_data['dob'])
        
        # Encode stance
        if 'stance' in fighter_data:
            fighter_data['stance_encoded'] = self.label_encoder.fit_transform([fighter_data['stance']])[0]
        
        # Clean ratio columns
        ratio_columns = ['sig_str_', 'sig_str__%', 'td_%']
        for col in ratio_columns:
            if col in fighter_data:
                fighter_data[col] = self._clean_ratio(fighter_data[col])
        
        return fighter_data
    
    def _calculate_style_matchup(self, fighter1_data, fighter2_data):
        """Calculate style matchup scores between fighters."""
        style_scores = {}
        
        # Striking matchup (already in decimal form)
        f1_striking = fighter1_data.get('sig.str. %', 0)
        f2_striking = fighter2_data.get('sig.str. %', 0)
        style_scores['striking_advantage'] = f1_striking - f2_striking
        
        # Takedown matchup (already in decimal form)
        f1_td = fighter1_data.get('td %', 0)
        f2_td = fighter2_data.get('td %', 0)
        style_scores['grappling_advantage'] = f1_td - f2_td
        
        # Physical matchup (heights/reaches already in inches)
        f1_height = fighter1_data.get('height', 0)
        f2_height = fighter2_data.get('height', 0)
        f1_reach = fighter1_data.get('reach', 0)
        f2_reach = fighter2_data.get('reach', 0)
        
        style_scores['physical_advantage'] = (
            (f1_height - f2_height) / 12 +  # Convert height difference to feet
            (f1_reach - f2_reach) / 72      # Normalize reach difference
        ) / 2
        
        # Experience advantage
        f1_fights = fighter1_data.get('fight_no', 0)
        f2_fights = fighter2_data.get('fight_no', 0)
        max_fights = max(f1_fights, f2_fights)
        style_scores['experience_advantage'] = (
            0 if max_fights == 0 else (f1_fights - f2_fights) / max_fights
        )
        
        return style_scores

    def _convert_height(self, height_str):
        """Convert height string (e.g. "5' 10"") to inches."""
        try:
            feet, inches = height_str.replace('"', '').split("' ")
            return (int(feet) * 12) + int(inches)
        except:
            return 0

    def _convert_reach(self, reach_str):
        """Convert reach string (e.g. "70"") to inches."""
        try:
            return int(reach_str.replace('"', ''))
        except:
            return 0

    def _convert_weight(self, weight_str):
        """Convert weight string (e.g. "155 lbs.") to pounds."""
        try:
            return int(weight_str.split()[0])
        except:
            return 0

    def _load_fighter_data(self, fighter_name):
        """Load fighter data from stored CSV files."""
        from ufc_predictor.config import RAW_DATA_DIR
        
        # Load fighter details
        fighter_details = pd.read_csv(RAW_DATA_DIR / 'fighter_details.csv')
        fighter_tott = pd.read_csv(RAW_DATA_DIR / 'fighter_tott.csv')
        fight_stats = pd.read_csv(RAW_DATA_DIR / 'fight_stats.csv')
        
        # Find the fighter
        fighter = fighter_details[
            (fighter_details['first'] + ' ' + fighter_details['last']).str.lower() == fighter_name.lower()
        ].iloc[0]
        
        # Get their most recent stats
        recent_stats = fight_stats[fight_stats['fighter'] == fighter_name].iloc[-1]
        
        # Get their time on the tools
        tott = fighter_tott[fighter_tott['url'] == fighter['url']].iloc[0]
        
        # Combine all data (already in correct format)
        fighter_data = {
            'name': fighter_name,
            'height': tott['height'],
            'weight': tott['weight'],
            'reach': tott['reach'],
            'stance': tott['stance'],
            'sig_str__%': recent_stats['sig.str. %'],
            'td_%': recent_stats['td %'],
            'fight_no': len(fight_stats[fight_stats['fighter'] == fighter_name]),
            'dob': tott['dob']
        }
        
        return fighter_data

    def predict_fight(self, fighter1_data, fighter2_data):
        """Predict the outcome of a fight between two fighters."""
        # Load full fighter data if only names provided
        if len(fighter1_data) == 1 and 'name' in fighter1_data:
            fighter1_data = self._load_fighter_data(fighter1_data['name'])
        if len(fighter2_data) == 1 and 'name' in fighter2_data:
            fighter2_data = self._load_fighter_data(fighter2_data['name'])
        
        # Get base prediction
        f1_processed = self._preprocess_fighter_data(fighter1_data)
        f2_processed = self._preprocess_fighter_data(fighter2_data)
        
        # Calculate style matchup scores
        style_scores = self._calculate_style_matchup(fighter1_data, fighter2_data)
        
        # Get model prediction
        fight_data = {
            **{k: v for k, v in f1_processed.items()},
            **{f"{k}_opponent": v for k, v in f2_processed.items()}
        }
        
        df = pd.DataFrame([fight_data])
        df.columns = df.columns.str.replace('.', '_').str.replace(' ', '_')
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        X = self.scaler.fit_transform(df)
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        win_prob = self.model.predict(X_reshaped, verbose=0)[0][0]
        
        return {
            'fighter1_win_probability': float(win_prob),
            'fighter2_win_probability': float(1 - win_prob),
            'style_matchup': {
                'striking': style_scores['striking_advantage'],
                'grappling': style_scores['grappling_advantage'],
                'physical': style_scores['physical_advantage'],
                'experience': style_scores['experience_advantage']
            }
        }

    def predict_fights(self, fights_data):
        """Predict outcomes for multiple fights.
        
        Args:
            fights_data (list): List of dicts containing fighter pairs
            
        Returns:
            list: Predictions for each fight
        """
        predictions = []
        for fight in fights_data:
            pred = self.predict_fight(fight['fighter1'], fight['fighter2'])
            predictions.append({
                'fighter1': fight['fighter1'].get('name', 'Fighter 1'),
                'fighter2': fight['fighter2'].get('name', 'Fighter 2'),
                **pred
            })
        return predictions 
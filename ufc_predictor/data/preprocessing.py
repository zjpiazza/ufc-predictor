import pandas as pd
import numpy as np
from pathlib import Path
import logging
from ufc_predictor.schemas import rounds_dtypes
from ufc_predictor.schemas import fights_dtypes
from ufc_predictor.schemas import events_dtypes
from ufc_predictor.schemas import fighters_dtypes
from typing import Optional
import matplotlib.pyplot as plt

class UFCDataPreprocessor:
    def __init__(self):
        self.rounds_df = None
        self.fights_df = None
        self.events_df = None
        self.fighters_df = None
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "preprocessing.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def load_data(self, rounds_path: str, fights_path: str, 
                 events_path: str, fighters_path: str) -> None:
        
        """Load the raw CSV data files"""
        # Define NA values to include dashes
        na_values = ['--------', '---', '--','']
        # Load data with proper dtypes from schema and NA handling
        self.rounds_df = pd.read_csv(rounds_path, na_values=na_values, converters={'ctrl_time': self._convert_control_time})
        self.fights_df = pd.read_csv(fights_path, na_values=na_values)
        self.events_df = pd.read_csv(events_path, na_values=na_values)
        self.fighters_df = pd.read_csv(fighters_path, na_values=na_values)
        
        # Convert date columns
        self.events_df['event_date'] = pd.to_datetime(self.events_df['event_date'])
        self.fighters_df['fighter_dob'] = pd.to_datetime(self.fighters_df['fighter_dob'])
        
        # Convert numeric columns in fighters_df
        numeric_cols = ['fighter_height_cm', 'fighter_weight_lbs', 'fighter_reach_cm']
        for col in numeric_cols:
            self.fighters_df[col] = pd.to_numeric(self.fighters_df[col], errors='coerce')

    def preprocess(self) -> None:
        """Main preprocessing pipeline"""
        # Load data using schema
        self.load_data(
            rounds_path="data/raw/round_data.csv",
            fights_path="data/raw/fight_data.csv",
            events_path="data/raw/event_data.csv",
            fighters_path="data/raw/fighter_data.csv"
        )
        
        # Merge fights with events to get dates
        fight_data = self.fights_df.merge(
            self.events_df[['event_id', 'event_date']], 
            on='event_id',
            how='left'
        )
        fight_data = fight_data.sort_values('event_date')
        
        # Add fight number for each fighter
        fight_data['fightNo_fighter1'] = fight_data.groupby('fighter_1').cumcount() + 1
        fight_data['fightNo_fighter2'] = fight_data.groupby('fighter_2').cumcount() + 1
        
        # Filter to fights where both fighters have experience
        fight_data = fight_data[
            (fight_data['fightNo_fighter1'] >= 3) & 
            (fight_data['fightNo_fighter2'] >= 3)
        ]
        
        # Calculate fight stats with historical lookback
        fight_stats = self._calculate_fight_stats(fight_data)
        
        # Save intermediate fight data
        fight_stats.to_csv(self.output_dir / "fight_with_stats_precomp.csv", index=False)
        
        # Generate and save fighter career stats
        self._generate_fighter_stats()
        
        # Create final features
        features_df = self._create_model_features(fight_stats)
        
        # Save processed data
        features_df.to_csv(self.output_dir / "features.csv", index=False)
        print(f"\nPreprocessed data saved to {self.output_dir}")
        print(f"Features shape: {features_df.shape}")

    def _calculate_fighter_stats_at_time(self, fighter_id: str, current_date: pd.Timestamp, 
                                       historical_fights: pd.DataFrame) -> Optional[dict]:
        """Calculate fighter statistics using only fights before the given date"""
        # Get all previous fights for this fighter
        fighter_fights = historical_fights[
            ((historical_fights['fighter_1'] == fighter_id) | 
             (historical_fights['fighter_2'] == fighter_id)) &
            (historical_fights['event_date'] < current_date)
        ]
        
        if len(fighter_fights) == 0:
            return None
        
        # Basic stats
        total_fights = len(fighter_fights)
        wins = sum(fighter_fights['winner'] == fighter_id)
        draws = sum(fighter_fights['result'] == 'Draw')
        no_contests = sum(fighter_fights['result'] == 'No Contest')
        
        # Get rounds data
        fighter_rounds = self.rounds_df[
            (self.rounds_df['fight_id'].isin(fighter_fights['fight_id'])) &
            (self.rounds_df['fighter_id'] == fighter_id)
        ]
        
        # Get fighter data first since we need it for several calculations
        fighter_data = self.fighters_df[self.fighters_df['fighter_id'] == fighter_id].iloc[0]
        
        # Calculate age at time of fight
        age = (current_date - fighter_data['fighter_dob']).days / 365.25
        
        # Calculate form score before calculating other stats
        fighter_results = []
        for _, fight in fighter_fights.sort_values('event_date').iterrows():
            is_win = fight['winner'] == fighter_id
            fighter_results.append(is_win)
        
        form_score = self.calculate_form_score(fighter_results)
        
        # Convert measurements to inches for consistency
        height_cm = fighter_data['fighter_height_cm']
        reach_cm = fighter_data['fighter_reach_cm']
        
        height_inches = height_cm / 2.54 if height_cm else 0
        reach_inches = reach_cm / 2.54 if reach_cm else 0
        
        # Just use the raw control time for now
        control_time = fighter_rounds['ctrl_time'].apply(self._convert_control_time).mean() if len(fighter_rounds) > 0 else 0
        
        # Create stats dictionary with the exact names needed for fighter_stats.csv
        stats = {
            'total_fights': total_fights,
            'win_ratio': wins / total_fights if total_fights > 0 else 0,
            'recent_win_ratio': None,  # Will be calculated later
            'height': height_inches,
            'weight': fighter_data['fighter_weight_lbs'],
            'reach': reach_inches,
            'stance': fighter_data['fighter_stance'],
            'age': age,
            'form_score': form_score,
            'avg_strikes_landed': fighter_rounds['strikes_succ'].sum() / total_fights if total_fights > 0 else 0,
            'avg_strikes_attempted': fighter_rounds['strikes_att'].sum() / total_fights if total_fights > 0 else 0,
            'avg_strikes_accuracy': fighter_rounds['strikes_succ'].sum() / fighter_rounds['strikes_att'].sum() if fighter_rounds['strikes_att'].sum() > 0 else 0,
            'avg_takedowns': fighter_rounds['takedown_succ'].sum() / total_fights if total_fights > 0 else 0,
            'avg_takedown_accuracy': fighter_rounds['takedown_succ'].sum() / fighter_rounds['takedown_att'].sum() if fighter_rounds['takedown_att'].sum() > 0 else 0,
            'avg_knockdowns': fighter_rounds['knockdowns'].sum() / total_fights if total_fights > 0 else 0,
            'ko_ratio': sum(fighter_fights['result'] == 'KO/TKO') / total_fights if total_fights > 0 else 0,
            'avg_control_time': control_time,  # Using raw control time
        }
        
        # Calculate form score (last 3 fights)
        recent_fights = fighter_fights.sort_values('event_date', ascending=False).head(3)
        stats['recent_win_ratio'] = sum(recent_fights['winner'] == fighter_id) / len(recent_fights) if len(recent_fights) > 0 else 0
        
        # Calculate strike stats by location
        location_mapping = {
            'head': ('head_strikes_succ', 'head_strikes_att'),
            'body': ('body_strikes_succ', 'body_strikes_att'),
            'leg': ('leg_strikes_succ', 'leg_strikes_att')
        }
        
        for location, (succ_col, att_col) in location_mapping.items():
            landed = fighter_rounds[succ_col].sum()
            attempted = fighter_rounds[att_col].sum()
            accuracy = landed / attempted if attempted > 0 else 0
            per_round = landed / len(fighter_rounds) if len(fighter_rounds) > 0 else 0
            
            stats.update({
                f'{location}_accuracy': accuracy,
                f'{location}_per_round': per_round
            })
        
        # Calculate strike stats by position
        position_mapping = {
            'distance': ('distance_strikes_succ', 'distance_strikes_att'),
            'clinch': ('clinch_strikes_succ', 'clinch_strikes_att'),
            'ground': ('ground_strikes_succ', 'ground_strikes_att')
        }
        
        for position, (succ_col, att_col) in position_mapping.items():
            landed = fighter_rounds[succ_col].sum()
            attempted = fighter_rounds[att_col].sum()
            accuracy = landed / attempted if attempted > 0 else 0
            per_round = landed / len(fighter_rounds) if len(fighter_rounds) > 0 else 0
            
            stats.update({
                f'{position}_accuracy': accuracy,
                f'{position}_per_round': per_round
            })
        
        return stats

    def _calculate_derived_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage and per-round statistics"""
        for col in df.columns:
            parts = col.split('_')
            if 'CTRL' in parts:
                # Control time percentage
                df[f"{parts[0]}_pct_{parts[1]}"] = df[col] / df[f"TotalTime_{parts[1]}"]
            elif 'attempts' in parts:
                # Accuracy
                df[f"{parts[0]}_acc_{parts[1]}"] = (
                    df[col.replace('attempts', 'landed')] / df[col]
                )
                # Per round stats
                df[f"{parts[0]}_perRound_{parts[1]}"] = (
                    df[col.replace('attempts', 'landed')] / df[f"TotalTime_{parts[1]}"]) * 300
        
        return df

    def _calculate_fight_stats(self, fight_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate fight statistics with historical lookback"""
        fight_stats = []
        
        for idx, fight in fight_data.iterrows():
            # Get historical fights before this one
            historical_fights = fight_data.iloc[:idx]
            
            # Calculate stats for both fighters
            fighter1_stats = self._calculate_fighter_stats_at_time(
                fight['fighter_1'], fight['event_date'], historical_fights)
            fighter2_stats = self._calculate_fighter_stats_at_time(
                fight['fighter_2'], fight['event_date'], historical_fights)
            
            if fighter1_stats and fighter2_stats:
                fight_stat = {
                    'event_id': fight['event_id'],
                    'fight_id': fight['fight_id'],
                    'fighter1_name': fight['fighter_1'],
                    'fighter2_name': fight['fighter_2'],
                    'winner': fight['winner'],
                    'fighter1_fight_no': fight['fightNo_fighter1'],
                    'fighter2_fight_no': fight['fightNo_fighter2'],
                    'weight_class': fight['weight_class'],
                    'gender': fight['gender'],
                    'time_format': fight['time_format']
                }
                
                # Add fighter 1 stats with proper prefix
                for stat, value in fighter1_stats.items():
                    fight_stat[f'fighter1_{stat}'] = value
                
                # Add fighter 2 stats with proper prefix
                for stat, value in fighter2_stats.items():
                    fight_stat[f'fighter2_{stat}'] = value
                
                fight_stats.append(fight_stat)
        
        return pd.DataFrame(fight_stats)

    def _convert_control_time(self, time_str: str) -> int:
        """Convert control time string (MM:SS) to seconds"""
        try:
            if pd.isna(time_str) or not isinstance(time_str, str):
                return 0
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except (ValueError, AttributeError):
            return 0

    def _create_model_features(self, fight_stats: pd.DataFrame) -> pd.DataFrame:
        """Create final features using clean naming scheme"""
        features = []
        
        for _, fight in fight_stats.iterrows():
            feature = {
                'win': 1 if fight['winner'] == fight['fighter1_name'] else 0,
                
                # Physical differences with weight class emphasis
                'height_diff': fight['fighter1_height'] - fight['fighter2_height'],
                'reach_diff': fight['fighter1_reach'] - fight['fighter2_reach'],
                'age_diff': fight['fighter1_age'] - fight['fighter2_age'],
                
                # Form and experience
                'fighter_form_score': fight['fighter1_form_score'],
                'opponent_form_score': fight['fighter2_form_score'],
                'fighter_fight_no': fight['fighter1_fight_no'],
                'opponent_fight_no': fight['fighter2_fight_no'],
                
                # Win ratios
                'fighter_win_ratio': fight['fighter1_win_ratio'],
                'fighter_recent_win_ratio': fight['fighter1_recent_win_ratio'],
                'opponent_win_ratio': fight['fighter2_win_ratio'],
                'opponent_recent_win_ratio': fight['fighter2_recent_win_ratio'],
                
                # Strike stats
                'fighter_strikes_landed': fight['fighter1_avg_strikes_landed'],
                'fighter_strikes_attempted': fight['fighter1_avg_strikes_attempted'],
                'fighter_strikes_accuracy': fight['fighter1_avg_strikes_accuracy'],
                'opponent_strikes_landed': fight['fighter2_avg_strikes_landed'],
                'opponent_strikes_attempted': fight['fighter2_avg_strikes_attempted'],
                'opponent_strikes_accuracy': fight['fighter2_avg_strikes_accuracy'],
                
                # Takedown stats
                'fighter_takedowns': fight['fighter1_avg_takedowns'],
                'fighter_takedown_accuracy': fight['fighter1_avg_takedown_accuracy'],
                'opponent_takedowns': fight['fighter2_avg_takedowns'],
                'opponent_takedown_accuracy': fight['fighter2_avg_takedown_accuracy'],
                
                # Other stats
                'fighter_knockdowns': fight['fighter1_avg_knockdowns'],
                'fighter_ko_ratio': fight['fighter1_ko_ratio'],
                'opponent_knockdowns': fight['fighter2_avg_knockdowns'],
                'opponent_ko_ratio': fight['fighter2_ko_ratio'],
                
                # Control time
                'fighter_control_time': fight['fighter1_avg_control_time'],
                'opponent_control_time': fight['fighter2_avg_control_time'],
            }
            features.append(feature)
        
        features_df = pd.DataFrame(features)
        
        # Log statistics for all numerical columns
        self.logger.info("\n📊 Feature Statistics:")
        for col in features_df.select_dtypes(include=[np.number]).columns:
            stats = features_df[col].describe()
            self.logger.info(f"\n{col}:")
            self.logger.info(f"  Min: {stats['min']:.3f}")
            self.logger.info(f"  Max: {stats['max']:.3f}")
            self.logger.info(f"  Mean: {stats['mean']:.3f}")
            self.logger.info(f"  Std: {stats['std']:.3f}")
            
            # Log distribution of values to check for uniformity
            unique_count = features_df[col].nunique()
            self.logger.info(f"  Unique values: {unique_count}")
            
            # If there are very few unique values, show their distribution
            if unique_count < 10:
                value_counts = features_df[col].value_counts()
                self.logger.info("  Value distribution:")
                for val, count in value_counts.items():
                    self.logger.info(f"    {val}: {count} ({count/len(features_df)*100:.1f}%)")
        
        # Create histograms for key features
        key_features = ['fighter_form_score', 'opponent_form_score', 
                       'height_diff', 'reach_diff', 'age_diff',
                       'fighter_win_ratio', 'opponent_win_ratio']
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(key_features, 1):
            plt.subplot(3, 3, i)
            plt.hist(features_df[feature], bins=30)
            plt.title(feature)
            plt.xlabel('Value')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png')
        plt.close()
        
        return features_df

    def _generate_fighter_stats(self) -> pd.DataFrame:
        """Generate career-level statistics for each fighter"""
        fighter_stats = []
        
        # Merge fights with events to get dates
        fights_with_dates = self.fights_df.merge(
            self.events_df[['event_id', 'event_date']], 
            on='event_id',
            how='left'
        ).sort_values('event_date')  # Sort by date to get most recent fights
        
        # Get the latest date to calculate career stats
        latest_date = self.events_df['event_date'].max()
        
        # Calculate career stats for each fighter
        for fighter_id in self.fighters_df['fighter_id'].unique():
            # Get all fights for this fighter
            fighter_fights = fights_with_dates[
                (fights_with_dates['fighter_1'] == fighter_id) | 
                (fights_with_dates['fighter_2'] == fighter_id)
            ]
            
            if len(fighter_fights) == 0:
                continue
            
            # Calculate career stats using the same logic as before
            stats = self._calculate_fighter_stats_at_time(
                fighter_id, 
                latest_date, 
                fights_with_dates  # Pass the merged DataFrame with dates
            )
            
            if stats:
                # Add fighter identification
                fighter_data = self.fighters_df[self.fighters_df['fighter_id'] == fighter_id].iloc[0]
                stats['fighter_id'] = fighter_id
                stats['name'] = f"{fighter_data['fighter_f_name']} {fighter_data['fighter_l_name']}".strip()
                
                # Calculate recent win ratio from last 3 fights
                recent_fights = fighter_fights.sort_values('event_date', ascending=False).head(3)
                if len(recent_fights) > 0:
                    recent_wins = sum(
                        (recent_fights['fighter_1'] == fighter_id) & (recent_fights['winner'] == fighter_id) |
                        (recent_fights['fighter_2'] == fighter_id) & (recent_fights['winner'] == fighter_id)
                    )
                    stats['recent_win_ratio'] = recent_wins / len(recent_fights)
                else:
                    stats['recent_win_ratio'] = 0.0
                
                fighter_stats.append(stats)
        
        # Convert to DataFrame
        fighter_stats_df = pd.DataFrame(fighter_stats)
        
        # Select only the columns we need for prediction
        columns_to_keep = [
            'fighter_id', 'name', 'win_ratio', 'recent_win_ratio',
            'height', 'weight', 'reach', 'stance', 'total_fights',
            'avg_strikes_landed', 'avg_strikes_attempted', 'avg_takedowns',
            'avg_knockdowns', 'avg_strikes_accuracy', 'avg_takedown_accuracy',
            'ko_ratio', 'avg_control_time', 'form_score'
        ]
        
        fighter_stats_df = fighter_stats_df[columns_to_keep]
        
        # Save to CSV
        fighter_stats_df.to_csv(self.output_dir / "fighter_stats.csv", index=False)
        
        return fighter_stats_df

    def calculate_form_score(self, wins, max_fights=5):
        """Calculate form score based on recent fight results"""
        score = 0
        coef = 0.5
        
        # Take only the last max_fights fights, reversed so most recent is first
        recent_results = wins[-max_fights:][::-1]
        
        for win in recent_results:
            if win:
                score += coef
            else:
                score -= coef
            coef -= 0.1
        
        # Log some debug info about form score calculation
        self.logger.debug(f"Form score calculation:")
        self.logger.debug(f"  Recent results: {recent_results}")
        self.logger.debug(f"  Final score: {score}")
        
        return score
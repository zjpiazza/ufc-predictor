# import xgboost as xgb
from pathlib import Path
from typing import Any, Dict, Optional

import joblib  # Add this import at the top
import numpy as np
import pandas as pd


class UFCPredictor:
    def __init__(
        self,
        model_dir="model",
        fighter_stats_path="data/processed/fighter_stats.csv",
        fighter_data_path="data/raw/fighter_data.csv",
    ):
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
        fighter = self.fighter_stats[
            self.fighter_stats["name"].str.lower() == fighter_name_str
        ]

        return fighter if not fighter.empty else None

    def get_fighter_features(self, fighter_name: str) -> Optional[Dict[str, Any]]:
        """Get features for a specific fighter"""
        fighter_df = self.find_fighter(fighter_name)
        if fighter_df is None:
            raise ValueError(f"Fighter not found: {fighter_name}")

        try:
            # Get fighter ID to look up DOB
            fighter_id = fighter_df["fighter_id"].iloc[0]

            # Look up fighter's DOB from raw data
            fighter_raw = self.fighter_data[
                self.fighter_data["fighter_id"] == fighter_id
            ]
            if not fighter_raw.empty and pd.notna(fighter_raw["fighter_dob"].iloc[0]):
                dob = pd.to_datetime(fighter_raw["fighter_dob"].iloc[0])
                age = (pd.Timestamp.now() - dob).days / 365.25
            else:
                age = None  # If DOB is not available

            fighter = {
                "fighter_id": fighter_id,
                "name": fighter_df["name"].iloc[0],
                "win_ratio": float(fighter_df["win_ratio"].iloc[0]),
                "recent_win_ratio": float(fighter_df["recent_win_ratio"].iloc[0]),
                "height": float(fighter_df["height"].iloc[0]),
                "weight": float(fighter_df["weight"].iloc[0]),
                "reach": float(fighter_df["reach"].iloc[0]),
                "age": age,  # Now calculated from DOB
                "stance": fighter_df["stance"].iloc[0],
                "total_fights": int(fighter_df["total_fights"].iloc[0]),
                "avg_strikes_landed": float(fighter_df["avg_strikes_landed"].iloc[0]),
                "avg_strikes_attempted": float(
                    fighter_df["avg_strikes_attempted"].iloc[0]
                ),
                "avg_strikes_accuracy": float(
                    fighter_df["avg_strikes_accuracy"].iloc[0]
                ),
                "avg_takedowns": float(fighter_df["avg_takedowns"].iloc[0]),
                "avg_takedown_accuracy": float(
                    fighter_df["avg_takedown_accuracy"].iloc[0]
                ),
                "avg_knockdowns": float(fighter_df["avg_knockdowns"].iloc[0]),
                "ko_ratio": float(fighter_df["ko_ratio"].iloc[0]),
                "avg_control_time": float(fighter_df["avg_control_time"].iloc[0]),
                "form_score": float(fighter_df["form_score"].iloc[0]),  # Add form score
            }
            return fighter
        except (KeyError, ValueError, IndexError) as e:
            print(f"Error processing fighter data for {fighter_name}: {e}")
            return None

    def load_model(self):
        """Load the trained model and scaler"""
        model_path = self.model_dir / "logistic_regression.joblib"
        scaler_path = self.model_dir / "scaler.joblib"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                "Model or scaler not found. Please train the model first."
            )

        # Load using joblib instead of XGBoost
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def get_model_features(self) -> list:
        """Get the feature names that the model was trained with"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # For scikit-learn models, get feature names from scaler
        return self.scaler.get_feature_names_out().tolist()

    def predict(self, fighter1_name: str, fighter2_name: str) -> dict:
        """Predict winner of a fight between two fighters"""

        fighter1_features = self.get_fighter_features(fighter1_name)
        fighter2_features = self.get_fighter_features(fighter2_name)

        if not fighter1_features or not fighter2_features:
            raise ValueError("Could not find features for one or both fighters")

        # Get predictions from both models
        nn_prob = self.nn_model.predict(fighter1_features)
        lr_prob = self.log_reg.predict_proba(fighter1_features)

        # Average the probabilities
        avg_prob = (nn_prob + lr_prob) / 2

        # Get form scores for display
        fighter1_form = fighter1_stats["form_score"]
        fighter2_form = fighter2_stats["form_score"]

        # Create prediction result
        result = {
            "predicted_winner": fighter1_name if avg_prob > 0.5 else fighter2_name,
            "confidence": max(avg_prob, 1 - avg_prob) * 100,
            "probabilities": {
                fighter1_name: avg_prob * 100,
                fighter2_name: (1 - avg_prob) * 100,
            },
            "form_scores": {fighter1_name: fighter1_form, fighter2_name: fighter2_form},
        }

        return result

    def predict_single_fight(self, fighter1_name: str, fighter2_name: str):
        """Make prediction for a single fight"""
        fighter1_features = self.get_fighter_features(fighter1_name)
        fighter2_features = self.get_fighter_features(fighter2_name)

        if not fighter1_features or not fighter2_features:
            raise ValueError("Could not find features for one or both fighters")

        # Get expected feature names in correct order
        expected_features = self.scaler.get_feature_names_out()

        # Calculate differences
        weight_diff = fighter1_features["weight"] - fighter2_features["weight"]
        height_diff = fighter1_features["height"] - fighter2_features["height"]
        reach_diff = fighter1_features["reach"] - fighter2_features["reach"]
        age_diff = (fighter1_features["age"] or 0) - (fighter2_features["age"] or 0)

        # Create feature dictionary matching the model's expected features
        feature_dict = {
            # Differences
            "height_diff": height_diff,
            "reach_diff": reach_diff,
            "age_diff": age_diff,
            "weight_diff": weight_diff,
            # Fighter features
            "fighter_fight_no": fighter1_features["total_fights"],
            "fighter_win_ratio": fighter1_features["win_ratio"],
            "fighter_recent_win_ratio": fighter1_features["recent_win_ratio"],
            "fighter_form_score": fighter1_features["form_score"],  # Add form score
            "fighter_strikes_landed": fighter1_features["avg_strikes_landed"],
            "fighter_strikes_attempted": fighter1_features["avg_strikes_attempted"],
            "fighter_strikes_accuracy": fighter1_features["avg_strikes_accuracy"],
            "fighter_takedowns": fighter1_features["avg_takedowns"],
            "fighter_takedown_accuracy": fighter1_features["avg_takedown_accuracy"],
            "fighter_knockdowns": fighter1_features["avg_knockdowns"],
            "fighter_ko_ratio": fighter1_features["ko_ratio"],
            "fighter_control_time": fighter1_features["avg_control_time"],
            # Opponent features
            "opponent_fight_no": fighter2_features["total_fights"],
            "opponent_win_ratio": fighter2_features["win_ratio"],
            "opponent_recent_win_ratio": fighter2_features["recent_win_ratio"],
            "opponent_form_score": fighter2_features["form_score"],  # Add form score
            "opponent_strikes_landed": fighter2_features["avg_strikes_landed"],
            "opponent_strikes_attempted": fighter2_features["avg_strikes_attempted"],
            "opponent_strikes_accuracy": fighter2_features["avg_strikes_accuracy"],
            "opponent_takedowns": fighter2_features["avg_takedowns"],
            "opponent_takedown_accuracy": fighter2_features["avg_takedown_accuracy"],
            "opponent_knockdowns": fighter2_features["avg_knockdowns"],
            "opponent_ko_ratio": fighter2_features["ko_ratio"],
            "opponent_control_time": fighter2_features["avg_control_time"],
        }

        # Create DataFrame with features in correct order
        features = pd.DataFrame(
            [{name: feature_dict.get(name, 0) for name in expected_features}]
        )

        # Scale features
        X_scaled = self.scaler.transform(features)

        # Get initial predictions from model (based on all stats)
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Adjust probabilities based on weight difference, but maintain influence of other factors
        if abs(weight_diff) > 20:  # Only adjust for significant weight differences
            if weight_diff > 0:  # Fighter 1 is heavier
                weight_boost = 0  # Base boost value
                if weight_diff > 60:
                    weight_boost = 0.60  # +15% for major mismatches
                elif weight_diff > 40:
                    weight_boost = 0.30  # +10% for significant mismatches
                elif weight_diff > 20:
                    weight_boost = 0.10  # +5% for moderate mismatches

                # Blend original probability with weight advantage
                p1_win = min(
                    0.90, probabilities[1] + (weight_boost * (1 - probabilities[1]))
                )
                probabilities = np.array([1 - p1_win, p1_win])
            else:  # Fighter 2 is heavier
                weight_diff = abs(weight_diff)
                weight_boost = 0
                if weight_diff > 60:
                    weight_boost = 0.60
                elif weight_diff > 40:
                    weight_boost = 0.30
                elif weight_diff > 20:
                    weight_boost = 0.10

                p2_win = min(
                    0.90, probabilities[0] + (weight_boost * (1 - probabilities[0]))
                )
                probabilities = np.array([p2_win, 1 - p2_win])

        # Create result dictionary with physical differences included
        result = {
            "fighter1": fighter1_name,
            "fighter2": fighter2_name,
            "probability_fighter1_wins": float(probabilities[1]),
            "probability_fighter2_wins": float(probabilities[0]),
            "predicted_winner": fighter1_name
            if probabilities[1] > 0.5
            else fighter2_name,
            "confidence": float(max(probabilities)),
            "form_scores": {
                fighter1_name: fighter1_features["form_score"],
                fighter2_name: fighter2_features["form_score"],
            },
            "height_diff": height_diff,
            "reach_diff": reach_diff,
            "age_diff": age_diff,
            "weight_diff": weight_diff,
        }

        return result

    def _determine_weight_class(self, weight_lbs):
        """Determine weight class based on weight"""
        if weight_lbs <= 125:
            return "flyweight"
        elif weight_lbs <= 135:
            return "bantamweight"
        elif weight_lbs <= 145:
            return "featherweight"
        elif weight_lbs <= 155:
            return "lightweight"
        elif weight_lbs <= 170:
            return "welterweight"
        elif weight_lbs <= 185:
            return "middleweight"
        elif weight_lbs <= 205:
            return "light_heavyweight"
        else:
            return "heavyweight"

    def _calculate_weight_penalty(self, weight_diff: float) -> float:
        """Calculate penalty factor for weight mismatches"""
        # Exponential penalty for weight differences
        return np.exp(
            abs(weight_diff) / 10.0
        )  # Adjust the divisor to control penalty strength

    def _adjust_probabilities_for_weight(
        self, probabilities: np.ndarray, weight_diff: float
    ) -> np.ndarray:
        """Adjust win probabilities based on weight difference"""
        if weight_diff > 0:  # Fighter 1 is heavier
            factor = self._calculate_weight_penalty(weight_diff)
            p1_new = min(0.90, probabilities[1] * factor)  # Cap at 90%
            return np.array([1 - p1_new, p1_new])
        else:  # Fighter 2 is heavier
            factor = self._calculate_weight_penalty(weight_diff)
            p2_new = min(0.90, probabilities[0] * factor)  # Cap at 90%
            return np.array([p2_new, 1 - p2_new])

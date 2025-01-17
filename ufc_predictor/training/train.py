import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout
from pathlib import Path
from datetime import datetime
import logging
import sys
import json
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import tensorflow as tf

class UFCModelTrainer:
    def __init__(self, features_path: str = "data/processed/features.csv", 
                 output_dir: str = "model",
                 log_dir: str = "logs"):
        # Setup paths
        self.features_path = Path(features_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_dir)
        
        # Initialize models and data
        self.scaler = StandardScaler()
        self.nn_model = None
        self.log_reg = None
        self.feature_columns = None
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Training results
        self.training_results = {'iterations': []}
        
    def setup_logging(self, log_dir: str):
        """Setup logging to both file and console"""
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.results_file = log_dir / f'results_{timestamp}.json'
        
    def train(self) -> dict:
        """Main training pipeline"""
        self.logger.info("🥊 Starting UFC model training pipeline...")
        
        # Load and prepare data
        self.logger.info("📊 Loading and preparing data...")
        self._load_and_prepare_data()
        
        # Train neural network
        self.logger.info("\n🧠 Training neural network...")
        self._train_neural_network()
        
        # Train logistic regression
        self.logger.info("\n📈 Training logistic regression...")
        self._train_logistic_regression()
        
        # Evaluate models
        self.logger.info("\n🎯 Evaluating models...")
        results = self._evaluate_models()
        
        # Save models
        self.logger.info("\n💾 Saving models...")
        self._save_models()
        
        # Save and display results
        self._save_results(results)
        self._display_results(results)
        
        self.logger.info("\n🎉 Training pipeline completed successfully!")
        return results
    
    def _load_and_prepare_data(self):
        """Load and prepare data for training"""
        # Load features
        df = pd.read_csv(self.features_path)
        
        # Define our expected feature columns based on the new preprocessing
        self.feature_columns = [
            # Physical differences
            'height_diff', 'reach_diff', 'age_diff',
            
            # Form and experience
            'fighter_form_score', 'opponent_form_score',
            'fighter_fight_no', 'opponent_fight_no',
            
            # Win ratios
            'fighter_win_ratio', 'fighter_recent_win_ratio',
            'opponent_win_ratio', 'opponent_recent_win_ratio',
            
            # Strike stats
            'fighter_strikes_landed', 'fighter_strikes_attempted', 'fighter_strikes_accuracy',
            'opponent_strikes_landed', 'opponent_strikes_attempted', 'opponent_strikes_accuracy',
            
            # Takedown stats
            'fighter_takedowns', 'fighter_takedown_accuracy',
            'opponent_takedowns', 'opponent_takedown_accuracy',
            
            # Other stats
            'fighter_knockdowns', 'fighter_ko_ratio',
            'opponent_knockdowns', 'opponent_ko_ratio',
            
            # Control time
            'fighter_control_time', 'opponent_control_time'
        ]
        
        # Verify all expected columns are present
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns in features.csv: {missing_cols}")
        
        # Check for NaN values
        nan_cols = df[self.feature_columns].isna().sum()
        if nan_cols.any():
            self.logger.warning("Found NaN values in the following columns:")
            for col, count in nan_cols[nan_cols > 0].items():
                self.logger.warning(f"{col}: {count} NaN values")
            
            self.logger.info("Filling NaN values with 0")
            df[self.feature_columns] = df[self.feature_columns].fillna(0)
        
        # Split features and target
        X = df[self.feature_columns]
        y = df['win']
        
        # Train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.logger.info(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
        self.logger.info(f"Feature count: {len(self.feature_columns)}")
        self.logger.info("\nFeatures used:")
        for col in self.feature_columns:
            self.logger.info(f"- {col}")
    
    def _train_neural_network(self):
        """Train neural network model with architecture optimized for fight prediction"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Calculate class weights to handle any imbalance
        class_counts = np.bincount(self.y_train)
        total = len(self.y_train)
        class_weights = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        
        history = model.fit(
            self.X_train, self.y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weights,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        self.nn_model = model
        self.training_results['neural_network_history'] = history.history
    
    def _train_logistic_regression(self):
        """Train logistic regression model with proper scaling"""
        # Scale features before training
        self.log_reg = LogisticRegression(
            C=0.1,  # Stronger regularization
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.log_reg.fit(self.X_train, self.y_train)
        
        # Analyze and log feature importance
        self._analyze_feature_importance()
    
    def _analyze_feature_importance(self):
        """Analyze and log feature importance from logistic regression"""
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': np.abs(self.log_reg.coef_[0])
        })
        importance = importance.sort_values('importance', ascending=False)
        
        self.logger.info("\n🔍 Feature Importance Analysis:")
        for _, row in importance.head(10).iterrows():
            self.logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        # Save feature importance plot
        plt.figure(figsize=(12, 6))
        plt.bar(importance['feature'].head(10), importance['importance'].head(10))
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()
    
    def _evaluate_models(self):
        """Evaluate both models"""
        results = {}
        
        try:
            # Neural Network evaluation
            nn_prob = self.nn_model.predict(self.X_test)
            
            # Reshape predictions if needed (squeeze extra dimensions)
            nn_prob = nn_prob.squeeze()
            
            # Check for and handle NaN values
            if np.any(np.isnan(nn_prob)):
                self.logger.warning("Found NaN values in neural network predictions. Replacing with 0.5")
                nn_prob = np.nan_to_num(nn_prob, nan=0.5)
            
            nn_pred = (nn_prob > 0.5).astype(int)
            
            # Ensure predictions match test data shape
            if nn_pred.shape != self.y_test.shape:
                self.logger.warning(f"Reshaping predictions from {nn_pred.shape} to {self.y_test.shape}")
                nn_pred = nn_pred.reshape(self.y_test.shape)
            
            # Validate against random baseline
            self._validate_against_random(nn_pred, 'neural_network')
            
            results['neural_network'] = {
                'accuracy': accuracy_score(self.y_test, nn_pred),
                'auc_roc': roc_auc_score(self.y_test, nn_prob),
                'classification_report': classification_report(self.y_test, nn_pred, output_dict=True)
            }
            
            # Plot ROC curve for neural network
            self._plot_roc_curve(self.y_test, nn_prob, 'neural_network')
            
            # Logistic Regression evaluation
            lr_prob = self.log_reg.predict_proba(self.X_test)[:, 1]
            
            # Check for and handle NaN values
            if np.any(np.isnan(lr_prob)):
                self.logger.warning("Found NaN values in logistic regression predictions. Replacing with 0.5")
                lr_prob = np.nan_to_num(lr_prob, nan=0.5)
                
            lr_pred = (lr_prob > 0.5).astype(int)
            
            # Validate against random baseline
            self._validate_against_random(lr_pred, 'logistic_regression')
            
            results['logistic_regression'] = {
                'accuracy': accuracy_score(self.y_test, lr_pred),
                'auc_roc': roc_auc_score(self.y_test, lr_prob),
                'classification_report': classification_report(self.y_test, lr_pred, output_dict=True)
            }
            
            # Plot ROC curve for logistic regression
            self._plot_roc_curve(self.y_test, lr_prob, 'logistic_regression')
            
            # Add example predictions
            self._show_example_predictions(nn_pred, 'neural_network')
            self._show_example_predictions(lr_pred, 'logistic_regression')
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            self.logger.error("Checking data for issues...")
            
            # Data validation
            self.logger.info(f"X_test shape: {self.X_test.shape}")
            self.logger.info(f"y_test shape: {self.y_test.shape}")
            self.logger.info(f"X_test NaN count: {np.isnan(self.X_test).sum()}")
            self.logger.info(f"y_test NaN count: {np.isnan(self.y_test).sum()}")
            
            # If there are NaN values in the test data, clean them
            if np.any(np.isnan(self.X_test)):
                self.logger.warning("Found NaN values in test features. Replacing with 0")
                self.X_test = np.nan_to_num(self.X_test)
                
            raise
        
        return results
    
    def _plot_roc_curve(self, y_test, y_pred_proba, model_name):
        """Plot ROC curve for a model"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.savefig(self.output_dir / f'roc_curve_{model_name}.png')
        plt.close()
    
    def _save_models(self):
        """Save trained models"""
        # Save neural network
        self.nn_model.save(self.output_dir / "nn_model.h5")
        
        # Save logistic regression and scaler
        import joblib
        joblib.dump(self.log_reg, self.output_dir / "logistic_regression.joblib")
        joblib.dump(self.scaler, self.output_dir / "scaler.joblib")
    
    def _save_results(self, results: dict):
        """Save training results to JSON"""
        self.training_results['evaluation'] = results
        with open(self.results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2)
    
    def _display_results(self, results: dict):
        """Display training results"""
        self.logger.info("\n📊 Model Performance:")
        for model_name, metrics in results.items():
            self.logger.info(f"\n{model_name.replace('_', ' ').title()}:")
            self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"ROC AUC: {metrics['auc_roc']:.4f}")
            self.logger.info("\nClassification Report:")
            self.logger.info("\n" + classification_report(self.y_test, 
                (self.nn_model.predict(self.X_test) > 0.5).astype(int) if model_name == 'neural_network'
                else self.log_reg.predict(self.X_test)))
    
    def _validate_against_random(self, predictions, model_name):
        """
        Validate model predictions against random baseline using chi-square test
        """
        # Create confusion matrix
        true_pos = np.sum((predictions == 1) & (self.y_test == 1))
        true_neg = np.sum((predictions == 0) & (self.y_test == 0))
        false_pos = np.sum((predictions == 1) & (self.y_test == 0))
        false_neg = np.sum((predictions == 0) & (self.y_test == 1))
        
        # Create contingency table
        contingency = np.array([[true_pos, false_pos],
                               [false_neg, true_neg]])
        
        # Perform chi-square test
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        # Calculate accuracy
        accuracy = (true_pos + true_neg) / len(predictions)
        random_accuracy = 0.5  # Random guessing in binary classification
        
        self.logger.info(f"\n🎲 Random Baseline Validation - {model_name}:")
        self.logger.info(f"Model Accuracy: {accuracy:.4f}")
        self.logger.info(f"Random Baseline: {random_accuracy:.4f}")
        self.logger.info(f"Chi-square statistic: {chi2:.4f}")
        self.logger.info(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            self.logger.info("✅ Model performs significantly better than random (p < 0.05)")
        else:
            self.logger.warning("⚠️ Model performance not significantly different from random")
        
        # Calculate and log effect size (Cramer's V)
        n = len(predictions)
        cramer_v = np.sqrt(chi2 / (n * 1))  # 1 is min(rows-1, cols-1)
        self.logger.info(f"Effect size (Cramer's V): {cramer_v:.4f}")
    
    def _show_example_predictions(self, predictions, model_name):
        """
        Show examples of model predictions compared to actual outcomes
        """
        # Load original feature data
        df = pd.read_csv(self.features_path)
        
        # Debug info
        self.logger.info("Available columns in features.csv:")
        self.logger.info(", ".join(df.columns))
        
        # Get the test set indices
        test_indices = df.index[df.index.isin(self.y_test.index)]
        
        # Create a comparison DataFrame - using fighter IDs if names aren't available
        comparison = pd.DataFrame({
            'Predicted': predictions,
            'Actual': self.y_test,
            'Correct': predictions == self.y_test
        })
        
        # Calculate accuracy
        accuracy = (comparison['Correct'].sum() / len(comparison)) * 100
        
        self.logger.info(f"\n🎯 Example Predictions - {model_name}")
        self.logger.info(f"Overall Accuracy: {accuracy:.2f}%")
        
        # Show some correct and incorrect predictions
        self.logger.info("\n✅ Sample Correct Predictions:")
        correct = comparison[comparison['Correct']].head(5)
        for idx, row in correct.iterrows():
            self.logger.info(
                f"Fight {idx}: "
                f"Predicted {'Win' if row['Predicted'] else 'Loss'} "
                f"(Actual: {'Win' if row['Actual'] else 'Loss'})"
            )
        
        self.logger.info("\n❌ Sample Incorrect Predictions:")
        incorrect = comparison[~comparison['Correct']].head(5)
        for idx, row in incorrect.iterrows():
            self.logger.info(
                f"Fight {idx}: "
                f"Predicted {'Win' if row['Predicted'] else 'Loss'} "
                f"(Actual: {'Win' if row['Actual'] else 'Loss'})"
            )
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime

from ufc_predictor.models.predictor import UFCPredictor
from ufc_predictor.config import PROCESSED_DATA_DIR, MODELS_DIR

def clean_ratio(value):
    """Convert 'X of Y' strings to float ratios."""
    if isinstance(value, str) and 'of' in value:
        try:
            numerator, denominator = value.split(' of ')
            return float(numerator) / float(denominator)
        except:
            return 0.0
    return value

def train_model():
    """Train the UFC prediction model."""
    # Load the data
    df = pd.read_csv(Path(PROCESSED_DATA_DIR) / 'fight_with_stats.csv')

    # Preprocessing steps
    df = df[(df['fight_no'] >= 3) & (df['fight_no_opponent'] >= 3)]  # Remove early career fights

    # Convert dates to numeric (age at fight time)
    def convert_date_to_age(date_str):
        try:
            return (datetime.now() - pd.to_datetime(date_str)).days / 365.25
        except:
            return np.nan

    df['age'] = df['dob'].apply(convert_date_to_age)
    df['age_opponent'] = df['dob_opponent'].apply(convert_date_to_age)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['stance_encoded'] = label_encoder.fit_transform(df['stance'].fillna('unknown'))
    df['stance_opponent_encoded'] = label_encoder.fit_transform(df['stance_opponent'].fillna('unknown'))

    # Clean up column names
    df.columns = df.columns.str.replace('.', '_').str.replace(' ', '_')

    # Drop unnecessary columns
    columns_to_drop = [
        'event', 'bout', 'url', 'url_opponent', 'fighter', 'fighter_opponent',
        'method', 'round_x', 'round_y', 'time', 'weightclass', 'outcome', 'outcome_opponent',
        'first', 'first_opponent', 'last', 'last_opponent', 
        'nickname', 'nickname_opponent', 'dob', 'dob_opponent',
        'stance', 'stance_opponent', 'fighter_tott', 'fighter_tott_opponent',
        'method_opponent', 'weightclass_opponent', 'round_x_opponent', 'round_y_opponent',
        'time_opponent'
    ]

    # Only drop columns that exist in the DataFrame
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_model = df.drop(existing_columns_to_drop, axis=1)

    # Clean up ratio columns
    ratio_columns = ['sig_str_', 'sig_str__%', 'td_%', 'sig_str__opponent', 'sig_str__%_opponent', 'td_%_opponent']
    for col in ratio_columns:
        if col in df_model.columns:
            df_model[col] = df_model[col].apply(clean_ratio)

    # Convert all columns to numeric, replacing non-numeric values with 0
    for col in df_model.columns:
        if col != 'win':  # Skip the target column
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)

    # Split features and target
    X = df_model.loc[:, ~df_model.columns.isin(['win'])]
    y = df_model['win']

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    # Reshape input data for LSTM
    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Define the model
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(1, X_train.shape[1])))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train_reshaped, y_train, 
        epochs=120, 
        batch_size=32, 
        validation_data=(X_test_reshaped, y_test),
        verbose=1
    )

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print('Test accuracy:', test_acc)

    # Model summary
    model.summary()

    # Save the model
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / 'model.h5')
    model.save_weights(MODELS_DIR / 'model.weights.h5')

if __name__ == "__main__":
    train_model() 
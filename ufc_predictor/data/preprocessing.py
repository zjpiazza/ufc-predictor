import pandas as pd
from pathlib import Path
from datetime import datetime
from ufc_predictor.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from ufc_predictor.schemas.raw import RawFighterStats
from ufc_predictor.schemas.processed import ProcessedFighter, FightStats, ProcessedFight

def standardize_bout_name(bout: str) -> str:
    """Standardize bout names by sorting fighter names and removing extra spaces."""
    fighters = bout.replace('  ', ' ').replace(' vs. ', ' vs ').split(' vs ')
    return ' vs. '.join(sorted(fighters)).strip()

def convert_height(height_str):
    try:
        feet, inches = height_str.replace('"', '').split("' ")
        return (int(feet) * 12) + int(inches)
    except:
        return None

def convert_reach(reach_str):
    try:
        return int(reach_str.replace('"', ''))
    except:
        return None

def convert_weight(weight_str):
    try:
        return int(weight_str.split()[0])
    except:
        return None
        
def clean_percentage(pct_str):
    """Clean percentage strings to float values."""
    try:
        if pd.isna(pct_str):
            return 0.0
        if isinstance(pct_str, str) and 'of' in pct_str:
            num, denom = map(int, pct_str.split('of'))
            return num / denom if denom != 0 else 0.0
        return float(pct_str.rstrip('%')) / 100 if isinstance(pct_str, str) else float(pct_str)
    except:
        return 0.0

def clean_measurements(df):
    """Clean physical measurements in the dataset."""
    # Clean physical measurements
    if 'height' in df.columns:
        df['height'] = df['height'].apply(convert_height)
    if 'reach' in df.columns:
        df['reach'] = df['reach'].apply(convert_reach)
    if 'weight' in df.columns:
        df['weight'] = df['weight'].apply(convert_weight)
    
    # Clean percentage columns
    percentage_cols = ['sig.str. %', 'td %']
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_percentage)
    
    return df

def parse_date(date_str):
    if pd.isna(date_str) or date_str == '--':
        return None
    try:
        # Try different date formats
        for fmt in ['%b %d, %Y', '%B %d, %Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        # If none of the formats work, try pandas' flexible parser
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return None


def clean_dates(df):
    """Clean and standardize date formats."""

    # Clean date columns
    date_columns = ['date', 'dob']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_date)
    
    return df

def analyze_data_completeness(raw_data: pd.DataFrame):
    """Print stats about data completeness for each column."""
    print("\nData Completeness Analysis:")
    print("===========================")
    total_rows = len(raw_data)
    
    stats = []
    for column in raw_data.columns:
        non_null = raw_data[column].notna().sum()
        non_zero = (raw_data[column] != 0).sum() if pd.api.types.is_numeric_dtype(raw_data[column]) else non_null
        non_unknown = (raw_data[column] != "Unknown").sum() if pd.api.types.is_string_dtype(raw_data[column]) else non_zero
        
        pct_complete = (non_null / total_rows) * 100
        pct_meaningful = (non_unknown / total_rows) * 100
        
        stats.append({
            'column': column,
            'total_rows': total_rows,
            'non_null': non_null,
            'pct_complete': pct_complete,
            'pct_meaningful': pct_meaningful
        })
    
    # Convert to DataFrame for nice display
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('pct_meaningful', ascending=False)
    
    # Print results
    for _, row in stats_df.iterrows():
        print(f"{row['column']:<30} {row['pct_complete']:>6.1f}% complete, {row['pct_meaningful']:>6.1f}% meaningful")

def load_raw_data():
    """Load and merge all raw data files."""
    # Load raw data files
    events = pd.read_csv(Path(RAW_DATA_DIR) / 'events.csv')
    fighter_details = pd.read_csv(Path(RAW_DATA_DIR) / 'fighter_details.csv')
    fight_details = pd.read_csv(Path(RAW_DATA_DIR) / 'fight_details.csv')
    fighter_stats = pd.read_csv(Path(RAW_DATA_DIR) / 'fighter_stats.csv')
    
    # Deduplicate fighter details by URL
    fighter_details = fighter_details.sort_values('date').drop_duplicates(subset=['url'], keep='last')
    
    # Create fighter name in fighter_details if not already present
    if 'fighter' not in fighter_details.columns:
        fighter_details['fighter'] = fighter_details['first'] + ' ' + fighter_details['last']
    
    # Clean measurements and dates
    fighter_stats = clean_measurements(fighter_stats)
    events = clean_dates(events)
    fighter_stats = clean_dates(fighter_stats)
    
    # First, get all fighters from fight details
    fighters_df = pd.concat([
        fight_details[['fighter1_url']].rename(columns={'fighter1_url': 'url'}),
        fight_details[['fighter2_url']].rename(columns={'fighter2_url': 'url'})
    ])
    
    # Merge fighter details and stats with the fighter URLs
    fighter_data = fighters_df.merge(
        fighter_details,
        on='url',
        how='left'
    ).merge(
        fighter_stats,
        on='url',
        how='left'
    )
    
    # Now merge with events
    merged_data = events.merge(
        fight_details,
        left_on='url',
        right_on='event_url',
        how='left'
    ).merge(
        fighter_data,
        left_on=['fighter1_url', 'fighter2_url'],
        right_on=['url', 'url'],
        how='left'
    )
    
    print(f"Loaded {len(merged_data)} total fights")
    print(f"With {len(fighter_data)} unique fighters")
    
    return merged_data

def process_training_data():
    """Process raw data into training format."""
    print("\n=== Processing Training Data ===")
    
    # Load raw data
    raw_data = load_raw_data()
    
    # Process fighters and fight stats separately
    processed_fighters = []
    processed_fight_stats = []
    
    # Track unique fighters to avoid duplicates
    seen_fighters = set()
    
    for _, row in raw_data.iterrows():
        fighter_name = row.get('fighter', 'Unknown')
        
        # Process fighter data (only once per fighter)
        if fighter_name not in seen_fighters:
            fighter_data = {
                'fighter_name': fighter_name,
                'height': row.get('height', None),
                'weight': row.get('weight', None),
                'reach': row.get('reach', None),
                'stance': row.get('stance', None),
                'dob': row.get('dob', None),
            }
            processed_fighters.append(fighter_data)
            seen_fighters.add(fighter_name)
        
        # Process fight statistics
        fight_stats = {
            'event_name': row.get('event', 'Unknown'),
            'fight_url': row.get('url', ''),
            'fighter_name': fighter_name,
            'date': row.get('date', None),
            'slpm': row.get('slpm', 0.0),
            'str_acc': row.get('str_acc', 0.0),
            'sapm': row.get('sapm', 0.0),
            'str_def': row.get('str_def', 0.0),
            'td_avg': row.get('td_avg', 0.0),
            'td_acc': row.get('td_acc', 0.0),
            'td_def': row.get('td_def', 0.0),
            'sub_avg': row.get('sub_avg', 0.0)
        }
        processed_fight_stats.append(fight_stats)
    
    # Convert to DataFrames and save
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    fighters_df = pd.DataFrame(processed_fighters)
    fighters_df.to_csv(Path(PROCESSED_DATA_DIR) / 'processed_fighters.csv', index=False)
    print(f"Saved {len(fighters_df)} processed fighters")
    
    fight_stats_df = pd.DataFrame(processed_fight_stats)
    fight_stats_df.to_csv(Path(PROCESSED_DATA_DIR) / 'processed_fight_stats.csv', index=False)
    print(f"Saved {len(fight_stats_df)} processed fight statistics")
    
    return fighters_df, fight_stats_df

def convert_raw_to_processed(raw_data: dict) -> ProcessedFight:
    """Convert raw fight data to processed format using schemas."""
    return ProcessedFight(
        event_name=raw_data.get('event', 'Unknown'),
        fight_url=raw_data.get('url', ''),
        fighter_name=raw_data.get('fighter', 'Unknown'),
        height=convert_height(raw_data.get('height', '')),
        weight=convert_weight(raw_data.get('weight', '')),
        reach=convert_reach(raw_data.get('reach', '')),
        stance=raw_data.get('stance', 'Unknown'),
        dob=parse_date(raw_data.get('dob', None)),
        slpm=float(raw_data.get('slpm', 0.0)),
        str_acc=clean_percentage(raw_data.get('str_acc', 0.0)),
        sapm=float(raw_data.get('sapm', 0.0)),
        str_def=clean_percentage(raw_data.get('str_def', 0.0)),
        td_avg=float(raw_data.get('td_avg', 0.0)),
        td_acc=clean_percentage(raw_data.get('td_acc', 0.0)),
        td_def=clean_percentage(raw_data.get('td_def', 0.0)),
        sub_avg=float(raw_data.get('sub_avg', 0.0))
    ) 
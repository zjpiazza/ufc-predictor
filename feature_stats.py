import pandas as pd
import numpy as np


features_df = pd.read_csv("/home/d3adb0y/code/UFC-predictor-app/model/df_model.csv")

print("\n📊 Feature Statistics:")
for col in features_df.select_dtypes(include=[np.number]).columns:
    stats = features_df[col].describe()
    print(f"\n{col}:")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    
    # Log distribution of values to check for uniformity
    unique_count = features_df[col].nunique()
    print(f"  Unique values: {unique_count}")
    
    # If there are very few unique values, show their distribution
    if unique_count < 10:
        value_counts = features_df[col].value_counts()
        print("  Value distribution:")
        for val, count in value_counts.items():
            print(f"    {val}: {count} ({count/len(features_df)*100:.1f}%)")
        
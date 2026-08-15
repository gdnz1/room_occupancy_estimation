"""
Feature engineering module for Room Occupancy Estimation.
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from src import config


def create_spatial_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physical mean and range aggregations across spatial sensor groups:
    - Temperature: Avg_Temp, Temp_Range
    - Light: Avg_Light, Light_Range
    - Sound: Avg_Sound, Sound_Range
    """
    df_feat = df.copy()
    
    # Temperature aggregations
    temp_matrix = df_feat[config.TEMP_COLS]
    df_feat["Avg_Temp"] = temp_matrix.mean(axis=1)
    df_feat["Temp_Range"] = temp_matrix.max(axis=1) - temp_matrix.min(axis=1)
    
    # Light aggregations
    light_matrix = df_feat[config.LIGHT_COLS]
    df_feat["Avg_Light"] = light_matrix.mean(axis=1)
    df_feat["Light_Range"] = light_matrix.max(axis=1) - light_matrix.min(axis=1)
    
    # Sound aggregations
    sound_matrix = df_feat[config.SOUND_COLS]
    df_feat["Avg_Sound"] = sound_matrix.mean(axis=1)
    df_feat["Sound_Range"] = sound_matrix.max(axis=1) - sound_matrix.min(axis=1)
    
    return df_feat


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive time-of-day and day-of-week contextual features from Date and Time strings.
    """
    df_feat = df.copy()
    if "Date" in df_feat.columns and "Time" in df_feat.columns:
        dt = pd.to_datetime(df_feat["Date"] + " " + df_feat["Time"], format="%Y/%m/%d %H:%M:%S")
        df_feat["Hour"] = dt.dt.hour
        df_feat["Minute"] = dt.dt.minute
        df_feat["DayOfWeek"] = dt.dt.dayofweek
    return df_feat


def build_full_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataframe into complete dataframe containing all raw, aggregated, and temporal features.
    """
    df_engineered = create_spatial_aggregations(df)
    df_engineered = create_temporal_features(df_engineered)
    return df_engineered


def get_feature_sets() -> Dict[str, List[str]]:
    """
    Return dictionary mapping feature set names to their respective column lists.
    """
    return {
        "Feature Set A (All Sensors)": config.FEATURE_SET_A,
        "Feature Set B (No PIR)": config.FEATURE_SET_B,
        "Feature Set C (Environmental Only)": config.FEATURE_SET_C,
        "Feature Set D (Engineered Summary)": config.FEATURE_SET_D,
        "Feature Set E (Sensors + Time)": config.FEATURE_SET_E,
    }


def prepare_and_save_processed_data() -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load raw data, engineer all features, and save processed dataframe to data/processed/.
    """
    from src.data_loader import load_raw_data
    df_raw = load_raw_data()
    df_processed = build_full_feature_dataset(df_raw)
    
    out_path = config.PROCESSED_DATA_DIR / "engineered_features.csv"
    df_processed.to_csv(out_path, index=False)
    print(f"Processed dataset successfully saved to: {out_path}")
    print(f"Dataset shape: {df_processed.shape}")
    
    feature_sets = get_feature_sets()
    print("\nConfigured Feature Sets:")
    for name, cols in feature_sets.items():
        print(f"  - {name} ({len(cols)} features): {', '.join(cols[:4])} ...")
        
    return df_processed, feature_sets


if __name__ == "__main__":
    prepare_and_save_processed_data()

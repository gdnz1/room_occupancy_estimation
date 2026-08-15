"""
Data loading and quality validation module for Room Occupancy Estimation.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src import config


def load_raw_data(file_path: str = None) -> pd.DataFrame:
    """
    Load raw CSV data from file path and parse basic structure.
    """
    path = file_path if file_path is not None else config.RAW_DATA_FILE
    df = pd.read_csv(path)
    return df


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks:
    - Shape, data types, missing values (NaN), infinite values
    - Duplicate rows
    - Descriptive statistics
    - Target class distribution
    - Date & time range and sampling frequency
    """
    # 1. Shape & columns
    n_rows, n_cols = df.shape
    columns = list(df.columns)
    
    # 2. Missing and infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    
    inf_counts = {col: int(np.isinf(df[col]).sum()) for col in numeric_cols}
    total_inf = sum(inf_counts.values())
    
    # 3. Duplicate rows
    exact_duplicates = int(df.duplicated().sum())
    
    # 4. Target distribution
    target_dist = df[config.TARGET_COL].value_counts().sort_index().to_dict()
    target_dist_pct = (df[config.TARGET_COL].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
    
    # 5. Temporal properties
    datetime_series = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y/%m/%d %H:%M:%S")
    start_time = datetime_series.min()
    end_time = datetime_series.max()
    time_diffs = datetime_series.diff().dt.total_seconds().dropna()
    median_sampling_sec = float(time_diffs.median())
    
    # 6. Physical range validation
    negative_checks = {}
    for col in config.TEMP_COLS + config.LIGHT_COLS + config.SOUND_COLS + [config.CO2_COLS[0]] + config.PIR_COLS:
        negative_checks[col] = int((df[col] < 0).sum())
        
    summary = {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "columns": columns,
        "total_missing": total_missing,
        "missing_by_column": missing_counts,
        "total_infinite": total_inf,
        "exact_duplicates": exact_duplicates,
        "target_distribution": target_dist,
        "target_distribution_pct": target_dist_pct,
        "start_time": str(start_time),
        "end_time": str(end_time),
        "sampling_median_seconds": median_sampling_sec,
        "negative_value_counts": negative_checks,
        "numeric_summary": df.describe().to_dict()
    }
    return summary


def print_quality_report(summary: Dict[str, Any]) -> None:
    """
    Print a human-readable data quality report to the console.
    """
    print("=" * 60)
    print("DATA QUALITY & INTEGRITY REPORT")
    print("=" * 60)
    print(f"Total Rows: {summary['num_rows']:,}")
    print(f"Total Columns: {summary['num_columns']}")
    print(f"Date Range: {summary['start_time']} to {summary['end_time']}")
    print(f"Median Sampling Interval: {summary['sampling_median_seconds']:.1f} seconds")
    print(f"Missing Values (NaN): {summary['total_missing']}")
    print(f"Infinite Values: {summary['total_infinite']}")
    print(f"Exact Duplicate Rows: {summary['exact_duplicates']}")
    print("-" * 60)
    print("Target Class Distribution (Room_Occupancy_Count):")
    for cls, count in summary["target_distribution"].items():
        pct = summary["target_distribution_pct"][cls]
        print(f"  Class {cls} ({cls} persons): {count:,} records ({pct:.2f}%)")
    print("-" * 60)
    print("Physical Non-Negative Checks:")
    has_unexpected_negatives = False
    for col, neg_count in summary["negative_value_counts"].items():
        if neg_count > 0:
            print(f"  [WARN] {col} has {neg_count} negative values!")
            has_unexpected_negatives = True
    if not has_unexpected_negatives:
        print("  All physical sensor columns satisfy non-negative bounds.")
    print("=" * 60)


if __name__ == "__main__":
    df = load_raw_data()
    report = inspect_data_quality(df)
    print_quality_report(report)

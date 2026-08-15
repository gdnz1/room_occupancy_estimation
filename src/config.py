"""
Configuration and constants for Room Occupancy Estimation project.
"""
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "Occupancy_Estimation.csv"

# Output paths
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

# Ensure essential output directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Random Seed for reproducibility
RANDOM_STATE = 42

# Target Variable
TARGET_COL = "Room_Occupancy_Count"
TARGET_CLASSES = [0, 1, 2, 3]

# Raw Feature Groups
DATETIME_COLS = ["Date", "Time"]
TEMP_COLS = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp"]
LIGHT_COLS = ["S1_Light", "S2_Light", "S3_Light", "S4_Light"]
SOUND_COLS = ["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]
CO2_COLS = ["S5_CO2", "S5_CO2_Slope"]
PIR_COLS = ["S6_PIR", "S7_PIR"]

# Feature Sets Definitions
FEATURE_SET_A = TEMP_COLS + LIGHT_COLS + SOUND_COLS + CO2_COLS + PIR_COLS  # All 16 sensors
FEATURE_SET_B = TEMP_COLS + LIGHT_COLS + SOUND_COLS + CO2_COLS             # No PIR (14 features)
FEATURE_SET_C = TEMP_COLS + LIGHT_COLS + SOUND_COLS + CO2_COLS             # Environmental Only
FEATURE_SET_D = [
    "Avg_Temp", "Temp_Range",
    "Avg_Light", "Light_Range",
    "Avg_Sound", "Sound_Range",
    "S5_CO2", "S5_CO2_Slope",
    "S6_PIR", "S7_PIR"
]                                                                          # Engineered Summary (10 features)
FEATURE_SET_E = FEATURE_SET_A + ["Hour", "Minute", "DayOfWeek"]             # Sensors + Time (19 features)

# Temporal Split Ratio
TRAIN_SPLIT_RATIO = 0.80

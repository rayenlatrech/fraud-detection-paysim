from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "PS_20174392719_1491204439457_log.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "paysim_processed.parquet"

MODEL_DIR = PROJECT_ROOT / "models"
MAIN_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"

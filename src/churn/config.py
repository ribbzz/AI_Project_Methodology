"""Central configuration: paths, column groups and shared constants.

Every script imports its paths from here so that no filename is hard-coded
twice in the project.
"""

from pathlib import Path

# --------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "E_Commerce_Dataset.xlsx"
RAW_SHEET = "E Comm"

CLEAN_FILE = INTERIM_DIR / "clean.parquet"
TRAIN_FILE = PROCESSED_DIR / "train.parquet"
TEST_FILE = PROCESSED_DIR / "test.parquet"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
PREDICTIONS_FILE = REPORTS_DIR / "predictions.csv"

#: MLflow tracking store and artefact root, pinned to the project root so that
#: every script writes to the same place whatever directory it is run from.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

for _directory in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- constants
TARGET = "Churn"
ID_COLUMN = "CustomerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

EXPERIMENT_NAME = "ecommerce-churn"
REGISTERED_MODEL_NAME = "ecommerce-churn-classifier"

#: Probability thresholds used to turn a score into an actionable risk band.
RISK_BANDS = ((0.60, "HIGH"), (0.35, "MEDIUM"))

# ------------------------------------------------------------- column groups
NUMERIC_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

#: Features created in :mod:`churn.features.build_features`.
ENGINEERED_FEATURES = [
    "OrdersPerTenure",
    "CouponRate",
    "CashbackPerOrder",
    "IsNewCustomer",
    "RecencyRatio",
]

#: Inconsistent category spellings in the raw file, mapped to a single label.
CATEGORY_FIXES = {
    "PreferredLoginDevice": {"Mobile Phone": "Phone"},
    "PreferredPaymentMode": {
        "COD": "Cash on Delivery",
        "CC": "Credit Card",
    },
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}

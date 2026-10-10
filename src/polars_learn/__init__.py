"""Native Polars feature engineering and schema validation for scikit-learn."""

from .exceptions import (
    FeatureEngineeringError,
    PandasConversionWarning,
    PolarsLearnError,
    SchemaValidationError,
)
from .feature_engineering import FeatureEngineer
from .feature_spec import ColumnSelector, FeatureSpec
from .schema_validation import SchemaSpec, SchemaValidator

__version__ = "0.1.0"

__all__ = [
    "ColumnSelector",
    "FeatureEngineer",
    "FeatureSpec",
    "SchemaSpec",
    "SchemaValidator",
    "PolarsLearnError",
    "SchemaValidationError",
    "FeatureEngineeringError",
    "PandasConversionWarning",
]

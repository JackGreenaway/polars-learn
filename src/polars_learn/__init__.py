"""Native Polars feature engineering and schema validation for scikit-learn."""

from .exceptions import (
    FeatureEngineeringError,
    PandasConversionWarning,
    PolarsLearnError,
    SchemaValidationError,
)
from .feature_engineering import FeatureEngineer
from .feature_spec import FeatureSpec
from .schema_validation import SchemaValidator

__version__ = "0.1.0"

__all__ = [
    "FeatureEngineer",
    "FeatureSpec",
    "SchemaValidator",
    "PolarsLearnError",
    "SchemaValidationError",
    "FeatureEngineeringError",
    "PandasConversionWarning",
]

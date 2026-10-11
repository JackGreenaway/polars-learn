"""Feature engineering and schema validation for native Polars input.

ColumnSelector, FeaturePassthrough, SchemaSpec, SchemaValidator, and FeatureEngineer
accept eager DataFrames and LazyFrames. Default output preserves input kind;
set_output can force eager or lazy Polars returns, or pandas compatibility output.
FeatureSpec describes an operation
for use by FeatureEngineer. All six classes accept verbosity=0, 1, or 2.
"""

from .exceptions import (
    FeatureEngineeringError,
    InconsistentVersionWarning,
    PandasConversionWarning,
    PolarsLearnError,
    SchemaValidationError,
    SerializationError,
)
from .feature_engineering import FeatureEngineer
from .feature_spec import ColumnSelector, FeatureSpec
from .passthrough import FeaturePassthrough
from .schema_validation import SchemaSpec, SchemaValidator

__version__ = "0.1.3"

__all__ = [
    "ColumnSelector",
    "FeaturePassthrough",
    "FeatureEngineer",
    "FeatureSpec",
    "SchemaSpec",
    "SchemaValidator",
    "PolarsLearnError",
    "SchemaValidationError",
    "FeatureEngineeringError",
    "PandasConversionWarning",
    "InconsistentVersionWarning",
    "SerializationError",
]

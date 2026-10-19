"""Keep Polars data preparation inside scikit-learn-style model pipelines.

Define native expressions with FeatureSpec and apply them in entry order with
FeatureEngineer. Use SchemaSpec for a declared input contract, SchemaValidator
for a learned training contract, ValueValidator for row-level rules,
ColumnSelector for selection/renaming, and FeaturePassthrough for unchanged
selected columns. Fit these transformers on training input, then transform new
observations with the fitted configuration.

Polars performs preparation for eager DataFrames, LazyFrames, and explicitly
allowed pandas inputs. Pandas input is converted with convert_pandas=True;
set_output(transform="pandas") independently requests pandas output for
compatibility. Install polars-learn[pandas] for pandas/PyArrow dependencies.
Prefer native Polars for speed and to avoid conversion overhead. Default output
is Polars, lazy inputs stay deferred unless eager output is requested, and
original input frames are never modified. All seven classes accept verbosity
levels 0 (silent), 1 (progress), and 2 (details).
"""

from .exceptions import (
    FeatureEngineeringError,
    InconsistentVersionWarning,
    PandasConversionWarning,
    PolarsLearnError,
    SchemaValidationError,
    SerializationError,
    ValueValidationError,
)
from .feature_engineering import FeatureEngineer
from .feature_spec import ColumnSelector, FeatureSpec
from .passthrough import FeaturePassthrough
from .schema_validation import SchemaSpec, SchemaValidator
from .value_validation import ValueValidator

__version__ = "0.1.6"

__all__ = [
    "ColumnSelector",
    "FeaturePassthrough",
    "FeatureEngineer",
    "FeatureSpec",
    "SchemaSpec",
    "SchemaValidator",
    "ValueValidator",
    "PolarsLearnError",
    "SchemaValidationError",
    "FeatureEngineeringError",
    "PandasConversionWarning",
    "InconsistentVersionWarning",
    "SerializationError",
    "ValueValidationError",
]

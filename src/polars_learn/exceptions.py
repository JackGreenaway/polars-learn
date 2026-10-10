"""Public errors and warnings."""


class PolarsLearnError(Exception):
    """Base package exception."""


class SchemaValidationError(PolarsLearnError):
    """Raised when input schema validation fails."""


class FeatureEngineeringError(PolarsLearnError):
    """Raised when a feature transformation fails."""


class PandasConversionWarning(UserWarning):
    """Warning emitted when pandas input is converted."""

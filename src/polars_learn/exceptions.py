"""Public errors and warnings."""


class PolarsLearnError(Exception):
    """Base of SchemaValidationError and FeatureEngineeringError.

    Configuration TypeError/ValueError, scikit-learn NotFittedError, and native
    Polars/source exceptions are outside this hierarchy.
    """


class SchemaValidationError(PolarsLearnError):
    """Raised by SchemaValidator for learned schema or cast-policy failures.

    Eager cast errors and Polars errors during explicitly requested collection
    preserve their causes. Lazy lossless integer-to-float checks can raise this
    at execution; permissive lazy cast failures can instead be native Polars
    errors. SchemaSpec uses ordinary ValueError for its explicit contract.
    """


class FeatureEngineeringError(PolarsLearnError):
    """Raised when feature planning, schema checks, or eager execution fail.

    Wrapped expression/execution errors preserve their causes. Value-dependent
    failures in a returned lazy plan can be native Polars exceptions at query
    execution rather than this package exception.
    """


class PandasConversionWarning(UserWarning):
    """Warning emitted on enabled pandas input conversion.

    Controlled by warn_on_conversion, independently of verbosity. This is a
    UserWarning, not a PolarsLearnError.
    """

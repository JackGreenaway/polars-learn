"""Public errors and warnings."""


class PolarsLearnError(Exception):
    """Base of package-specific processing and persistence errors.

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


class SerializationError(PolarsLearnError):
    """Signal that a model component could not be saved or restored.

    Raised when expression serialization or saved-state restoration fails,
    including incompatible state after a dependency change. The original
    exception is retained as the cause so callers can inspect the failure.
    Catch this when reporting artifact loading/saving errors; a version
    mismatch alone emits InconsistentVersionWarning rather than this error.
    """


class InconsistentVersionWarning(UserWarning):
    """A saved class is being restored with a different dependency version.

    Attributes identify the class, dependency, original_version, and current_version.
    Loading is attempted unless the caller promotes this warning to an error.
    Silence accepted mismatches with warnings.simplefilter("ignore",
    InconsistentVersionWarning), preferably inside warnings.catch_warnings().
    """

    def __init__(
        self,
        class_name: str,
        dependency: str,
        original_version: str,
        current_version: str,
    ) -> None:
        self.class_name = class_name
        self.dependency = dependency
        self.original_version = original_version
        self.current_version = current_version
        super().__init__(class_name, dependency, original_version, current_version)

    def __str__(self) -> str:
        return (
            f"Loading {self.class_name} saved with {self.dependency} "
            f"{self.original_version} using {self.current_version}. Restoration will "
            "be attempted, but compatibility and identical results are not guaranteed. "
            "Validate the loaded object or use its original environment."
        )

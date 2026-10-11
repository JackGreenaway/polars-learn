# Errors and warnings

## polars_learn.PolarsLearnError

Base of package-specific processing and persistence errors.

Configuration TypeError/ValueError, scikit-learn NotFittedError, and native
Polars/source exceptions are outside this hierarchy.

## polars_learn.SchemaValidationError

Raised by SchemaValidator for learned schema or cast-policy failures.

Eager cast errors and Polars errors during explicitly requested collection
preserve their causes. Lazy lossless integer-to-float checks can raise this
at execution; permissive lazy cast failures can instead be native Polars
errors. SchemaSpec uses ordinary ValueError for its explicit contract.

## polars_learn.FeatureEngineeringError

Raised when feature planning, schema checks, or eager execution fail.

Wrapped expression/execution errors preserve their causes. Value-dependent
failures in a returned lazy plan can be native Polars exceptions at query
execution rather than this package exception.

## polars_learn.PandasConversionWarning

Warning emitted on enabled pandas input conversion.

Controlled by warn_on_conversion, independently of verbosity. This is a
UserWarning, not a PolarsLearnError.

## polars_learn.InconsistentVersionWarning

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/exceptions.py#L50">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">InconsistentVersionWarning</span>(
    class_name,
    dependency,
    original_version,
    current_version,
)</code></pre></div>

A saved class is being restored with a different dependency version.

Attributes identify the class, dependency, original_version, and current_version.
Loading is attempted unless the caller promotes this warning to an error.
Silence accepted mismatches with warnings.simplefilter("ignore",
InconsistentVersionWarning), preferably inside warnings.catch_warnings().

## polars_learn.SerializationError

Signal that a model component could not be saved or restored.

Raised when expression serialization or saved-state restoration fails,
including incompatible state after a dependency change. The original
exception is retained as the cause so callers can inspect the failure.
Catch this when reporting artifact loading/saving errors; a version
mismatch alone emits InconsistentVersionWarning rather than this error.

"""Learned schema validation and explicit, predictable type normalisation."""

from typing import Self

import numpy as np
import polars as pl
from sklearn.utils.validation import (
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)

from ._utils import NativeTransformer, frame_schema, to_polars
from .exceptions import SchemaValidationError
from .types import ExtraColumnPolicy, Frame, OutputFrame, Schema, ValidationMode


class SchemaSpec(NativeTransformer, auto_wrap_output_keys=None):
    """Describe a fixed contract for Polars DataFrames and LazyFrames.

    Use this transformer when your application already defines the required
    column names and dtypes. Both ``fit`` and ``transform`` check that explicit
    contract; required dtypes must match exactly and are never cast.
    Output places required columns in schema-key order and handles additional
    columns according to the extra-column options. ``fit`` records the output
    layout, including preserved extras, which later transformations must match.
    Choose ``SchemaValidator`` instead to learn the contract from training data
    or permit configured dtype conversions. Neither class checks business
    rules such as allowed value ranges or required non-null values.

    Polars is the execution engine. Native DataFrames and LazyFrames are
    accepted; pandas input requires ``convert_pandas=True`` and is converted
    behind the scenes. Output is native Polars by default, including an eager
    Polars frame for converted pandas input. Use
    ``set_output(transform="pandas")`` for compatibility output; install
    ``polars-learn[pandas]`` for pandas and PyArrow. Lazy output defers row
    execution, while eager Polars or pandas output collects during transform.
    Prefer native Polars input/output for speed and to avoid conversion overhead.

    Parameters
    ----------
    schema : dict of str to polars.DataType
        Required input column names and dtypes in order.
    allow_extra_columns : bool, default=False
        Whether additional, non-required columns are accepted.
    enforce_column_order : bool, default=True
        Require the relative order of required columns to match schema key order.
    extra_columns : {"preserve", "drop"}, default="preserve"
        If extra columns are permitted, whether to keep or drop them in output.
    convert_pandas : bool, default=False
        Allow pandas DataFrames and convert them to Polars. Requires pandas;
        some dtypes also require PyArrow.
    verbosity : {0, 1, 2}, default=0
        Print progress at 1 and schema/operation details at 2; 0 is silent.
    warn_on_conversion : bool, default=True
        Emit ``PandasConversionWarning`` when converting pandas input.

    Attributes
    ----------
    schema_ : dict of str to polars.DataType
        Copy of the required explicit schema recorded by fit.
    feature_names_in_ : ndarray of str, dtype=object
        All training input column names in input order, including extras.
    n_features_in_ : int
        Number of all training input columns.
    output_features_ : ndarray of str, dtype=object
        Fitted output names, including preserved training extras.

    Raises
    ------
    TypeError
        If schema keys/values or common options have invalid types.
    ValueError
        If schema is empty or not a dict, or an option has an invalid value.

    See Also
    --------
    SchemaValidator : Learn required dtypes and allow inference-only extra names.

    Notes
    -----
    DataFrames return DataFrames and LazyFrames return unevaluated LazyFrames
    by default. Use set_output to force eager, lazy, or pandas output.
    Required dtypes must match exactly; there is no casting. Required output
    columns follow schema key order, then preserved extras follow input order.
    Transform requires fitting and exact fitted output names/order, including
    preserved extras. Extra dtypes are not validated.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import SchemaSpec
    >>> X = pl.DataFrame({"extra": [9], "x": [1]})
    >>> contract = SchemaSpec(
    ...     {"x": pl.Int64}, allow_extra_columns=True, extra_columns="drop"
    ... )
    >>> contract.fit_transform(X.lazy()).collect().columns
    ['x']
    """

    def __init__(
        self,
        schema: Schema,
        *,
        allow_extra_columns: bool = False,
        enforce_column_order: bool = True,
        extra_columns: ExtraColumnPolicy = "preserve",
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
        verbosity: int = 0,
    ) -> None:
        self.schema = schema
        self.allow_extra_columns = allow_extra_columns
        self.enforce_column_order = enforce_column_order
        self.extra_columns = extra_columns
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion
        self.verbosity = verbosity
        self._validate_options()
        self._validate_schema()

    def _validate_schema(self) -> None:
        if not isinstance(self.schema, dict) or not self.schema:
            raise ValueError(
                "schema must be a nonempty dict of column names to dtypes."
            )
        if any(not isinstance(column, str) or not column for column in self.schema):
            raise TypeError("schema keys must be nonempty strings.")
        for dtype in self.schema.values():
            valid = isinstance(dtype, pl.DataType) or (
                isinstance(dtype, type) and issubclass(dtype, pl.DataType)
            )
            if not valid:
                raise TypeError("schema values must be Polars dtypes.")
        if self.extra_columns not in ("preserve", "drop"):
            raise ValueError("extra_columns must be 'preserve' or 'drop'.")

    def _input(self, X: object) -> Frame:
        self._validate_options()
        self._validate_schema()
        self._report("Validating input")
        frame = to_polars(
            X,
            convert_pandas=self.convert_pandas,
            warn_on_conversion=self.warn_on_conversion,
        )
        self._report_frame(frame, stage="Input")
        return frame

    def fit(self, X: object, y: object = None) -> Self:
        """Validate the explicit contract and record fitted output names.

        Call before transforming to verify the declared contract and record the
        output layout. Checks required names, exact dtypes, ordering, and allowed
        extras, then stores the explicit schema and output names. The schema comes
        from the constructor, not from inferred training dtypes. Preserved training
        extras become part of the output-name contract. No values are cast.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Input with all required columns and exact required dtypes. Extra
            columns and ordering follow the constructor's policies. Pandas
            requires conversion enabled.
        y : object, default=None
            Ignored; accepted for scikit-learn compatibility.

        Returns
        -------
        self : SchemaSpec
            Fitted contract. The provided schema is checked, not inferred anew.

        Raises
        ------
        TypeError
            If input, option types, or schema keys/values are unsupported.
        ValueError
            If the schema or input contract is invalid.

        Notes
        -----
        Lazy input is checked through schema metadata without collecting rows.
        Fitted output names include preserved training extras. No casts occur.
        """
        frame = self._input(X)
        input_schema = frame_schema(frame)
        output_names = self._output_names(frame)
        self.schema_ = dict(self.schema)
        self.feature_names_in_ = np.asarray(list(input_schema), dtype=object)
        self.n_features_in_ = len(input_schema)
        self.output_features_ = np.asarray(output_names, dtype=object)
        self._report(f"Fit complete; output columns: {self.output_features_.tolist()}")
        return self

    def fit_transform(
        self, X: object, y: object = None, **fit_params: object
    ) -> OutputFrame:
        """Fit and transform the same input using the configured return kind.

        Convenience method for preparing training data: calls ``fit(X, y)`` and
        then ``transform(X)`` on the same input. This updates fitted state and
        returns the prepared frame. The target is ignored and extra fit parameters
        are unsupported. See ``fit`` for the class-specific learned contract and
        ``transform`` for validation and output behavior.

        Output follows ``set_output``: native Polars by default, eager Polars for
        ``"polars"``, a LazyFrame for ``"polars-lazy"``, or a pandas DataFrame for
        ``"pandas"``. Eager Polars and pandas output collect lazy rows. Pandas output
        uses Polars conversion defaults and may change dtype/null representation;
        missing pandas or PyArrow raises ``ImportError`` at conversion. Pandas input
        requires ``convert_pandas=True`` independently of the output choice.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Input accepted by fit and transform. Pandas requires conversion enabled.
        y : object, default=None
            Ignored; accepted for scikit-learn compatibility.
        **fit_params : dict
            Extra fit parameters are unsupported and raise TypeError.

        Returns
        -------
        X_transformed : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Transformed input. Default preserves input kind; set_output can
            force an eager DataFrame or LazyFrame. Converted pandas is eager.

        See Also
        --------
        fit : Validate input and record fitted state.
        transform : Apply the fitted transformation and output configuration.
        """
        if fit_params:
            raise TypeError(
                f"Additional fit parameters are unsupported: {sorted(fit_params)}."
            )
        return self.fit(X, y).transform(X)

    def _validate(self, frame: Frame) -> Frame:
        output_names = self._output_names(frame)
        return frame.select(output_names)

    def _output_names(self, frame: Frame) -> list[str]:
        """Validate the frame contract and return output names from its schema."""
        self._validate_schema()
        input_schema = frame_schema(frame)
        expected_names = list(self.schema)
        missing = [column for column in expected_names if column not in input_schema]
        if missing:
            raise ValueError(f"Missing required columns: {missing}.")
        extras = [column for column in input_schema if column not in self.schema]
        if extras and not self.allow_extra_columns:
            raise ValueError(f"Unexpected columns: {extras}.")
        present = [column for column in input_schema if column in self.schema]
        if self.enforce_column_order and present != expected_names:
            raise ValueError(
                f"Column order differs. Expected {expected_names}, received {present}."
            )
        for column, expected_dtype in self.schema.items():
            if column not in input_schema:
                continue
            got = input_schema[column]
            if got != expected_dtype:
                raise ValueError(
                    f"Column {column!r} has dtype {got}, expected {expected_dtype}."
                )
        if self.extra_columns == "drop":
            return expected_names
        if self.allow_extra_columns:
            return expected_names + extras
        return expected_names

    def transform(self, X: object) -> OutputFrame:
        """Check exact required dtypes and return the fitted output columns.

        Call after fitting to enforce the application's fixed contract on new data.
        Checks required names, exact dtypes, configured order, and extra-column
        policies, then selects required columns in schema-key order. The output
        names and order must match fitting, including preserved extras. No casting
        occurs and the original input is never modified.

        Output follows ``set_output``: native Polars by default, eager Polars for
        ``"polars"``, a LazyFrame for ``"polars-lazy"``, or a pandas DataFrame for
        ``"pandas"``. Eager Polars and pandas output collect lazy rows. Pandas output
        uses Polars conversion defaults and may change dtype/null representation;
        missing pandas or PyArrow raises ``ImportError`` at conversion. Pandas input
        requires ``convert_pandas=True`` independently of the output choice.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Input satisfying the explicit required schema and configured
            extra-column/order policies. Pandas requires conversion enabled.

        Returns
        -------
        X_validated : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Required columns in schema order, then preserved extras in input
            order. Default preserves input kind; set_output can force eager, lazy,
            or pandas output.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If fit has not completed successfully.
        TypeError
            If input or parameter types are unsupported.
        ValueError
            If the contract fails, configuration is invalid, or resulting output
            names/order differ from fitting, including preserved extra names.

        Notes
        -----
        No casting occurs; extra dtypes are not checked. Lazy validation uses
        metadata. Explicit eager output collects the result; source/collection
        errors are not wrapped.
        """
        check_is_fitted(self, "schema_")
        frame = self._input(X)
        result = self._validate(frame)
        if list(frame_schema(result)) != self.output_features_.tolist():
            raise ValueError(
                "Output columns differ from the fitted schema. "
                f"Expected {self.output_features_.tolist()}, "
                f"received {list(frame_schema(result))}."
            )
        return self._finish(result)


_INTEGER_RANGES: dict[pl.DataType | type[pl.DataType], tuple[int, int]] = {
    pl.Int8: (-(2**7), 2**7 - 1),
    pl.Int16: (-(2**15), 2**15 - 1),
    pl.Int32: (-(2**31), 2**31 - 1),
    pl.Int64: (-(2**63), 2**63 - 1),
    pl.UInt8: (0, 2**8 - 1),
    pl.UInt16: (0, 2**16 - 1),
    pl.UInt32: (0, 2**32 - 1),
    pl.UInt64: (0, 2**64 - 1),
}


def _widening_policy(source: pl.DataType, target: pl.DataType) -> bool | None:
    """Return a schema-only decision, or None if values must be checked."""
    if source in _INTEGER_RANGES and target in _INTEGER_RANGES:
        source_min, source_max = _INTEGER_RANGES[source]
        target_min, target_max = _INTEGER_RANGES[target]
        return target_min <= source_min and source_max <= target_max
    if source == pl.Float32 and target == pl.Float64:
        return True
    if source in _INTEGER_RANGES and target == pl.Float64:
        return None
    return False


def _safe_widen(series: pl.Series, target: pl.DataType) -> bool:
    """Return whether this series satisfies the lossless widening policy."""
    decision = _widening_policy(series.dtype, target)
    if decision is not None:
        return decision
    try:
        restored = series.cast(pl.Float64).cast(series.dtype, strict=True)
        return bool((series == restored).fill_null(True).all())
    except pl.exceptions.PolarsError:
        return False


def _lossless_float_cast(series: pl.Series) -> pl.Series:
    """Check value-dependent widening at lazy execution time."""
    if not _safe_widen(series, pl.Float64()):
        raise SchemaValidationError(
            f"Column {series.name!r} cannot be losslessly widened to Float64 "
            "in semi-strict mode."
        )
    return series.cast(pl.Float64, strict=True)


class SchemaValidator(NativeTransformer, auto_wrap_output_keys=None):
    """Learn a schema and validate subsequent Polars DataFrames or LazyFrames.

    Use this transformer when training data defines your model's input contract.
    ``fit`` records every training column's name, order, and Polars dtype;
    ``transform`` checks new data against that learned schema and handles
    permitted casts, extra columns, and ordering according to your options.
    Strict mode rejects dtype changes, semi-strict allows defined lossless
    widening, and permissive attempts Polars casts that may lose information.
    Missing required columns always fail. This validates structure and dtype
    compatibility, not business meaning, nullability, or feature distributions.
    Choose ``SchemaSpec`` when the required schema is already known.

    Polars is the execution engine. Native DataFrames and LazyFrames are
    accepted; pandas input requires ``convert_pandas=True`` and is converted
    behind the scenes. Output is native Polars by default, including an eager
    Polars frame for converted pandas input. Use
    ``set_output(transform="pandas")`` for compatibility output; install
    ``polars-learn[pandas]`` for pandas and PyArrow. Lazy output defers row
    execution, while eager Polars or pandas output collects during transform.
    Prefer native Polars input/output for speed and to avoid conversion overhead.

    Parameters
    ----------
    mode : {"strict", "semi-strict", "permissive"}, default="strict"
        Validation policy. Strict mode requires identical dtypes. Semi-strict
        allows only the documented lossless widening policy. Permissive mode
        attempts Polars casts with strict=True. Successful casts can be lossy;
        failure occurs immediately for eager input or at lazy execution.
    allow_extra_columns : bool, default=False
        Allow columns not present during fitting. Missing required columns
        always raise, regardless of the validation mode.
    enforce_column_order : bool, default=True
        Require the relative order of required columns to match training.
        If false, output uses fitted column order followed by preserved extras.
    extra_columns : {"preserve", "drop"}, default="preserve"
        Treatment of permitted extras. This option does not itself permit
        extra columns; ``allow_extra_columns`` must also be true.
    convert_pandas : bool, default=False
        Allow pandas DataFrames and convert them to Polars. Requires pandas;
        some dtypes also require PyArrow.
    verbosity : {0, 1, 2}, default=0
        Print progress at 1 and schema/operation details at 2; 0 is silent.
    warn_on_conversion : bool, default=True
        Emit ``PandasConversionWarning`` when converting pandas input.

    Attributes
    ----------
    schema_ : dict of str to polars.DataType
        Learned required columns and dtypes, in training order.
    n_features_in_ : int
        Number of training columns.
    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=object
        Required input names in training order.
    output_features_ : ndarray of shape (n_features_in_,), dtype=object
        Fitted required names. Inference-only extras are not included.

    Raises
    ------
    TypeError
        If common options have invalid types.
    ValueError
        If mode, extra_columns, or verbosity is invalid.

    See Also
    --------
    FeatureEngineer : Engineer features after validating raw inputs.

    Notes
    -----
    Semi-strict allows integer widening when the source dtype's entire range
    fits inside the target dtype, Float32 to Float64, and integer to Float64
    only when every non-null value round-trips exactly. It rejects string to
    numeric conversion, narrowing, and Null dtype to another dtype. Null values
    in compatible columns are preserved; nullability constraints are not learned.
    Fit always learns a fresh schema; transform always validates against it.
    Input data and schemas owned by other transformers are never mutated.
    Lazy inputs are validated using metadata without collecting rows. Cast
    failures and value-dependent integer-to-float checks run at collection.
    Explicit eager output wraps Polars cast errors in SchemaValidationError;
    collecting a returned LazyFrame can raise native Polars exceptions.
    Default output preserves input kind. set_output can force "polars" (eager)
    or "polars-lazy" output, or pandas output for compatibility.
    With enforced order and preserved extras, output
    keeps input order; disabling order enforcement puts required columns first.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import SchemaValidator
    >>> train = pl.DataFrame({"x": pl.Series([1], dtype=pl.Int64)})
    >>> test = pl.DataFrame({"x": pl.Series([2], dtype=pl.Int32)})
    >>> validator = SchemaValidator(mode="semi-strict").fit(train)
    >>> validator.transform(test).schema["x"]
    Int64
    """

    def __init__(
        self,
        mode: ValidationMode = "strict",
        *,
        allow_extra_columns: bool = False,
        enforce_column_order: bool = True,
        extra_columns: ExtraColumnPolicy = "preserve",
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
        verbosity: int = 0,
    ) -> None:
        self.mode = mode
        self.allow_extra_columns = allow_extra_columns
        self.enforce_column_order = enforce_column_order
        self.extra_columns = extra_columns
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion
        self.verbosity = verbosity
        self._validate_options()
        self._mode()

    def _mode(self) -> ValidationMode:
        if self.mode not in ("strict", "semi-strict", "permissive"):
            raise ValueError("mode must be 'strict', 'semi-strict', or 'permissive'.")
        if self.extra_columns not in ("preserve", "drop"):
            raise ValueError("extra_columns must be 'preserve' or 'drop'.")
        return self.mode

    def _input(self, X: object) -> Frame:
        self._validate_options()
        self._report("Validating input")
        frame = to_polars(
            X,
            convert_pandas=self.convert_pandas,
            warn_on_conversion=self.warn_on_conversion,
        )
        self._report_frame(frame, stage="Input")
        return frame

    def fit(self, X: object, y: object = None) -> Self:
        """Learn column names, order, and dtypes without changing the input.

        Call on training data to establish the schema used by later transforms.
        Reads column metadata and records every input column as required, in its
        training order. It does not cast values, learn nullability, or evaluate lazy
        rows. A successful refit replaces the previous learned contract; ``y`` is
        ignored because validation does not learn from the target.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Training data whose schema becomes the required schema. All input
            columns become required. Pandas requires ``convert_pandas=True``.
            Lazy input uses metadata without evaluating rows.
        y : object, default=None
            Ignored. Present for scikit-learn estimator compatibility.

        Returns
        -------
        self : SchemaValidator
            Fitted validator with a newly learned schema.

        Raises
        ------
        TypeError
            If the input type is unsupported or pandas conversion is disabled.
        ValueError
            If validation configuration is invalid.
        """
        self._mode()
        frame = self._input(X)
        input_schema = frame_schema(frame)
        self.schema_: Schema = input_schema
        self.feature_names_in_ = np.asarray(list(input_schema), dtype=object)
        self.n_features_in_ = len(input_schema)
        self.output_features_ = self.feature_names_in_.copy()
        self._report(f"Fit complete; output columns: {self.output_features_.tolist()}")
        return self

    def fit_transform(
        self, X: object, y: object = None, **fit_params: object
    ) -> OutputFrame:
        """Fit to training data and return its transformed representation.

        Convenience method for preparing training data: calls ``fit(X, y)`` and
        then ``transform(X)`` on the same input. This updates fitted state and
        returns the prepared frame. The target is ignored and extra fit parameters
        are unsupported. See ``fit`` for the class-specific learned contract and
        ``transform`` for validation and output behavior.

        Output follows ``set_output``: native Polars by default, eager Polars for
        ``"polars"``, a LazyFrame for ``"polars-lazy"``, or a pandas DataFrame for
        ``"pandas"``. Eager Polars and pandas output collect lazy rows. Pandas output
        uses Polars conversion defaults and may change dtype/null representation;
        missing pandas or PyArrow raises ``ImportError`` at conversion. Pandas input
        requires ``convert_pandas=True`` independently of the output choice.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Training data. Pandas input requires ``convert_pandas=True``.
            Lazy input is checked through metadata without evaluating rows.
        y : object, default=None
            Ignored. Present for scikit-learn estimator compatibility.
        **fit_params : dict
            Accepted for the scikit-learn method signature. Additional fit
            parameters are unsupported and raise ``TypeError`` when supplied.

        Returns
        -------
        X_transformed : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Result of fitting and transforming the same input. Default preserves
            input kind; set_output can force eager, lazy, or pandas output.

        See Also
        --------
        fit : Learn the fitted schema and feature names.
        transform : Apply the fitted transformer and its validation policies.
        """
        if fit_params:
            raise TypeError(
                f"Additional fit parameters are unsupported: {sorted(fit_params)}."
            )
        return self.fit(X, y).transform(X)

    def _collect(self, result: pl.LazyFrame) -> pl.DataFrame:
        try:
            return result.collect()
        except pl.exceptions.PolarsError as exc:
            raise SchemaValidationError(
                f"Lazy schema validation failed: {exc}"
            ) from exc

    def transform(self, X: object) -> OutputFrame:
        """Validate the learned schema and normalise permitted differences.

        Call after fitting to validate new observations before feature engineering
        or prediction. Checks all required columns against ``schema_``, applies the
        chosen dtype policy, and handles permitted extras and ordering. The fitted
        schema remains unchanged. Structural errors fail during this call; lazy
        value-dependent cast checks run when the resulting query is collected.
        The original input is never modified.

        Output follows ``set_output``: native Polars by default, eager Polars for
        ``"polars"``, a LazyFrame for ``"polars-lazy"``, or a pandas DataFrame for
        ``"pandas"``. Eager Polars and pandas output collect lazy rows. Pandas output
        uses Polars conversion defaults and may change dtype/null representation;
        missing pandas or PyArrow raises ``ImportError`` at conversion. Pandas input
        requires ``convert_pandas=True`` independently of the output choice.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Input containing all fitted required columns. Additional columns
            must be explicitly permitted. Pandas requires conversion enabled.

        Returns
        -------
        X_validated : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Validated input with any configured casts, ordering, and extra
            column policy applied. Default preserves input kind; set_output can
            force the return container. The original input is never mutated.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit`` has not completed successfully.
        TypeError
            If input or options are unsupported, or pandas conversion is disabled.
        ValueError
            If validation configuration is invalid.
        SchemaValidationError
            If required columns are missing, extras or order are disallowed,
            dtypes are incompatible, or a cast fails. Multiple mismatches are
            reported together, with eager cast exceptions preserved as the cause.

        Notes
        -----
        Lazy schema and policy failures occur during this method. Permissive
        cast failures occur at lazy execution and can be native Polars errors.
        Lossless integer-to-float checks run at execution and raise
        SchemaValidationError on failure. Explicit eager output wraps Polars
        collection errors in SchemaValidationError with their causes.
        """
        check_is_fitted(self, "schema_")
        mode = self._mode()
        frame = self._input(X)
        input_schema = frame_schema(frame)
        expected_names = list(self.schema_)
        missing = [col for col in expected_names if col not in input_schema]
        extras = [col for col in input_schema if col not in self.schema_]
        issues: list[str] = []
        if missing:
            issues.append(f"Missing required columns: {missing}.")
        if extras and not self.allow_extra_columns:
            issues.append(f"Unexpected columns: {extras}.")
        present = [col for col in input_schema if col in self.schema_]
        if self.enforce_column_order and not missing and present != expected_names:
            issues.append(
                f"Column order differs. Expected {expected_names}, received {present}."
            )
        conversions: list[pl.Expr | pl.Series] = []
        cause: Exception | None = None
        for column, target in self.schema_.items():
            if column in missing:
                continue
            source = input_schema[column]
            if source == target:
                continue
            detail = (
                f"Column {column!r} has an incompatible type. "
                f"Expected: {target}; Received: {source}; Validation mode: {mode}."
            )
            widening_allowed = True
            if mode == "semi-strict":
                widening_allowed = (
                    _safe_widen(frame[column], target)
                    if isinstance(frame, pl.DataFrame)
                    else _widening_policy(source, target) is not False
                )
            if mode == "strict" or not widening_allowed:
                issues.append(detail)
                continue
            try:
                if isinstance(frame, pl.DataFrame):
                    # Cast once and preserve eager validation exception context.
                    conversions.append(frame[column].cast(target, strict=True))
                    continue
                expression = pl.col(column).cast(target, strict=True)
                if (
                    isinstance(frame, pl.LazyFrame)
                    and mode == "semi-strict"
                    and source in _INTEGER_RANGES
                    and target == pl.Float64
                ):
                    expression = pl.col(column).map_batches(
                        _lossless_float_cast,
                        return_dtype=pl.Float64,
                        is_elementwise=True,
                    )
                conversions.append(expression)
            except pl.exceptions.PolarsError as exc:
                cause = exc
                issues.append(f"{detail} Cast failed: {exc}")
        if issues:
            raise SchemaValidationError("\n".join(issues)) from cause
        result = frame.with_columns(conversions) if conversions else frame
        if self.extra_columns == "drop" or not self.enforce_column_order:
            columns = expected_names
            if self.extra_columns == "preserve":
                columns = columns + extras
            result = result.select(columns)
        return self._finish(result)

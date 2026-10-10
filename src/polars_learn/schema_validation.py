"""Learned schema validation and explicit, predictable type normalisation."""

from typing import Self

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted  # pyright: ignore[reportUnknownVariableType]

from ._utils import NativeTransformer, to_polars
from .exceptions import SchemaValidationError
from .types import ExtraColumnPolicy, Schema, ValidationMode

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


def _safe_widen(series: pl.Series, target: pl.DataType) -> bool:
    """Return whether this series satisfies the lossless widening policy."""
    source = series.dtype
    if source in _INTEGER_RANGES and target in _INTEGER_RANGES:
        source_min, source_max = _INTEGER_RANGES[source]
        target_min, target_max = _INTEGER_RANGES[target]
        return target_min <= source_min and source_max <= target_max
    if source == pl.Float32 and target == pl.Float64:
        return True
    if source in _INTEGER_RANGES and target == pl.Float64:
        try:
            restored = series.cast(pl.Float64).cast(source, strict=True)
            return bool((series == restored).fill_null(True).all())
        except pl.exceptions.PolarsError:
            return False
    return False


class SchemaValidator(NativeTransformer, auto_wrap_output_keys=None):
    """Learn an input schema and validate subsequent Polars DataFrames.

    Parameters
    ----------
    mode : {"strict", "semi-strict", "permissive"} or None, default=None
        Validation policy. ``None`` resolves to strict mode unless ``strict``
        is explicitly false. Strict mode requires identical dtypes. Semi-strict
        allows only the documented lossless widening policy. Permissive mode
        attempts strict Polars casts, which may be lossy but raise on failure.
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
    warn_on_conversion : bool, default=True
        Emit ``PandasConversionWarning`` when converting pandas input.
    strict : bool or None, default=None
        Shorthand for strict mode when true or permissive mode when false.
        Conflicting explicit ``mode`` and ``strict`` values raise ``ValueError``.

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
        mode: ValidationMode | None = None,
        *,
        allow_extra_columns: bool = False,
        enforce_column_order: bool = True,
        extra_columns: ExtraColumnPolicy = "preserve",
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
        strict: bool | None = None,
    ) -> None:
        self.mode = mode
        self.allow_extra_columns = allow_extra_columns
        self.enforce_column_order = enforce_column_order
        self.extra_columns = extra_columns
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion
        self.strict = strict

    def _mode(self) -> ValidationMode:
        if self.mode not in (None, "strict", "semi-strict", "permissive"):
            raise ValueError("mode must be 'strict', 'semi-strict', or 'permissive'.")
        if self.strict is not None and not isinstance(self.strict, bool):
            raise TypeError("strict must be a bool or None.")
        legacy = "strict" if self.strict is not False else "permissive"
        if self.strict is not None and self.mode is not None and self.mode != legacy:
            raise ValueError("strict and mode specify conflicting validation modes.")
        if self.extra_columns not in ("preserve", "drop"):
            raise ValueError("extra_columns must be 'preserve' or 'drop'.")
        return self.mode or legacy

    def _input(self, X: object) -> pl.DataFrame:
        return to_polars(
            X,
            convert_pandas=self.convert_pandas,
            warn_on_conversion=self.warn_on_conversion,
        )

    def fit(self, X: object, y: object = None) -> Self:
        """Learn column names, order, and dtypes without changing the input.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame of shape (n_samples, n_features)
            Training data whose schema becomes the required schema. Pandas
            input requires ``convert_pandas=True``.
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
            If validation configuration is invalid or conflicting.
        """
        self._mode()
        frame = self._input(X)
        self.schema_: Schema = dict(frame.schema)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.width
        self.output_features_ = self.feature_names_in_.copy()
        return self

    def fit_transform(
        self, X: object, y: object = None, **fit_params: object
    ) -> pl.DataFrame:
        """Fit to training data and return its transformed representation.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame of shape (n_samples, n_features)
            Training data. Pandas input requires ``convert_pandas=True``.
        y : object, default=None
            Ignored. Present for scikit-learn estimator compatibility.
        **fit_params : dict
            Accepted for the scikit-learn method signature. Additional fit
            parameters are unsupported and raise ``TypeError`` when supplied.

        Returns
        -------
        X_transformed : polars.DataFrame of shape (n_samples, n_features_out)
            Result of fitting this transformer and transforming the same data.

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

    def transform(self, X: object) -> pl.DataFrame:
        """Validate the learned schema and normalise permitted differences.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame of shape (n_samples, n_columns)
            Input containing all fitted required columns. Additional columns
            must be explicitly permitted. Pandas requires conversion enabled.

        Returns
        -------
        X_validated : polars.DataFrame of shape (n_samples, n_output_columns)
            Validated input with any configured casts, ordering, and extra
            column policy applied. The original input is never mutated.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit`` has not completed successfully.
        TypeError
            If the input is unsupported or pandas conversion is disabled.
        SchemaValidationError
            If required columns are missing, extras or order are disallowed,
            dtypes are incompatible, or a cast fails. Multiple mismatches are
            reported together, with cast exceptions preserved as the cause.
        """
        check_is_fitted(self, "schema_")
        mode = self._mode()
        frame = self._input(X)
        expected_names = list(self.schema_)
        missing = [col for col in expected_names if col not in frame.columns]
        extras = [col for col in frame.columns if col not in self.schema_]
        issues: list[str] = []
        if missing:
            issues.append(f"Missing required columns: {missing}.")
        if extras and not self.allow_extra_columns:
            issues.append(f"Unexpected columns: {extras}.")
        present = [col for col in frame.columns if col in self.schema_]
        if self.enforce_column_order and not missing and present != expected_names:
            issues.append(
                f"Column order differs. Expected {expected_names}, received {present}."
            )
        conversions: list[pl.Series] = []
        cause: Exception | None = None
        for column, target in self.schema_.items():
            if column in missing:
                continue
            series = frame[column]
            source = series.dtype
            if source == target:
                continue
            detail = (
                f"Column {column!r} has an incompatible type. "
                f"Expected: {target}; Received: {source}; Validation mode: {mode}."
            )
            if mode == "strict" or (
                mode == "semi-strict" and not _safe_widen(series, target)
            ):
                issues.append(detail)
                continue
            try:
                conversions.append(series.cast(target, strict=True))
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
        return result

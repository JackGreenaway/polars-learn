"""Shared input conversion and estimator conventions."""

import importlib
import warnings
from typing import Any, Self, SupportsIndex, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)

from ._serialization import reduce_object
from .exceptions import PandasConversionWarning
from .types import FeatureNames, Frame, OutputFrame, OutputMode, Schema


def validate_verbosity(verbosity: int) -> None:
    """Reject unsupported diagnostic levels before doing any work."""
    if type(verbosity) is not int:
        raise TypeError("verbosity must be an integer (0, 1, or 2).")
    if verbosity not in (0, 1, 2):
        raise ValueError("verbosity must be 0, 1, or 2.")


def report(owner: object, verbosity: int, message: str, *, level: int = 1) -> None:
    """Emit an opt-in diagnostic without printing input values."""
    if verbosity >= level:
        print(f"[{type(owner).__name__}] {message}")


def to_polars(X: object, *, convert_pandas: bool, warn_on_conversion: bool) -> Frame:
    """Validate the container and optionally convert pandas input to Polars.

    Parameters
    ----------
    X : object
        Candidate Polars DataFrame or LazyFrame. NumPy arrays are unsupported.
    convert_pandas : bool
        Whether pandas DataFrames may be converted with ``pl.from_pandas``.
    warn_on_conversion : bool
        Whether conversion emits ``PandasConversionWarning``.

    Returns
    -------
    frame : polars.DataFrame or polars.LazyFrame
        Original Polars input or converted pandas DataFrame.

    Raises
    ------
    TypeError
        If input is unsupported, conversion is disabled, or a conversion
        dependency is unavailable.

    Warns
    -----
    PandasConversionWarning
        If pandas input is converted and warnings are enabled.
    """
    if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
        return X
    # Inspect the MRO so subclasses work without importing pandas for native inputs.
    if any(cls.__module__.startswith("pandas.") for cls in type(X).__mro__):
        if not convert_pandas:
            raise TypeError(
                "Received pandas input; set convert_pandas=True to enable conversion."
            )
        try:
            pd = importlib.import_module("pandas")
        except ImportError as exc:
            raise TypeError("Pandas conversion requires the 'pandas' extra.") from exc
        if isinstance(X, pd.DataFrame):
            try:
                # Runtime isinstance above narrows an optional, lazily imported type.
                result = cast(pl.DataFrame, pl.from_pandas(cast(Any, X)))  # pyright: ignore[reportUnknownMemberType, reportUnnecessaryCast]
            except ImportError as exc:
                raise TypeError(
                    "This pandas dtype requires PyArrow; install polars-learn[pandas]."
                ) from exc
            if warn_on_conversion:
                warnings.warn(
                    "Received a pandas DataFrame and converted it to Polars. "
                    "For optimal performance, provide a Polars DataFrame directly; "
                    "conversion may add memory and performance overhead.",
                    PandasConversionWarning,
                    stacklevel=3,
                )
            return result
    raise TypeError(
        "Expected a Polars DataFrame or LazyFrame (or explicitly enabled pandas input)."
    )


def frame_schema(frame: Frame) -> Schema:
    """Resolve metadata without collecting a lazy query's rows."""
    return dict(frame.collect_schema())


class NativeTransformer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Keep native output and extra columns free of wrapper renames."""

    feature_names_in_: NDArray[np.object_]
    n_features_in_: int
    output_features_: NDArray[np.object_]

    verbosity: int

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        """Warn about changed dependencies before restoring saved transformer state."""
        return reduce_object(self, protocol)

    def _validate_options(self) -> None:
        validate_verbosity(self.verbosity)
        for name in (
            "convert_pandas",
            "warn_on_conversion",
            "allow_extra_columns",
            "enforce_column_order",
        ):
            if hasattr(self, name) and type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")

    def _report(self, message: str, *, level: int = 1) -> None:
        report(self, self.verbosity, message, level=level)

    def fit_transform(self, X: object, y: object = None, **fit_params: object) -> Any:
        """Bridge sklearn's ndarray annotation to native container overrides.

        Concrete subclasses provide typed Polars and pandas output implementations.
        sklearn's mixin supports non-array outputs at runtime.
        """
        return cast(Any, super()).fit_transform(X, y, **fit_params)

    def _report_frame(self, frame: Frame, *, stage: str) -> None:
        if not self.verbosity:
            return
        schema = frame_schema(frame)
        rows = str(frame.height) if isinstance(frame, pl.DataFrame) else "unknown"
        suffix = (
            " (lazy plan; values not evaluated)"
            if isinstance(frame, pl.LazyFrame)
            else ""
        )
        self._report(f"{stage}: {rows} rows, {len(schema)} columns{suffix}")
        self._report(f"{stage} schema: {schema}", level=2)

    def _collect(self, result: pl.LazyFrame) -> pl.DataFrame:
        """Execute an explicitly requested eager result."""
        return result.collect()

    def _finish(self, result: Frame) -> OutputFrame:
        mode = getattr(self, "_sklearn_output_config", {}).get("transform", "default")
        if mode in ("polars", "pandas") and isinstance(result, pl.LazyFrame):
            result = self._collect(result)
        elif mode == "polars-lazy" and isinstance(result, pl.DataFrame):
            result = result.lazy()
        self._report_frame(result, stage="Transform complete")
        if mode == "pandas":
            try:
                return cast(pl.DataFrame, result).to_pandas()
            except ImportError as exc:
                raise ImportError(
                    "Pandas output requires pandas and PyArrow; "
                    "install polars-learn[pandas]."
                ) from exc
        return result

    def get_feature_names_out(
        self, input_features: FeatureNames | None = None
    ) -> NDArray[np.object_]:
        """Return the fitted output column names.

        Use these names to label model inputs or inspect a fitted pipeline.
        Returns a copy of the names recorded during fitting; it does not
        transform data, collect lazy rows, or refit the transformer. Supplied
        input names are checked against all fitted input names in exact order.
        For SchemaValidator, inference-only extra columns are not included;
        this is the fitted contract rather than a live query's column list.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            If supplied, names must exactly match ``feature_names_in_``.

        Returns
        -------
        feature_names_out : ndarray of shape (n_features_out_,), dtype=object
            Copy of fitted output names. SchemaValidator excludes inference-only
            extras. SchemaSpec includes preserved training extras. A selector
            lists selected names, including any renames; a passthrough lists unchanged
            selected names. An engineer lists all output names.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the transformer has not been fitted.
        ValueError
            If supplied input names differ from fitted names or order.
        """
        check_is_fitted(self, "output_features_")
        if input_features is not None and not np.array_equal(
            np.asarray(input_features, dtype=object), self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_ exactly.")
        return self.output_features_.copy()

    # sklearn exposes set_output as an available_if descriptor.
    def set_output(self, *, transform: OutputMode | None = None) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride, reportIncompatibleVariableOverride]
        """Configure native Polars output or pandas compatibility output.

        Call this when a downstream consumer needs a particular frame container.
        It only stores a choice for subsequent transform/fit_transform calls;
        it does not fit, transform, or collect data at configuration time.
        Use "default" to follow each call's input kind, "polars" for eager
        Polars, "polars-lazy" for a lazy plan, or "pandas" for an eager pandas
        DataFrame. Converted pandas input is treated as eager Polars by default.
        Pandas input permission remains separate: enable convert_pandas on any
        transformer receiving pandas, even when pandas output is requested.

        Parameters
        ----------
        transform : {"default", "polars", "polars-lazy", "pandas"} or None, default=None
            Default preserves the input kind. "polars" collects an eager
            DataFrame; "polars-lazy" returns a LazyFrame. ``None`` leaves
            the configuration unchanged. "pandas" collects if necessary and converts
            the result to a pandas DataFrame. Requires the pandas extra.

        Returns
        -------
        self : estimator instance
            This transformer.

        Raises
        ------
        ValueError
            If an unsupported output container is requested.

        Notes
        -----
        Prefer native Polars output for speed and to avoid conversion overhead.
        Pandas output is provided for compatibility and uses Polars to_pandas
        defaults, including a fresh RangeIndex and dtype/null conversion.
        Conversion raises ImportError if pandas or PyArrow is unavailable.
        Input conversion remains separately controlled by convert_pandas.
        Automatic scikit-learn output wrapping and global output settings are
        bypassed to preserve native names and permitted extra columns.
        Output choices survive cloning. None leaves the current choice intact;
        "default" resets input-kind preservation. Fit always returns self and
        does not collect lazy rows. Forcing lazy output from eager input may
        wrap an already evaluated result; use lazy input to defer evaluation.
        """
        if transform not in (None, "default", "polars", "polars-lazy", "pandas"):
            raise ValueError(
                "Only 'default', 'polars', 'polars-lazy', and 'pandas' "
                "output are supported."
            )
        if transform is not None:
            self._sklearn_output_config = {"transform": transform}
        return self

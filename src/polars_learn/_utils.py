"""Shared input conversion and estimator conventions."""

import importlib
import warnings
from typing import Any, Self, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted  # pyright: ignore[reportUnknownVariableType]

from .exceptions import PandasConversionWarning
from .types import FeatureNames, OutputMode


def to_polars(
    X: object, *, convert_pandas: bool, warn_on_conversion: bool
) -> pl.DataFrame:
    """Validate the container and optionally convert pandas input to Polars.

    Parameters
    ----------
    X : object
        Candidate eager DataFrame. NumPy arrays and LazyFrames are unsupported.
    convert_pandas : bool
        Whether pandas DataFrames may be converted with ``pl.from_pandas``.
    warn_on_conversion : bool
        Whether conversion emits ``PandasConversionWarning``.

    Returns
    -------
    frame : polars.DataFrame
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
    if isinstance(X, pl.DataFrame):
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
    raise TypeError("Expected a Polars DataFrame (or explicitly enabled pandas input).")


class NativeTransformer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Keep native output and extra columns free of wrapper renames."""

    feature_names_in_: NDArray[np.object_]
    n_features_in_: int
    output_features_: NDArray[np.object_]

    def get_feature_names_out(
        self, input_features: FeatureNames | None = None
    ) -> NDArray[np.object_]:
        """Return the fitted output column names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            If supplied, names must exactly match ``feature_names_in_``.

        Returns
        -------
        feature_names_out : ndarray of shape (n_features_out_,), dtype=object
            Copy of the fitted output names. For schema validators this
            describes required columns, excluding inference-only extras.

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
    def set_output(self, *, transform: OutputMode | None = None) -> Self:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configure native output without automatic container conversion.

        Parameters
        ----------
        transform : {"default", "polars"} or None, default=None
            Both supported values return Polars DataFrames. ``None`` leaves
            the configuration unchanged. Pandas output is unsupported.

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
        Automatic scikit-learn output wrapping and global output settings are
        bypassed to preserve native names and permitted extra columns.
        """
        if transform not in (None, "default", "polars"):
            raise ValueError("Only 'default' and 'polars' output are supported.")
        return self

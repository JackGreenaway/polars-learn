"""Sequential feature engineering for scikit-learn pipelines."""

from collections.abc import Sequence
from typing import Self

import numpy as np
import polars as pl
from sklearn.utils.validation import (
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)

from ._utils import NativeTransformer, to_polars
from .exceptions import FeatureEngineeringError
from .feature_spec import FeatureSpec
from .types import Schema


class FeatureEngineer(NativeTransformer, auto_wrap_output_keys=None):
    """Apply ordered, stateless feature specifications to Polars DataFrames.

    Parameters
    ----------
    features : sequence of FeatureSpec
        Specifications evaluated in order. Later specifications may reference
        earlier outputs. An empty sequence produces an identity transformation.
        The constructor retains this parameter unchanged for estimator cloning.
    convert_pandas : bool, default=False
        Allow pandas DataFrames and convert them to Polars. Requires pandas;
        some dtypes also require PyArrow. No conversion occurs for Polars input.
    warn_on_conversion : bool, default=True
        Emit ``PandasConversionWarning`` when converting pandas input.

    Attributes
    ----------
    n_features_in_ : int
        Number of columns observed during fitting.
    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=object
        Input column names in training order.
    input_schema_ : dict of str to polars.DataType
        Input dtypes observed during fitting, independent of other estimators.
    output_schema_ : dict of str to polars.DataType
        Expected transformed names, order, and dtypes.
    output_features_ : ndarray of shape (n_features_out_,), dtype=object
        Output column names, including retained inputs and engineered features.

    See Also
    --------
    FeatureSpec : Describe one feature operation.
    SchemaValidator : Validate and normalise raw input before engineering.

    Notes
    -----
    The public API accepts eager DataFrames. Internally, lazy operations retain
    dependency order and collect once per execution. Fitting executes the
    training transformation to validate it but learns no feature statistics.
    Transform requires fitted input names and order and the fitted output schema.
    Both default and explicitly requested Polars output remain native Polars.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import FeatureEngineer, FeatureSpec
    >>> X = pl.DataFrame({"price": [10.0], "quantity": [2]})
    >>> engineer = FeatureEngineer([
    ...     FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
    ... ])
    >>> engineer.fit_transform(X)["notional"].to_list()
    [20.0]
    >>> engineer.get_feature_names_out().tolist()
    ['price', 'quantity', 'notional']
    """

    def __init__(
        self,
        features: Sequence[FeatureSpec],
        *,
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
    ) -> None:
        self.features = features
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion

    def _input(self, X: object) -> pl.DataFrame:
        return to_polars(
            X,
            convert_pandas=self.convert_pandas,
            warn_on_conversion=self.warn_on_conversion,
        )

    def _execute(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(self.features, Sequence) or isinstance(self.features, str):
            raise TypeError(
                "features must be an ordered sequence of FeatureSpec objects."
            )
        plan = frame.lazy()
        labels: list[str] = []
        for index, spec in enumerate(self.features):
            if not isinstance(spec, FeatureSpec):
                raise TypeError(f"features[{index}] must be a FeatureSpec.")
            label = spec.name or f"specification {index}: {spec!r}"
            labels.append(label)
            try:
                plan = plan.with_columns(spec.expressions)
                # Resolve each stage for precise dependency/drop errors, without
                # materialising data until the final collect.
                schema = plan.collect_schema()
                missing = set(spec.drop) - set(schema.names())
                if missing:
                    raise ValueError(f"Cannot drop missing columns: {sorted(missing)}")
                if spec.drop:
                    plan = plan.drop(spec.drop)
            except Exception as exc:
                raise FeatureEngineeringError(
                    f"Failed to generate {label}: {exc}"
                ) from exc
        try:
            return plan.collect()
        except Exception as exc:
            raise FeatureEngineeringError(
                f"Failed to execute feature plan ({'; '.join(labels)}): {exc}"
            ) from exc

    def fit(self, X: object, y: object = None) -> Self:
        """Validate training expressions and learn input and output schemas.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame of shape (n_samples, n_features)
            Training data. Pandas input requires ``convert_pandas=True``.
        y : object, default=None
            Ignored. Present for scikit-learn estimator compatibility.

        Returns
        -------
        self : FeatureEngineer
            Fitted transformer.

        Raises
        ------
        TypeError
            If input or feature specifications have unsupported types.
        FeatureEngineeringError
            If expressions or drop operations fail. The underlying exception
            is preserved as the cause.
        """
        frame = self._input(X)
        result = self._execute(frame)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.width
        self.input_schema_: Schema = dict(frame.schema)
        self.output_schema_: Schema = dict(result.schema)
        self.output_features_ = np.asarray(result.columns, dtype=object)
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
        """Generate features and verify the fitted output schema.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame of shape (n_samples, n_features)
            Input with the fitted column names and order. Pandas input requires
            ``convert_pandas=True``. Input dtypes may differ only if the final
            output schema still matches the fitted expectations.

        Returns
        -------
        X_transformed : polars.DataFrame of shape (n_samples, n_features_out)
            Retained and engineered columns in native Polars order.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit`` has not completed successfully.
        TypeError
            If the input is unsupported or pandas conversion is disabled.
        FeatureEngineeringError
            If input columns differ, feature execution fails, or the generated
            output schema differs from the fitted schema.
        """
        check_is_fitted(self, "output_schema_")
        frame = self._input(X)
        if frame.columns != self.feature_names_in_.tolist():
            raise FeatureEngineeringError(
                f"Input columns must match fitted names and order. Expected "
                f"{self.feature_names_in_.tolist()}, received {frame.columns}."
            )
        result = self._execute(frame)
        if list(result.schema.items()) != list(self.output_schema_.items()):
            raise FeatureEngineeringError(
                f"Output schema differs from fit. Expected {self.output_schema_}, "
                f"received {dict(result.schema)}."
            )
        return result

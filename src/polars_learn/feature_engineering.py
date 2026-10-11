"""Sequential feature engineering for scikit-learn pipelines."""

from collections.abc import Sequence
from typing import Any, Self, TypeAlias, cast

import numpy as np
import polars as pl
from sklearn.utils.validation import (
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)

from ._utils import NativeTransformer, frame_schema, to_polars
from .exceptions import FeatureEngineeringError
from .feature_spec import FeatureSpec
from .types import FeatureSpecConfig, Frame, OutputFrame, Schema

# Ordinary dict variables have heterogeneous values; validate their kwargs at runtime.
_FeatureDefinition: TypeAlias = FeatureSpec | FeatureSpecConfig | dict[str, Any]


class FeatureEngineer(NativeTransformer, auto_wrap_output_keys=None):
    """Apply ordered feature specifications to Polars DataFrames or LazyFrames.

    Use this transformer to keep Polars feature logic inside a model pipeline,
    so training and prediction apply the same preparation to raw observations.
    It applies each ``FeatureSpec`` in entry order using Polars ``with_columns``,
    then applies that specification's drops before starting the next. Later
    specifications can therefore use retained outputs from earlier stages.
    ``fit`` records input names/order and the resulting output schema;
    ``transform`` replays the definitions and checks that schema. It does not
    learn expression-specific training statistics: add a fitted statistical
    transformer for quantities that must remain fixed after training.

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
    features : FeatureSpec, dict, or sequence of FeatureSpec/dict
        A specification or a dict of FeatureSpec keyword arguments, optionally
        in an ordered sequence mixing both forms. Dicts are unpacked into
        FeatureSpec and validated before processing input. Single definitions
        are treated as one-element sequences. Specifications are applied in the
        supplied sequence order. Each specification adds its expressions, then
        drops its configured columns, before the next specification is applied.
        Later specifications may reference earlier outputs that remain present.
        An empty sequence produces an identity transformation;
        an empty dict is invalid. The original parameter is retained unchanged
        for estimator cloning.
    convert_pandas : bool, default=False
        Allow pandas DataFrames and convert them to Polars. Requires pandas;
        some dtypes also require PyArrow. No conversion occurs for Polars input.
    verbosity : {0, 1, 2}, default=0
        Print progress at 1 and schema/operation details at 2; 0 is silent.
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

    Raises
    ------
    TypeError
        If feature definitions or common options have invalid types, a config
        dict has unknown keys, or a config has invalid FeatureSpec value types.
    ValueError
        If verbosity is outside 0, 1, and 2 or a config dict fails FeatureSpec
        validation, such as an empty expression sequence or an ambiguous alias.

    See Also
    --------
    FeatureSpec : Describe one feature operation.
    SchemaValidator : Validate and normalise raw input before engineering.

    Notes
    -----
    Specifications preserve the supplied sequence order in fit and transform.
    Expressions within one specification are siblings: they read the columns
    available before that specification, not new sibling outputs. Use later
    specifications to reference newly created columns, or compose expressions
    directly from existing input columns in one specification.
    The public API accepts DataFrames and LazyFrames. Lazy inputs are checked
    using schema metadata and return unevaluated plans by default. Eager input
    executes the training transformation; lazy fit validates its schema only.
    Value-dependent errors in lazy plans surface when collected.
    Transform requires fitted input names and order and the fitted output schema.
    Default output preserves input kind. set_output(transform="polars")
    collects a DataFrame; "polars-lazy" returns a LazyFrame, which may wrap an
    evaluated result for eager input. "pandas" converts an eager result for
    compatibility. Prefer native Polars output to avoid conversion overhead.
    Fit always returns self. fit_transform
    calls fit then transform and evaluates eager feature expressions twice.
    Pickling retains input definitions, fitted schemas, and output settings.
    Expressions use Polars' native persistence, with cloudpickle for embedded
    Python callbacks. Version mismatches warn before state restoration, then
    compatible loading is attempted. Incompatible state raises SerializationError.
    Load trusted artifacts only and validate features after upgrading.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import FeatureEngineer, FeatureSpec
    >>> X = pl.DataFrame({"price": [10.0], "quantity": [2]})
    >>> engineer = FeatureEngineer(
    ...     FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
    ... )
    >>> engineer.fit_transform(X)["notional"].to_list()
    [20.0]
    >>> engineer.get_feature_names_out().tolist()
    ['price', 'quantity', 'notional']
    >>> configured = FeatureEngineer({
    ...     "expression": pl.col("price") * pl.col("quantity"),
    ...     "alias": "notional",
    ... })
    >>> configured.fit_transform(X)["notional"].to_list()
    [20.0]
    """

    def __init__(
        self,
        features: _FeatureDefinition | Sequence[_FeatureDefinition],
        *,
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
        verbosity: int = 0,
    ) -> None:
        self.features = features
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion
        self.verbosity = verbosity
        self._validate_options()
        self._validate_features()

    @staticmethod
    def _as_spec(definition: object, *, label: str) -> FeatureSpec:
        if isinstance(definition, FeatureSpec):
            return definition
        if isinstance(definition, dict):
            try:
                return FeatureSpec(**cast(FeatureSpecConfig, definition))
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"{label}: invalid FeatureSpec config: {exc}") from exc
        raise TypeError(f"{label} must be a FeatureSpec or a FeatureSpec config dict.")

    def _validate_features(self) -> tuple[FeatureSpec, ...]:
        """Normalize and validate operations without mutating estimator parameters."""
        if isinstance(self.features, (FeatureSpec, dict)):
            return (self._as_spec(self.features, label="features"),)
        if not isinstance(self.features, Sequence) or isinstance(self.features, str):
            raise TypeError(
                "features must be a FeatureSpec, a FeatureSpec config dict, or an "
                "ordered sequence of those definitions."
            )
        return tuple(
            self._as_spec(definition, label=f"features[{index}]")
            for index, definition in enumerate(self.features)
        )

    def _input(self, X: object) -> tuple[Frame, tuple[FeatureSpec, ...]]:
        self._validate_options()
        specs = self._validate_features()
        self._report("Validating input")
        frame = to_polars(
            X,
            convert_pandas=self.convert_pandas,
            warn_on_conversion=self.warn_on_conversion,
        )
        self._report_frame(frame, stage="Input")
        return frame, specs

    def _execute(self, frame: Frame, specs: tuple[FeatureSpec, ...]) -> Frame:
        plan = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        labels: list[str] = []
        for index, spec in enumerate(specs):
            label = spec.name or f"specification {index}: {spec!r}"
            self._report(f"Applying {label}", level=2)
            labels.append(label)
            try:
                plan = plan.with_columns(spec.expression)
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
            if isinstance(frame, pl.LazyFrame):
                plan.collect_schema()
                return plan
            return plan.collect()
        except Exception as exc:
            raise FeatureEngineeringError(
                f"Failed to execute feature plan ({'; '.join(labels)}): {exc}"
            ) from exc

    def _collect(self, result: pl.LazyFrame) -> pl.DataFrame:
        try:
            return result.collect()
        except Exception as exc:
            raise FeatureEngineeringError(
                f"Failed to execute feature plan: {exc}"
            ) from exc

    def fit(self, X: object, y: object = None) -> Self:
        """Validate training expressions and learn input and output schemas.

        Call on raw training data to establish the expected input and feature layout.
        Builds the ordered feature plan and records input names/order and resulting
        output names/dtypes. Eager input executes the plan; lazy input checks schema
        metadata without evaluating rows. Expressions are not converted into learned
        statistics. Returns the fitted transformer, not the engineered frame.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Training data. Pandas input requires ``convert_pandas=True``.
            Lazy training validates plan schemas without evaluating row values.
        y : object, default=None
            Ignored. Present for scikit-learn estimator compatibility.

        Returns
        -------
        self : FeatureEngineer
            Fitted transformer.

        Raises
        ------
        TypeError
            If input, feature definitions, or options have invalid types, or a
            config dict contains unknown keys.
        ValueError
            If verbosity or FeatureSpec config values are invalid.
        FeatureEngineeringError
            If expressions or drop operations fail. The underlying exception
            is preserved as the cause.
        """
        frame, specs = self._input(X)
        input_schema = frame_schema(frame)
        result = self._execute(frame, specs)
        self.feature_names_in_ = np.asarray(list(input_schema), dtype=object)
        self.n_features_in_ = len(input_schema)
        self.input_schema_: Schema = input_schema
        self.output_schema_: Schema = frame_schema(result)
        self.output_features_ = np.asarray(list(frame_schema(result)), dtype=object)
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
        Eager feature expressions run twice because fit and transform each execute
        the feature plan. Lazy input remains a plan unless eager output is requested.

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
            Lazy training validates plan schemas without evaluating row values.
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

    def transform(self, X: object) -> OutputFrame:
        """Generate features and verify the fitted output schema.

        Call after fitting to prepare new raw observations with the same feature
        logic. Requires the fitted input names in their original order, applies
        specifications in entry order, and checks output names, order, and dtypes
        against ``output_schema_``. Each stage adds its expressions and then drops
        its configured columns; subsequent stages can use retained earlier outputs.
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
            Input with the fitted column names and order. Pandas input requires
            ``convert_pandas=True``. Input dtypes may differ only if the final
            output schema still matches the fitted expectations.

        Returns
        -------
        X_transformed : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Retained and engineered columns in native Polars order. Default
            preserves input kind; set_output can force eager, lazy, or pandas output.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit`` has not completed successfully.
        TypeError
            If input, features, or options are unsupported, config keys are unknown,
            or pandas conversion is disabled.
        ValueError
            If verbosity or FeatureSpec config values are invalid.
        FeatureEngineeringError
            If input columns differ, feature execution fails, or the generated
            output schema differs from the fitted schema.

        Notes
        -----
        Schema/dependency errors fail before a lazy plan is returned. Value
        errors in that plan occur when collected and can be native Polars
        errors. Explicit eager output wraps execution errors in
        FeatureEngineeringError, preserving their causes.
        """
        check_is_fitted(self, "output_schema_")
        frame, specs = self._input(X)
        input_schema = frame_schema(frame)
        if list(input_schema) != self.feature_names_in_.tolist():
            raise FeatureEngineeringError(
                f"Input columns must match fitted names and order. Expected "
                f"{self.feature_names_in_.tolist()}, received {list(input_schema)}."
            )
        result = self._execute(frame, specs)
        if list(frame_schema(result).items()) != list(self.output_schema_.items()):
            raise FeatureEngineeringError(
                f"Output schema differs from fit. Expected {self.output_schema_}, "
                f"received {frame_schema(result)}."
            )
        return self._finish(result)

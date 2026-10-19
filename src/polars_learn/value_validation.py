"""Validate row-level conditions in Polars frames."""

from typing import Self

import numpy as np
import polars as pl
from sklearn.utils.validation import (
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)

from ._utils import NativeTransformer, frame_schema, to_polars
from .exceptions import ValueValidationError
from .types import Frame, OutputFrame, Schema


class ValueValidator(NativeTransformer, auto_wrap_output_keys=None):
    """Reject rows that fail named Boolean Polars expressions.

    Use this transformer after structural schema validation for row-level rules
    such as non-null requirements, bounds, and allowed-value checks. Every check
    must produce one Boolean value per row. False and null results are violations.
    The input is returned unchanged when all checks pass.

    Parameters
    ----------
    checks : dict of str to polars.Expr
        Nonempty mapping from diagnostic labels to row-wise Boolean expressions.
        For example, ``{"positive_price": pl.col("price") > 0}``.
    convert_pandas : bool, default=False
        Allow pandas DataFrames and convert them to Polars.
    warn_on_conversion : bool, default=True
        Emit ``PandasConversionWarning`` when converting pandas input.
    verbosity : {0, 1, 2}, default=0
        Print progress at 1 and schema/check details at 2; 0 is silent.

    Attributes
    ----------
    feature_names_in_ : ndarray of str, dtype=object
        Input column names and order recorded during fit.
    n_features_in_ : int
        Number of input columns recorded during fit.
    output_features_ : ndarray of str, dtype=object
        Input column names, unchanged by validation.

    Raises
    ------
    TypeError
        If checks, expressions, or common options have invalid types.
    ValueError
        If checks are empty, labels are invalid, or expressions do not resolve
        to Boolean values on the fitted schema.
    ValueValidationError
        If an input schema differs from fitting, checks cannot execute, or any
        check returns false or null for one or more rows.
    sklearn.exceptions.NotFittedError
        If transform is called before a successful fit.

    Notes
    -----
    ``fit`` resolves the check expressions against schema metadata only; it never
    evaluates row values. ``transform`` evaluates the checks once. Lazy inputs
    are materialized during transform so rule violations fail before it returns
    and default output can preserve the lazy container using an in-memory frame.
    Value checks do not add columns, mutate input, or change row order.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import ValueValidator
    >>> frame = pl.DataFrame({"price": [2.0, 3.0], "quantity": [1, 2]})
    >>> checks = {
    ...     "positive_price": pl.col("price") > 0,
    ...     "nonnegative_quantity": pl.col("quantity") >= 0,
    ... }
    >>> validator = ValueValidator(checks)
    >>> validator.fit_transform(frame).equals(frame)
    True
    """

    def __init__(
        self,
        checks: dict[str, pl.Expr],
        *,
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
        verbosity: int = 0,
    ) -> None:
        self.checks = checks
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion
        self.verbosity = verbosity
        self._validate_checks()

    def _validate_checks(self) -> dict[str, pl.Expr]:
        self._validate_options()
        if not isinstance(self.checks, dict):
            raise TypeError("checks must be a dict mapping labels to Polars Exprs.")
        if not self.checks:
            raise ValueError("checks must contain at least one value check.")
        for name, expression in self.checks.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Check labels must be nonempty strings.")
            if not isinstance(expression, pl.Expr):
                raise TypeError(f"Check {name!r} must be a Polars Expr.")
            if expression.meta.has_multiple_outputs():
                raise ValueError(f"Check {name!r} must produce one Boolean output.")
        return self.checks

    def _input(self, X: object) -> Frame:
        self._validate_checks()
        frame = to_polars(
            X,
            convert_pandas=self.convert_pandas,
            warn_on_conversion=self.warn_on_conversion,
        )
        self._report_frame(frame, stage="Input")
        return frame

    def _resolve(self, frame: Frame) -> tuple[Schema, list[pl.Expr]]:
        schema = frame_schema(frame)
        checks = self._validate_checks()
        try:
            plan = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
            check_schema = plan.select(
                [expression.alias(name) for name, expression in checks.items()]
            ).collect_schema()
        except Exception as exc:
            raise ValueValidationError(
                f"Could not resolve value checks: {exc}"
            ) from exc
        invalid_types = [
            name for name, dtype in check_schema.items() if dtype != pl.Boolean
        ]
        if invalid_types:
            raise ValueError(
                f"Value checks must resolve to Boolean expressions: {invalid_types}."
            )
        aggregates = [
            expression.fill_null(False).all().alias(name)
            for name, expression in checks.items()
        ]
        return schema, aggregates

    def _check_values(self, frame: Frame, aggregates: list[pl.Expr]) -> None:
        try:
            result = frame.select(aggregates)
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            failed = [
                name
                for name, passed in result.row(0, named=True).items()
                if passed is not True
            ]
        except ValueValidationError:
            raise
        except Exception as exc:
            raise ValueValidationError(
                f"Could not evaluate value checks: {exc}"
            ) from exc
        if failed:
            raise ValueValidationError(f"Input failed value checks: {failed}.")

    def _fit_frame(self, frame: Frame) -> None:
        schema, _ = self._resolve(frame)
        self.feature_names_in_ = np.asarray(list(schema), dtype=object)
        self.n_features_in_ = len(schema)
        self.output_features_ = self.feature_names_in_.copy()
        self.input_schema_: Schema = schema
        self._report("Fit complete; value check schema resolved")

    def fit(self, X: object, y: object = None) -> Self:
        """Resolve checks against the input schema without evaluating rows."""
        self._fit_frame(self._input(X))
        return self

    def fit_transform(
        self, X: object, y: object = None, **fit_params: object
    ) -> OutputFrame:
        """Fit the schema contract, validate values, and return the input."""
        if fit_params:
            raise TypeError(
                f"Additional fit parameters are unsupported: {sorted(fit_params)}."
            )
        frame = self._input(X)
        self._fit_frame(frame)
        return self._transform_frame(frame)

    def transform(self, X: object) -> OutputFrame:
        """Validate fitted input names, order, dtypes, and configured values."""
        check_is_fitted(self, "input_schema_")
        frame = self._input(X)
        return self._transform_frame(frame)

    def _transform_frame(self, frame: Frame) -> OutputFrame:
        was_lazy = isinstance(frame, pl.LazyFrame)
        schema = frame_schema(frame)
        if list(schema.items()) != list(self.input_schema_.items()):
            raise ValueValidationError(
                "Input schema differs from fit. "
                f"Expected {self.input_schema_}, received {schema}."
            )
        aggregates = [
            expression.fill_null(False).all().alias(name)
            for name, expression in self._validate_checks().items()
        ]
        if was_lazy:
            try:
                frame = frame.collect()
            except Exception as exc:
                raise ValueValidationError(
                    f"Could not materialize input for value checks: {exc}"
                ) from exc
        self._check_values(frame, aggregates)
        if (
            was_lazy
            and getattr(self, "_sklearn_output_config", {}).get("transform", "default")
            == "default"
        ):
            frame = frame.lazy()
        return self._finish(frame)

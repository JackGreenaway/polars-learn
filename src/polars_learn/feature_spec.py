"""Composable descriptions of native Polars feature operations."""

from collections.abc import Sequence
from typing import Any, Self, SupportsIndex

import numpy as np
import polars as pl
from sklearn.utils.validation import (
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)

from ._serialization import reduce_object
from ._utils import (
    NativeTransformer,
    frame_schema,
    report,
    to_polars,
    validate_verbosity,
)
from .types import Frame


class ColumnSelector(NativeTransformer, auto_wrap_output_keys=None):
    """Select and optionally rename columns in a Polars DataFrame or LazyFrame.

    Parameters
    ----------
    columns : sequence of str
        Nonempty ordered sequence of unique, nonempty input column names to keep.
    rename : dict of str to str, default=None
        Optional mapping from selected source names to nonempty output names.
        Final output names, including unchanged names, must remain unique.
    convert_pandas : bool, default=False
        Allow pandas DataFrames and convert them to Polars. Requires pandas;
        some dtypes also require PyArrow.
    verbosity : {0, 1, 2}, default=0
        Print progress at 1 and schema/operation details at 2; 0 is silent.
    warn_on_conversion : bool, default=True
        Emit ``PandasConversionWarning`` when converting pandas input.

    Attributes
    ----------
    feature_names_in_ : ndarray of str, dtype=object
        All training input column names in input order, including unselected names.
    n_features_in_ : int
        Number of all training input columns.
    columns_ : tuple of str
        Selected source names recorded by fit.
    output_features_ : ndarray of str, dtype=object
        Selected output names after renaming, in selection order.

    Raises
    ------
    TypeError
        If columns, rename, or common options have invalid types.
    ValueError
        If selected or final renamed names are duplicated, rename keys are
        unselected, or verbosity is outside 0, 1, and 2.

    See Also
    --------
    SchemaSpec : Validate explicit required dtypes before selecting columns.
    FeaturePassthrough : Explicit unchanged-column selection without renaming.

    Notes
    -----
    DataFrames return DataFrames and LazyFrames return unevaluated LazyFrames
    by default. Use set_output to force an eager or lazy output container.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import ColumnSelector
    >>> X = pl.DataFrame({"a": [1], "b": [2]})
    >>> selector = ColumnSelector(["b"], rename={"b": "value"})
    >>> selector.fit_transform(X.lazy()).collect().columns
    ['value']
    """

    def __init__(
        self,
        columns: Sequence[str],
        *,
        rename: dict[str, str] | None = None,
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
        verbosity: int = 0,
    ) -> None:
        self.columns = columns
        self.rename = rename
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion
        self.verbosity = verbosity
        self._validate_options()
        self._validate_columns()

    def _input(self, X: object) -> Frame:
        self._validate_options()
        self._validate_columns()
        self._report("Validating input")
        frame = to_polars(
            X,
            convert_pandas=self.convert_pandas,
            warn_on_conversion=self.warn_on_conversion,
        )
        self._report_frame(frame, stage="Input")
        return frame

    def _validate_columns(self) -> tuple[str, ...]:
        if isinstance(self.columns, str) or not isinstance(self.columns, Sequence):
            raise TypeError("columns must be a sequence of nonempty column names.")
        columns = tuple(self.columns)
        if not columns or any(
            not isinstance(column, str) or not column for column in columns
        ):
            raise TypeError("columns must be a sequence of nonempty column names.")
        if len(set(columns)) != len(columns):
            raise ValueError("columns must be unique.")
        if self.rename is not None:
            if not isinstance(self.rename, dict):
                raise TypeError(
                    "rename must be a dict mapping source names to output names."
                )
            invalid = [
                (source, target)
                for source, target in self.rename.items()
                if not isinstance(source, str)
                or not source
                or not isinstance(target, str)
                or not target
            ]
            if invalid:
                raise TypeError("rename keys and values must be nonempty strings.")
            if len(set(self.rename.values())) != len(self.rename):
                raise ValueError("rename output names must be unique.")
            missing = [source for source in self.rename if source not in columns]
            if missing:
                raise ValueError(f"rename contains unknown columns: {missing}.")
        outputs = [self.rename.get(col, col) if self.rename else col for col in columns]
        if len(set(outputs)) != len(outputs):
            raise ValueError("Renamed output columns must be unique.")
        return columns

    def fit(self, X: object, y: object = None) -> Self:
        """Check selected columns and record input and output names.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Input containing every selected column. Other columns and a
            different input order are allowed. Pandas requires conversion enabled.
        y : object, default=None
            Ignored; accepted for scikit-learn compatibility.

        Returns
        -------
        self : ColumnSelector
            Fitted selector. Dtypes are not learned or validated.

        Raises
        ------
        TypeError
            If input, options, or column/rename parameter types are unsupported.
        ValueError
            If configuration is invalid or selected columns are missing.

        Notes
        -----
        Lazy input is checked through schema metadata without collecting rows.
        Output configuration does not change the return from fit.
        """
        frame = self._input(X)
        input_schema = frame_schema(frame)
        selected = self._validate_columns()
        missing = [column for column in selected if column not in input_schema]
        if missing:
            raise ValueError(f"Missing required columns: {missing}.")
        result = frame.select(selected)
        if self.rename is not None:
            result = result.rename(self.rename)
        self.feature_names_in_ = np.asarray(list(input_schema), dtype=object)
        self.n_features_in_ = len(input_schema)
        self.output_features_ = np.asarray(list(frame_schema(result)), dtype=object)
        self.columns_ = selected
        self._report(f"Fit complete; output columns: {self.output_features_.tolist()}")
        return self

    def fit_transform(self, X: object, y: object = None, **fit_params: object) -> Frame:
        """Fit and transform the same input using the configured return kind.

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
        X_transformed : polars.DataFrame or polars.LazyFrame
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

    def transform(self, X: object) -> Frame:
        """Select fitted columns, rename them, and verify output names.

        Parameters
        ----------
        X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
            Input containing the selected fitted columns. Unselected columns
            are ignored; input order and dtypes may differ from fitting. Pandas
            requires conversion enabled.

        Returns
        -------
        X_selected : polars.DataFrame or polars.LazyFrame
            Selected columns in configured order, including renames. Default
            preserves input kind; set_output can force eager or lazy output.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If fit has not completed successfully.
        TypeError
            If input or parameter types are unsupported.
        ValueError
            If selected columns are missing, configuration is invalid, or
            renamed output names differ from fitting.

        Notes
        -----
        Lazy metadata errors occur now; source/value errors can occur when a
        returned plan is collected. Explicit eager output collects within this
        method. Source and collection errors are not wrapped.
        """
        check_is_fitted(self, "output_features_")
        frame = self._input(X)
        input_schema = frame_schema(frame)
        missing = [column for column in self.columns_ if column not in input_schema]
        if missing:
            raise ValueError(f"Missing required columns: {missing}.")
        result = frame.select(self.columns_)
        if self.rename is not None:
            result = result.rename(self.rename)
        if list(frame_schema(result)) != self.output_features_.tolist():
            raise ValueError(
                "Output columns differ from the fitted selector state. "
                f"Expected {self.output_features_.tolist()}, "
                f"received {list(frame_schema(result))}."
            )
        return self._finish(result)


class FeatureSpec:
    """Describe a native Polars feature operation.

    Parameters
    ----------
    expression : polars.Expr or sequence of polars.Expr
        Single expression or nonempty sequence evaluated together, normalized to
        a tuple. Native output names are used unless ``alias`` is set. Sibling
        expressions cannot depend on outputs created within the same specification.
    alias : str or None, default=None
        Override the output name of a single expression. Multiple expressions
        or an expression expanding to multiple outputs cannot use this option.
    drop : sequence of str or None, default=None
        Columns to remove after evaluating the expressions. Missing names fail
        during execution; later specifications cannot use dropped columns.
    verbosity : {0, 1, 2}, default=0
        Print validation summaries at 1 and output/drop details at 2.
    name : str or None, default=None
        Descriptive label included in feature execution errors.

    Attributes
    ----------
    expression : tuple of polars.Expr
        Canonical expression representation, including any explicit alias.
    drop : tuple of str
        Columns to drop after evaluating this specification.
    name : str or None
        Optional debugging label.
    verbosity : int
        Configured diagnostic level, independent of the feature engineer.

    Raises
    ------
    TypeError
        If expressions, drop columns, containers, or verbosity have invalid types.
    ValueError
        If the expression sequence is empty, an explicit
        alias is ambiguous, output names are duplicated, a label is invalid,
        or verbosity is outside the supported levels.

    See Also
    --------
    FeatureEngineer : Execute an ordered sequence of specifications.

    Notes
    -----
    Existing columns may be overwritten deliberately. Expressions follow native
    Polars naming, so an additional alias is not required when the expression
    already has a determinable output name. Selector names and duplicates that
    depend on the input schema are resolved by the feature engineer.
    This class is an operation description, not a scikit-learn estimator. Its
    verbosity reports construction independently of an engineer's verbosity.
    Pickling preserves native expressions using Polars serialization, including
    Python callbacks with serializable state. Version mismatches emit
    InconsistentVersionWarning before state restoration. Compatible cross-version
    expressions are attempted using JSON; incompatible state raises
    SerializationError. Load trusted artifacts only and validate after upgrading.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import FeatureSpec
    >>> spec = FeatureSpec(
    ...     pl.col("price") * pl.col("quantity"), alias="notional"
    ... )
    >>> spec.expression[0].meta.output_name()
    'notional'
    """

    def __init__(
        self,
        expression: pl.Expr | Sequence[pl.Expr],
        *,
        alias: str | None = None,
        drop: Sequence[str] | None = None,
        name: str | None = None,
        verbosity: int = 0,
    ) -> None:
        validate_verbosity(verbosity)
        self.verbosity = verbosity
        if isinstance(expression, pl.Expr):
            canonical = (expression,)
        elif isinstance(expression, Sequence) and not isinstance(
            expression, (str, bytes)
        ):
            canonical = tuple(expression)
        else:
            raise TypeError(
                "expression must be a Polars Expr or a sequence of Polars Expr objects."
            )
        if not canonical:
            raise ValueError("expression must not be an empty sequence.")
        if any(not isinstance(expr, pl.Expr) for expr in canonical):
            raise TypeError("Every expression must be a Polars Expr.")
        if alias is not None:
            if not isinstance(alias, str) or not alias:
                raise ValueError("alias must be a nonempty string.")
            if len(canonical) != 1 or canonical[0].meta.has_multiple_outputs():
                raise ValueError("An explicit alias requires exactly one output.")
            canonical = (canonical[0].alias(alias),)
        names: list[str] = []
        for expr in canonical:
            if not expr.meta.has_multiple_outputs():
                output_name = expr.meta.output_name(raise_if_undetermined=False)
                if output_name is None:
                    raise ValueError("Expression needs a determinable name or alias.")
                names.append(output_name)
        if len(names) != len(set(names)):
            raise ValueError("Expressions have duplicate output names.")
        if drop is not None and (
            isinstance(drop, str)
            or not isinstance(drop, Sequence)
            or any(not isinstance(col, str) or not col for col in drop)
        ):
            raise TypeError("drop must be a sequence of nonempty column names.")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a nonempty string.")
        self.expression: tuple[pl.Expr, ...] = canonical
        self.drop: tuple[str, ...] = tuple(drop or ())
        self.name: str | None = name
        report(self, verbosity, f"Validated {len(canonical)} expression(s)")
        report(self, verbosity, f"Outputs: {names}; drop: {list(self.drop)}", level=2)

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        """Check creating versions before restoring serialized expressions."""
        return reduce_object(self, protocol)

    def __repr__(self) -> str:
        """Return the canonical specification representation."""
        return (
            f"FeatureSpec(expression={self.expression!r}, "
            f"drop={self.drop!r}, name={self.name!r})"
        )

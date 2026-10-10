"""Composable descriptions of native Polars feature operations."""

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.utils.validation import (
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)

from ._utils import NativeTransformer, to_polars


class ColumnSelector(NativeTransformer, auto_wrap_output_keys=None):
    """Select and optionally rename columns in a native Polars DataFrame.

    Parameters
    ----------
    columns : sequence of str
        Ordered sequence of input column names to keep.
    rename : dict of str to str, default=None
        Optional mapping from source names to new output names. Renamed output
        names must remain unique.
    convert_pandas : bool, default=False
        Allow pandas DataFrames and convert them to Polars. Requires pandas;
        some dtypes also require PyArrow.
    warn_on_conversion : bool, default=True
        Emit ``PandasConversionWarning`` when converting pandas input.
    """

    def __init__(
        self,
        columns: Sequence[str],
        *,
        rename: dict[str, str] | None = None,
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
    ) -> None:
        self.columns = columns
        self.rename = rename
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion

    def _input(self, X: object) -> pl.DataFrame:
        return to_polars(
            X,
            convert_pandas=self.convert_pandas,
            warn_on_conversion=self.warn_on_conversion,
        )

    def _validate_columns(self) -> tuple[str, ...]:
        if isinstance(self.columns, str) or not isinstance(self.columns, Sequence):
            raise TypeError("columns must be a sequence of nonempty column names.")
        columns = tuple(self.columns)
        if not columns or any(
            not isinstance(column, str) or not column for column in columns
        ):
            raise TypeError("columns must be a sequence of nonempty column names.")
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
        return columns

    def fit(self, X: object, y: object = None) -> "ColumnSelector":
        frame = self._input(X)
        selected = tuple(self._validate_columns())
        missing = [column for column in selected if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}.")
        result = frame.select(selected)
        if self.rename is not None:
            result = result.rename(self.rename)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.width
        self.output_features_ = np.asarray(result.columns, dtype=object)
        self.columns_ = selected
        return self

    def fit_transform(
        self, X: object, y: object = None, **fit_params: object
    ) -> pl.DataFrame:
        if fit_params:
            raise TypeError(
                f"Additional fit parameters are unsupported: {sorted(fit_params)}."
            )
        return self.fit(X, y).transform(X)

    def transform(self, X: object) -> pl.DataFrame:
        check_is_fitted(self, "output_features_")
        frame = self._input(X)
        missing = [column for column in self.columns_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}.")
        result = frame.select(self.columns_)
        if self.rename is not None:
            result = result.rename(self.rename)
        if list(result.columns) != self.output_features_.tolist():
            raise ValueError(
                "Output columns differ from the fitted selector state. "
                f"Expected {self.output_features_.tolist()}, received {result.columns}."
            )
        return result


class FeatureSpec:
    """Describe a native Polars feature operation.

    Parameters
    ----------
    expression : polars.Expr or None, default=None
        Single expression to evaluate. Specify exactly one of ``expression`` or
        ``expressions``. Its native output name is used unless ``alias`` is set.
    expressions : sequence of polars.Expr or None, default=None
        Nonempty sequence of expressions evaluated together. Expressions within
        one specification cannot reference columns created by siblings.
    alias : str or None, default=None
        Override the output name of a single expression. Multiple expressions
        or an expression expanding to multiple outputs cannot use this option.
    drop : sequence of str or None, default=None
        Columns to remove after evaluating the expressions. Missing names fail
        during execution; later specifications cannot use dropped columns.
    name : str or None, default=None
        Descriptive label included in feature execution errors.

    Attributes
    ----------
    expressions : tuple of polars.Expr
        Canonical expression representation, including any explicit alias.
    drop : tuple of str
        Columns to drop after evaluating this specification.
    name : str or None
        Optional debugging label.

    Raises
    ------
    TypeError
        If expressions, drop columns, or their containers have invalid types.
    ValueError
        If expression arguments conflict, expressions are empty, an explicit
        alias is ambiguous, output names are duplicated, or a label is invalid.

    See Also
    --------
    FeatureEngineer : Execute an ordered sequence of specifications.

    Notes
    -----
    Existing columns may be overwritten deliberately. Expressions follow native
    Polars naming, so an additional alias is not required when the expression
    already has a determinable output name. Selector names and duplicates that
    depend on the input schema are resolved by the feature engineer.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import FeatureSpec
    >>> spec = FeatureSpec(
    ...     pl.col("price") * pl.col("quantity"), alias="notional"
    ... )
    >>> spec.expressions[0].meta.output_name()
    'notional'
    """

    def __init__(
        self,
        expression: pl.Expr | None = None,
        *,
        expressions: Sequence[pl.Expr] | None = None,
        alias: str | None = None,
        drop: Sequence[str] | None = None,
        name: str | None = None,
    ) -> None:
        if (expression is None) == (expressions is None):
            raise ValueError("Specify exactly one of expression or expressions.")
        if expressions is not None and (
            isinstance(expressions, (str, pl.Expr))
            or not isinstance(expressions, Sequence)
        ):
            raise TypeError("expressions must be a sequence of Polars expressions.")
        canonical = (
            (expression,) if expression is not None else tuple(expressions or ())
        )
        if not canonical:
            raise ValueError("expressions must not be empty.")
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
        self.expressions: tuple[pl.Expr, ...] = canonical
        self.drop: tuple[str, ...] = tuple(drop or ())
        self.name: str | None = name

    def __repr__(self) -> str:
        """Return the canonical specification representation."""
        return (
            f"FeatureSpec(expressions={self.expressions!r}, "
            f"drop={self.drop!r}, name={self.name!r})"
        )

"""Composable descriptions of native Polars feature operations."""

from collections.abc import Sequence

import polars as pl


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

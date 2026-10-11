"""Shared public type aliases."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pandas import DataFrame as PandasDataFrame

ValidationMode: TypeAlias = Literal["strict", "semi-strict", "permissive"]
ExtraColumnPolicy: TypeAlias = Literal["preserve", "drop"]
Schema: TypeAlias = dict[str, pl.DataType]

OutputMode: TypeAlias = Literal["default", "polars", "polars-lazy", "pandas"]
FeatureNames: TypeAlias = Sequence[str] | NDArray[np.str_] | NDArray[np.object_]

Frame: TypeAlias = pl.DataFrame | pl.LazyFrame
OutputFrame: TypeAlias = Union[Frame, "PandasDataFrame"]


class FeatureSpecConfig(TypedDict, total=False):
    """Keyword arguments unpacked into FeatureSpec by FeatureEngineer.

    The expression argument is required at runtime. This type
    describes a regular dict; FeatureEngineer applies FeatureSpec validation.
    """

    expression: pl.Expr | Sequence[pl.Expr]
    alias: str | None
    drop: Sequence[str] | None
    name: str | None
    verbosity: int

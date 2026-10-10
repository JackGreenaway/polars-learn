"""Shared public type aliases."""

from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
import polars as pl
from numpy.typing import NDArray

ValidationMode: TypeAlias = Literal["strict", "semi-strict", "permissive"]
ExtraColumnPolicy: TypeAlias = Literal["preserve", "drop"]
Schema: TypeAlias = dict[str, pl.DataType]

OutputMode: TypeAlias = Literal["default", "polars"]
FeatureNames: TypeAlias = Sequence[str] | NDArray[np.str_] | NDArray[np.object_]

"""Explicit unchanged-column branches for preprocessing pipelines."""

from collections.abc import Sequence

from .feature_spec import ColumnSelector


class FeaturePassthrough(ColumnSelector, auto_wrap_output_keys=None):
    """Pass explicitly selected columns through without changing their contents.

    Parameters
    ----------
    columns : sequence of str
        Nonempty ordered sequence of unique, nonempty column names to keep.
        Other input columns are omitted. A bare string is invalid.
    convert_pandas : bool, default=False
        Permit pandas DataFrames and convert them to Polars. Some dtypes require
        PyArrow; install the pandas extra for both dependencies.
    warn_on_conversion : bool, default=True
        Emit PandasConversionWarning on enabled pandas conversion, independently
        of verbosity.
    verbosity : {0, 1, 2}, default=0
        Print progress at 1 and additional schema details at 2; 0 is silent.

    Attributes
    ----------
    feature_names_in_ : ndarray of str, dtype=object
        All training input names in input order, including unselected columns.
    n_features_in_ : int
        Number of all training input columns.
    columns_ : tuple of str
        Selected source names recorded by fit.
    output_features_ : ndarray of str, dtype=object
        Selected column names in the requested output order.

    Raises
    ------
    TypeError
        If columns or common options have invalid types.
    ValueError
        If selected names are duplicated or verbosity is outside 0, 1, and 2.
        Fit and transform also reject missing selected columns.

    See Also
    --------
    ColumnSelector : Select columns with optional renaming.

    Notes
    -----
    This is a scikit-learn transformer with inherited fit, transform,
    fit_transform, get_feature_names_out, get_params, set_params, and set_output.
    Fit checks that selected columns exist; transform requires fitting.
    Selected Polars values, names, dtypes, nulls, and row order are preserved.
    Input order and dtypes may change between calls; no dtype validation occurs.
    There is no renaming option. Refit after changing selected columns.

    Default returns preserve input kind: DataFrame input returns DataFrame and
    LazyFrame input returns an unevaluated LazyFrame. set_output can force
    "polars" (eager) or "polars-lazy" output. Inputs are never mutated.
    Source/schema-resolution and collection errors are not wrapped.

    A sequential pipeline step returns only its selected columns. To retain
    them alongside columns transformed in another branch, use ColumnTransformer
    with eager input, a branch for each operation, and Polars output.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_learn import FeaturePassthrough
    >>> X = pl.DataFrame({"id": [1, 2], "value": [3.0, 4.0]})
    >>> keep = FeaturePassthrough(["id"])
    >>> keep.fit_transform(X.lazy()).collect().to_dict(as_series=False)
    {'id': [1, 2]}
    >>> keep.get_feature_names_out().tolist()
    ['id']
    """

    # A fixed class default keeps rename out of estimator constructor state.
    rename: dict[str, str] | None = None

    def __init__(
        self,
        columns: Sequence[str],
        *,
        convert_pandas: bool = False,
        warn_on_conversion: bool = True,
        verbosity: int = 0,
    ) -> None:
        self.columns = columns
        self.convert_pandas = convert_pandas
        self.warn_on_conversion = warn_on_conversion
        self.verbosity = verbosity
        self._validate_options()
        self._validate_columns()

    def _validate_columns(self) -> tuple[str, ...]:
        if self.rename is not None:
            raise ValueError("FeaturePassthrough does not support renaming.")
        return super()._validate_columns()

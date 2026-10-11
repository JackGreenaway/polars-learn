# FeaturePassthrough

## About

An explicit unchanged-feature step selects named columns while preserving their Polars values, names, dtypes, nulls, and row order. It communicates the selection contract more clearly than an anonymous passthrough string.

## Why use it?

Use it for features that need no transformation, particularly a ColumnTransformer branch alongside transformed features. In a sequential pipeline it returns only the named columns, so the next step receives that subset.

## Example: input to output

### Input

| id (Int64) | flag (Boolean) | unused (Int64) |
| --- | --- | --- |
| 7 | true | 9 |
| null | false | 8 |

### Code

```python
import polars as pl

from polars_learn import FeaturePassthrough

raw = pl.DataFrame({"id": [7, None], "flag": [True, False], "unused": [9, 8]})
keep = FeaturePassthrough(["flag", "id"])
prepared = keep.fit_transform(raw)
assert prepared.equals(raw.select("flag", "id"))
assert prepared.to_dict(as_series=False) == {"flag": [True, False], "id": [7, None]}
assert dict(prepared.schema) == {"flag": pl.Boolean, "id": pl.Int64}
```

### Output

| flag (Boolean) | id (Int64) |
| --- | --- |
| true | 7 |
| false | null |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Production considerations

There is no rename or cast option. Input order and dtypes may change without rejection; add schema validation when those changes must fail. See [parallel branch examples](../user-guide/selection.md) for combining unchanged and transformed features.

## Reference

```text
FeaturePassthrough(
    columns,
    *,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

`columns` is a nonempty sequence of unique, nonempty string names, in output
order. A bare string is invalid. Only the selected columns are returned, with
Polars values, names, dtypes, nulls, and row order preserved. Unselected input
columns are omitted. There is no `rename` parameter; use `ColumnSelector` for
renaming.

This class specializes `ColumnSelector` and shares its methods and fitted
attributes, including `columns_`. Fitting checks selected columns exist;
transform requires fitting. Input order and dtypes may change without rejection.
`feature_names_in_` and `n_features_in_` describe the whole training input,
including unselected columns; `output_features_` contains unchanged selected names.
The common input conversion, verbosity, cloning, and output policies apply.

Invalid selection types raise `TypeError`; duplicate names, missing columns,
and invalid verbosity levels raise `ValueError`. Transform before fitting raises
`NotFittedError`. `get_params` exposes only its constructor parameters, with no
rename option. Source/schema-resolution and collection errors are not wrapped.
Refit after changing `columns` through `set_params`.

A sequential pipeline step selects its output subset. Use an eager
`ColumnTransformer` branch when selected unchanged columns should be combined
with another branch's transformed columns. Configure the outer transformer
with `set_output(transform="polars")` to preserve separate column dtypes, and
`verbose_feature_names_out=False` to keep unprefixed names when they are unique.
See the [branch example](../user-guide/selection.md).

```python
import polars as pl

from polars_learn import FeaturePassthrough

X = pl.DataFrame({"id": [1, None], "flag": [True, False], "unused": [9, 8]})
keep = FeaturePassthrough(["flag", "id"])
plan = keep.fit_transform(X.lazy())
assert plan.collect().equals(X.select("flag", "id"))
assert keep.get_feature_names_out().tolist() == ["flag", "id"]
```

## Shared methods and configuration

All transformers share fit, transform, fit_transform, get_feature_names_out, get_params, set_params, and set_output. Their common constructor options are convert_pandas, warn_on_conversion, and verbosity. Consult the [shared reference](../api.md#shared-transformer-behavior) for signatures, return types, fitted attributes, cloning, and failure behavior.

See [persistence](../user-guide/persistence.md) for serialization and warning controls, and [diagnostics](../user-guide/diagnostics.md) for verbosity and error timing.

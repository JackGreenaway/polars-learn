# ColumnSelector

## About

A named selection step establishes the exact feature subset and order, with optional renaming. It makes the model-facing layout visible in the pipeline instead of leaving selection to an external helper.

## Why use it?

Use it to remove unused raw or intermediate columns, reorder features deliberately, or adapt names for downstream consumers. Use FeaturePassthrough when selection should communicate that names and values remain unchanged.

## Example: input to output

### Input

| price | quantity | request_id |
| --- | --- | --- |
| 10.0 | 2 | 7 |
| 20.0 | 3 | 8 |

### Code

```python
import polars as pl

from polars_learn import ColumnSelector

raw = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3], "request_id": [7, 8]})
selector = ColumnSelector(["quantity", "price"], rename={"quantity": "qty"})
prepared = selector.fit_transform(raw)
assert prepared.to_dict(as_series=False) == {"qty": [2, 3], "price": [10.0, 20.0]}
assert selector.get_feature_names_out().tolist() == ["qty", "price"]
```

### Output

| qty | price |
| --- | --- |
| 2 | 10.0 |
| 3 | 20.0 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Production considerations

Selection does not validate dtype drift or cast values. Pair it with a schema step when dtype enforcement matters. Selected columns must exist, and output names must be unique. Unselected columns are omitted.

## Reference

```text
ColumnSelector(
    columns,
    *,
    rename=None,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

| Parameter | Contract |
| --- | --- |
| `columns` | Nonempty sequence of unique, nonempty string column names. Determines output order. A bare string is invalid. |
| `rename` | Optional dict mapping selected source names to nonempty string output names. Keys must be selected columns, and final output names must be unique, including names left unchanged. |

Fitting checks that every selected column exists and records its output name.
The input can contain other columns, and its order need not match `columns`.
Transform selects the fitted `columns_`, applies the rename mapping, and verifies
that output names still match fitting. Input and output dtypes are not checked.
Unselected columns are omitted.

Additional fitted attribute: `columns_`, a tuple of selected source names.
`output_features_` lists the selected output names after renaming.

Wrong argument types raise `TypeError`; duplicate names, invalid rename mappings,
missing selected columns, or changed output names raise `ValueError`.
Source query errors can propagate from Polars.

```python
import polars as pl

from polars_learn import ColumnSelector

X = pl.DataFrame({"a": [1], "b": [2], "unused": [3]})
selector = ColumnSelector(["b", "a"], rename={"b": "value"})
result = selector.fit_transform(X.lazy())
assert result.collect().columns == ["value", "a"]
assert selector.get_feature_names_out().tolist() == ["value", "a"]
```

## Shared methods and configuration

All transformers share fit, transform, fit_transform, get_feature_names_out, get_params, set_params, and set_output. Their common constructor options are convert_pandas, warn_on_conversion, and verbosity. Consult the [shared reference](../api.md#shared-transformer-behavior) for signatures, return types, fitted attributes, cloning, and failure behavior.

See [persistence](../user-guide/persistence.md) for serialization and warning controls, and [diagnostics](../user-guide/diagnostics.md) for verbosity and error timing.

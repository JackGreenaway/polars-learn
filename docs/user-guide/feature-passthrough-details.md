# FeaturePassthrough workflows

## About

An explicit unchanged-feature step selects named columns while preserving their Polars values, names, dtypes, nulls, and row order. It communicates the selection contract more clearly than an anonymous passthrough string.

## Why use it?

Use it for features that need no transformation, particularly a ColumnTransformer branch alongside transformed features. In a sequential pipeline it returns only the named columns, so the next step receives that subset.

## Examples

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

## Practical notes

There is no rename or cast option. Input order and dtypes may change without rejection; add schema validation when those changes must fail. See [parallel branch examples](../user-guide/selection.md) for combining unchanged and transformed features.
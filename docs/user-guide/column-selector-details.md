# ColumnSelector workflows

## About

A named selection step establishes the exact feature subset and order, with optional renaming. It makes the model-facing layout visible in the pipeline instead of leaving selection to an external helper.

## Why use it?

Use it to remove unused raw or intermediate columns, reorder features deliberately, or adapt names for downstream consumers. Use FeaturePassthrough when selection should communicate that names and values remain unchanged.

## Examples

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

## Practical notes

Selection does not validate dtype drift or cast values. Pair it with a schema step when dtype enforcement matters. Selected columns must exist, and output names must be unique. Unselected columns are omitted.
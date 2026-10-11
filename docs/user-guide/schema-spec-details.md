# SchemaSpec workflows

## About

A declared schema is an executable input contract. SchemaSpec checks that required names, dtypes, and order match your application definition, instead of inferring the definition from whichever training frame happens to arrive.

## Why use it?

Use it when the upstream interface is known in advance, such as a documented scoring request schema. Allow extra transport metadata while dropping it from model input, or reject extras entirely.

## Examples

### Input

| request_id (Int64) | price (Float64) | quantity (Int64) |
| --- | --- | --- |
| 7 | 10.0 | 2 |
| 8 | 20.0 | 3 |

### Code

```python
import polars as pl

from polars_learn import SchemaSpec

raw = pl.DataFrame({"request_id": [7, 8], "price": [10.0, 20.0], "quantity": [2, 3]})
contract = SchemaSpec(
    {"price": pl.Float64, "quantity": pl.Int64},
    allow_extra_columns=True,
    extra_columns="drop",
)
prepared = contract.fit_transform(raw)
assert prepared.equals(raw.select("price", "quantity"))
assert contract.schema_ == {"price": pl.Float64, "quantity": pl.Int64}
assert contract.get_feature_names_out().tolist() == ["price", "quantity"]
```

### Output

| price (Float64) | quantity (Int64) |
| --- | --- |
| 10.0 | 2 |
| 20.0 | 3 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

This class does not cast: wrong required dtypes are errors. It validates names and types, not nullability, value ranges, or business rules. Use SchemaValidator for a learned schema and explicit cast policy.
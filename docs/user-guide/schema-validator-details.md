# SchemaValidator workflows

## About

Training data establishes the schema that new observations must satisfy. SchemaValidator learns names and dtypes during fit, then checks incoming structure and applies the selected dtype policy during transform.

## Why use it?

Use it to detect prediction-time schema drift, keep required feature order stable, and make conversions a deliberate policy. Strict rejects dtype changes; semi-strict permits defined widening; permissive attempts strict Polars casts.

## Examples

### Input

Training: `price: Float64`, `quantity: Int64`.

| quantity (String) | price (String) |
| --- | --- |
| "2" | "15.0" |
| "4" | "25.0" |

### Code

```python
import polars as pl

from polars_learn import SchemaValidator

train = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
incoming = pl.DataFrame({"quantity": ["2", "4"], "price": ["15.0", "25.0"]})
validator = SchemaValidator(mode="permissive", enforce_column_order=False).fit(train)
prepared = validator.transform(incoming)
assert prepared.to_dict(as_series=False) == {"price": [15.0, 25.0], "quantity": [2, 4]}
assert dict(prepared.schema) == validator.schema_
```

### Output

| price (Float64) | quantity (Int64) |
| --- | --- |
| 15.0 | 2 |
| 25.0 | 4 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

Permissive casts may lose information even when successful; choose the policy for your data contract. Missing required columns still fail. Lazy value checks execute at collection, and nullability is not learned. Consult the [class reference](../classes/schema-validator.md#reference) and [casting policy details](../user-guide/schemas.md#learn-the-training-schema) for the exact widening and precision rules.
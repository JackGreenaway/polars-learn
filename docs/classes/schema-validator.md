# SchemaValidator

## About

Training data establishes the schema that new observations must satisfy. SchemaValidator learns names and dtypes during fit, then checks incoming structure and applies the selected dtype policy during transform.

## Why use it?

Use it to detect prediction-time schema drift, keep required feature order stable, and make conversions a deliberate policy. Strict rejects dtype changes; semi-strict permits defined widening; permissive attempts strict Polars casts.

## Example: input to output

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

## Production considerations

Permissive casts may lose information even when successful; choose the policy for your data contract. Missing required columns still fail. Lazy value checks execute at collection, and nullability is not learned. Consult the reference below for the exact widening and precision rules.

## Reference

```text
SchemaValidator(
    mode="strict",
    *,
    allow_extra_columns=False,
    enforce_column_order=True,
    extra_columns="preserve",
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

| Parameter | Contract |
| --- | --- |
| `mode` | `"strict"`, `"semi-strict"`, or `"permissive"`. Defaults to `"strict"`; `None` is invalid. |
| `allow_extra_columns` | Bool; allow input names not observed during fitting. All fitted names remain required. |
| `enforce_column_order` | Bool; require the relative order of fitted columns to match training. False normalizes output order. |
| `extra_columns` | `"preserve"` or `"drop"`; only affects extras that are allowed. |

Fit learns every input column's name and dtype, regardless of mode. Additional
fitted attribute: `schema_`, a dict in training column order. `output_features_`
contains those required names, excluding inference-only extras.

Transform validates structure and dtypes, collecting multiple schema mismatches
into one `SchemaValidationError`. Policies for differing dtypes:

| Mode | Accepted conversion |
| --- | --- |
| Strict | None; dtype must equal the fitted dtype. |
| Semi-strict | Integer widening based on entire dtype ranges; Float32 to Float64; integer to Float64 only when every evaluated non-null value round-trips exactly. |
| Permissive | Any successful Polars `cast(strict=True)` to the fitted dtype; successful casts can be lossy. |

Semi-strict rejects integer narrowing even when current values would fit,
Float64 to Float32, string to numeric, and Null dtype to a different dtype.
Compatible null values are preserved. Nullability is not learned. Permissive
mode can cast Null dtype columns, but invalid cast values raise rather than
becoming null silently.

With enforced order and preserved extras, output keeps input column order.
With order enforcement disabled, output uses required training order followed
by preserved extras in input order. Dropping extras returns required columns
in training order.

Lazy structural and dtype-policy failures raise during `transform`. Lazy
permissive casts execute when the plan is collected and can raise native Polars
errors. Lazy semi-strict integer-to-float value checks also run at execution and
raise `SchemaValidationError` on precision loss. An explicit eager output request
wraps Polars collection errors in `SchemaValidationError` with their causes.

```python
import polars as pl

from polars_learn import SchemaValidator

train = pl.DataFrame({"x": [1]})
incoming = pl.DataFrame({"extra": [7], "x": ["2"]})
validator = SchemaValidator(
    mode="permissive", allow_extra_columns=True, enforce_column_order=False
).fit(train.lazy())
result = validator.transform(incoming.lazy()).collect()
assert result.columns == ["x", "extra"]
assert result["x"].to_list() == [2]
assert validator.get_feature_names_out().tolist() == ["x"]
```

## Shared methods and configuration

All transformers share fit, transform, fit_transform, get_feature_names_out, get_params, set_params, and set_output. Their common constructor options are convert_pandas, warn_on_conversion, and verbosity. Consult the [shared reference](../api.md#shared-transformer-behavior) for signatures, return types, fitted attributes, cloning, and failure behavior.

See [persistence](../user-guide/persistence.md) for serialization and warning controls, and [diagnostics](../user-guide/diagnostics.md) for verbosity and error timing.

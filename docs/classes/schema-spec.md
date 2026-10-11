# SchemaSpec

## About

A declared schema is an executable input contract. SchemaSpec checks that required names, dtypes, and order match your application definition, instead of inferring the definition from whichever training frame happens to arrive.

## Why use it?

Use it when the upstream interface is known in advance, such as a documented scoring request schema. Allow extra transport metadata while dropping it from model input, or reject extras entirely.

## Example: input to output

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

## Production considerations

This class does not cast: wrong required dtypes are errors. It validates names and types, not nullability, value ranges, or business rules. Use SchemaValidator for a learned schema and explicit cast policy.

## Reference

```text
SchemaSpec(
    schema,
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
| `schema` | Nonempty dict from nonempty string names to Polars dtypes, in required column order. Accepts dtype classes such as `pl.Int64` and instances such as `pl.Datetime("ms")`. |
| `allow_extra_columns` | Bool; whether input may include names absent from the explicit schema. |
| `enforce_column_order` | Bool; require the relative order of required input columns to match the schema's key order. |
| `extra_columns` | `"preserve"` or `"drop"`; treatment of extras if allowed. Does not itself allow extras. |

Fit and transform both require all schema columns and exact required dtypes;
there is no casting policy. Output always orders required columns by schema key
order. Preserved extras follow in their input order.

Fitting checks the declared contract, not an inferred replacement. Additional
fitted attribute: `schema_`, a copy of the required schema.
`output_features_` contains every fitted output column, including preserved extras.
Subsequent output names and order must match exactly. Extra dtypes are not checked.
Use `extra_columns="drop"` for a fixed output set containing only required columns.

Missing required columns, unexpected extras, disallowed order, wrong required
dtypes, and output-name mismatches raise `ValueError`. An invalid schema
container or empty schema raises `ValueError`; invalid schema keys/values raise
`TypeError`. Source/collection errors are not wrapped.

```python
import polars as pl

from polars_learn import SchemaSpec

X = pl.DataFrame({"extra": [9], "x": [1]})
contract = SchemaSpec({"x": pl.Int64}, allow_extra_columns=True, extra_columns="drop")
assert contract.fit_transform(X).columns == ["x"]
```

## Shared methods and configuration

All transformers share fit, transform, fit_transform, get_feature_names_out, get_params, set_params, and set_output. Their common constructor options are convert_pandas, warn_on_conversion, and verbosity. Consult the [shared reference](../api.md#shared-transformer-behavior) for signatures, return types, fitted attributes, cloning, and failure behavior.

See [persistence](../user-guide/persistence.md) for serialization and warning controls, and [diagnostics](../user-guide/diagnostics.md) for verbosity and error timing.

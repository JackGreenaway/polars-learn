# Schema contracts and drift

`SchemaSpec` checks an explicit, nonempty schema. Required columns must have exactly the specified dtypes; this class does not cast them. Both its `fit` and `transform` validate input, and `transform` requires a successful fit.

`ColumnSelector` returns just the columns you choose, in the requested order. Unselected columns are ignored, input order may differ, and dtypes are not validated. Selected and renamed output names must be unique.

```python
import polars as pl

from polars_learn import ColumnSelector, SchemaSpec

X = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
contract = SchemaSpec({"price": pl.Float64, "quantity": pl.Int64})
validated = contract.fit_transform(X)

selector = ColumnSelector(["quantity", "price"], rename={"quantity": "qty"})
selected = selector.fit_transform(validated)
assert selected.columns == ["qty", "price"]
```

For `SchemaSpec`, extra columns are rejected by default. If extras are allowed, `extra_columns="drop"` produces only the required columns; `"preserve"` puts required columns first and then extras in their input order. Preserved extra column names become part of the fitted output names. Later transforms must produce those same names in the same order; their dtypes are not checked. Use `SchemaValidator` if inference may introduce new extra columns.

## Learn the training schema

`SchemaValidator.fit` learns all training columns as required columns. Transform checks them against the learned dtypes according to `mode`:

| Mode                 | Treatment of a different input dtype                                                |
| -------------------- | ----------------------------------------------------------------------------------- |
| `"strict"` (default) | Reject it; no casting.                                                              |
| `"semi-strict"`      | Allow only the lossless widening policy below.                                      |
| `"permissive"`       | Attempt a Polars cast with `strict=True`; some successful casts may lose precision. |

Semi-strict permits integer widening when the **entire source dtype's range** fits inside the fitted integer dtype's range, including compatible unsigned to signed widening. Small observed values alone do not make narrowing acceptable. It also permits Float32 to Float64, and integer to Float64 only when every non-null value round-trips exactly. For example, `2**53` and `2**54` are accepted, but `2**53 + 1` is rejected. String to numeric conversion, narrowing, and Null dtype to another dtype are rejected in this mode.

```python
import polars as pl

from polars_learn import SchemaValidator

train = pl.DataFrame({"x": pl.Series([1], dtype=pl.Int64)})
incoming = pl.DataFrame({"x": pl.Series([2, None], dtype=pl.Int32)})
validator = SchemaValidator(mode="semi-strict").fit(train)
result = validator.transform(incoming)
assert result.schema["x"] == pl.Int64
assert result["x"].to_list() == [2, None]
```

Required columns must always be present. Extra columns fail unless `allow_extra_columns=True`; allowing extras does not make fitted columns optional. `extra_columns="drop"` removes allowed extras, while the default `"preserve"` keeps them. The drop setting alone does not allow extras.

`enforce_column_order=True` checks the relative order of required columns, so allowed extras can appear between them. With `enforce_column_order=False`, output puts fitted columns first, followed by preserved extras in their input order. `get_feature_names_out()` lists required fitted columns, excluding inference-only extras. Preserving extras can change the output width; drop them before an estimator that expects a fixed number of features.

Null values are preserved when the dtype is compatible. Nullability constraints are not learned. In permissive mode, a column with Null dtype can be cast to the fitted dtype. Failed strict casts raise; they do not silently replace invalid values with nulls. Choose the validation policy through `mode`: `"strict"`, `"semi-strict"`, or `"permissive"`. The default is `"strict"`.

## Schema drift at prediction time

Schema drift means the structure of incoming data differs from the contract used during training: columns may be missing, added, reordered, or represented with different dtypes. Put validation before feature engineering in a pipeline so these changes are handled before the model receives its features. A change in value distributions or business meaning is a separate concern and needs additional checks.

### Reject unexpected structure and dtype changes

Fit a validator on training data once, then reuse it for incoming batches. The default strict policy rejects each of these four changes:

```python
import polars as pl

from polars_learn import SchemaValidationError, SchemaValidator

train = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
validator = SchemaValidator().fit(train)

incoming_batches = {
    "missing quantity": pl.DataFrame({"price": [15.0]}),
    "quantity changed to strings": pl.DataFrame({"price": [15.0], "quantity": ["2"]}),
    "unexpected debug column": pl.DataFrame(
        {"price": [15.0], "quantity": [2], "debug": [True]}
    ),
    "reordered columns": pl.DataFrame({"quantity": [2], "price": [15.0]}),
}

for change, incoming in incoming_batches.items():
    try:
        validator.transform(incoming)
    except SchemaValidationError as error:
        print(f"Rejected {change}: {error}")
    else:
        raise AssertionError(f"Expected rejection: {change}")
```

| Incoming change | Why the default policy rejects it |
| --- | --- |
| Missing `quantity` | Every training column is required; the validator does not invent missing values. |
| String `quantity` instead of Int64 | Strict mode requires the fitted dtype and does not cast. |
| New `debug` column | Extra columns are disallowed unless explicitly enabled. |
| `quantity` before `price` | Required-column order must match training unless order enforcement is disabled. |

Keep the fitted validator's contract. Calling `fit` again on each incoming batch would replace it with the new batch's schema rather than detect drift against training.

### Normalize changes that your contract allows

An upstream system may send an Int32 quantity instead of Int64, reorder the columns, and attach a request identifier. If those changes are acceptable, configure them explicitly. This validator allows lossless widening, restores the training column order, and drops permitted extra columns:

```python
import polars as pl

from polars_learn import SchemaValidator

train = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
validator = SchemaValidator(
    mode="semi-strict",
    enforce_column_order=False,
    allow_extra_columns=True,
    extra_columns="drop",
).fit(train)

incoming = pl.DataFrame(
    {
        "request_id": ["request-1"],
        "quantity": pl.Series([2], dtype=pl.Int32),
        "price": [15.0],
    }
)
result = validator.transform(incoming)
assert result.columns == ["price", "quantity"]
assert result.schema == {"price": pl.Float64, "quantity": pl.Int64}
assert result.to_dict(as_series=False) == {"price": [15.0], "quantity": [2]}
```

Before validation the columns are `request_id`, `quantity: Int32`, and `price: Float64`. After validation they are `price: Float64` and `quantity: Int64`, in training order. This policy still rejects missing required columns, narrowing casts, and string-to-number changes. Set `allow_extra_columns=True` as well as `extra_columns="drop"`: dropping extras does not itself permit them.

### Enforce a declared application contract

Use `SchemaSpec` when a contract is defined independently of training. Unlike the previous example, it never casts Int32 to Int64, even when widening would be safe:

```python
import polars as pl

from polars_learn import SchemaSpec

contract = SchemaSpec({"price": pl.Float64, "quantity": pl.Int64})
contract.fit(pl.DataFrame({"price": [10.0], "quantity": [2]}))

incoming = pl.DataFrame({"price": [15.0], "quantity": pl.Series([2], dtype=pl.Int32)})
try:
    contract.transform(incoming)
except ValueError as error:
    print(f"Rejected a change to the declared dtype: {error}")
else:
    raise AssertionError("The declared quantity dtype must remain Int64")
```

This is useful when the data producer must honor an exact API or ingestion contract. `SchemaSpec` reports violations with `ValueError`; `SchemaValidator` reports learned-schema and casting-policy violations with `SchemaValidationError`.

### Stop drift before prediction in a fitted pipeline

The schema validator can guard the full preparation and prediction path. A deployment calls `predict` with raw data; validation runs before feature generation and the regressor:

```python
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from polars_learn import (
    ColumnSelector,
    FeatureEngineer,
    FeatureSpec,
    SchemaValidationError,
    SchemaValidator,
)

pipeline = Pipeline(
    [
        ("schema", SchemaValidator()),
        (
            "features",
            FeatureEngineer(
                FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
            ),
        ),
        ("model_columns", ColumnSelector(["notional"])),
        ("model", LinearRegression()),
    ]
)
train = pl.DataFrame({"price": [10.0, 20.0, 30.0], "quantity": [2, 3, 4]})
pipeline.fit(train, np.array([2.0, 6.0, 12.0]))

valid = pl.DataFrame({"price": [15.0], "quantity": [2]})
assert np.allclose(pipeline.predict(valid), [3.0])

drifted = pl.DataFrame({"price": [15.0], "quantity": ["2"]})
try:
    pipeline.predict(drifted)
except SchemaValidationError as error:
    print(f"Prediction blocked by raw-input schema drift: {error}")
else:
    raise AssertionError("Prediction should reject the changed quantity dtype")
```

The valid batch produces a prediction. The string-valued quantity is rejected at the first step, before `notional` is computed or passed to the model. Save and reuse the whole fitted pipeline so this guard remains part of deployment.

These examples use eager frames, so the demonstrated errors occur during the call. With lazy inputs, structural errors are still checked through metadata, but value-dependent casting errors can be deferred until collection. Exact names and dtypes do not detect a unit change such as dollars to cents, a changed distribution, or invalid business values; add domain validation and monitoring for those risks.

## Choosing a production policy

Use `SchemaSpec` when an input schema is already defined by your application. It checks exact required dtypes and never casts. Use `SchemaValidator` when the training frame defines the input schema. Strict mode catches dtype changes; semi-strict permits defined widening with value checks where necessary; permissive mode attempts strict Polars casts and may accept lossy conversions. Permissive means a cast policy, not permission to silently insert missing columns or discard bad values.

Extra-column allowance and output treatment are separate options. `allow_extra_columns=True` permits extras; `extra_columns="drop"` provides a fixed model-facing subset. Neither class learns nullability, value ranges, uniqueness, category vocabularies, or business constraints. Add explicit validation for those requirements. If your model needs a fixed feature set, avoid passing inference-only extras to it.

See the [SchemaSpec](../api.md#schemaspec) and [SchemaValidator](../api.md#schemavalidator) reference for ordering and output-name differences.

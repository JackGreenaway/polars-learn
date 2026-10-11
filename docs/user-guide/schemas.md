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

## Choosing a production policy

Use `SchemaSpec` when an input schema is already defined by your application. It checks exact required dtypes and never casts. Use `SchemaValidator` when the training frame defines the input schema. Strict mode catches dtype changes; semi-strict permits defined widening with value checks where necessary; permissive mode attempts strict Polars casts and may accept lossy conversions. Permissive means a cast policy, not permission to silently insert missing columns or discard bad values.

Extra-column allowance and output treatment are separate options. `allow_extra_columns=True` permits extras; `extra_columns="drop"` provides a fixed model-facing subset. Neither class learns nullability, value ranges, uniqueness, category vocabularies, or business constraints. Add explicit validation for those requirements. If your model needs a fixed feature set, avoid passing inference-only extras to it.

See the [SchemaSpec](../api.md#schemaspec) and [SchemaValidator](../api.md#schemavalidator) reference for ordering and output-name differences.

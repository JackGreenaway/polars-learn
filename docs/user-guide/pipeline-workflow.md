# Preparing raw data in a pipeline

## Why pipeline scaling but leave feature generation outside?

**If we put standard scaling and polynomial features inside a fitted pipeline, why not put full feature generation there too?** Scaling and polynomial expansion are feature preparation. So are calculating transaction value, deriving a spread, extracting calendar features, and selecting model inputs. Keeping the complete preparation path with the estimator gives training and deployment the same entry point: raw observations in, predictions out.

### A separate feature function leaves callers responsible

A custom function is a useful way to express Polars transformations. This example trains a regressor on `price` and a derived `notional`:

```python
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression


def prepare_features(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.with_columns(
        (pl.col("price") * pl.col("quantity")).alias("notional")
    ).select("price", "notional")


raw_train = pl.DataFrame({"price": [10.0, 20.0, 30.0], "quantity": [2, 3, 4]})
y = np.array([2.0, 6.0, 12.0])
model = LinearRegression().fit(prepare_features(raw_train), y)

raw_predict = pl.DataFrame({"price": [15.0], "quantity": [2]})
# Every caller must remember this preparation and use the same function version.
predictions = model.predict(prepare_features(raw_predict))
assert np.allclose(predictions, [3.0])
```

This works when every caller applies the correct function. The risk is that preparation and the saved estimator are managed separately. Saving only `model` leaves the feature calculation outside that artifact. A deployment might omit the function, use an older definition, or select different columns. Calling `model.predict(raw_predict)` would supply `quantity` where the model expects `notional`; depending on the receiving estimator's checks, a mismatch can fail or produce incorrect predictions. A changed calculation can also keep the same names and dtypes while changing the feature's meaning.

Cross-validation has the same boundary: it can only fit the steps included in the object being evaluated. This example's arithmetic is deterministic, but preprocessing that learns statistics must be fitted within each training fold. Separate preparation code needs its own versioning, testing, and explicit handling of those learned quantities.

### Put the feature definitions alongside the estimator

`polars-learn` wraps native Polars expressions in estimator-style components. The pipeline below contains the raw-input contract, feature calculation, selected model inputs, scaling, and regressor:

```python
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polars_learn import ColumnSelector, FeatureEngineer, FeatureSpec, SchemaValidator

pipeline = Pipeline(
    [
        ("raw_schema", SchemaValidator()),
        (
            "features",
            FeatureEngineer(
                FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
            ),
        ),
        ("model_columns", ColumnSelector(["price", "notional"])),
        ("scale", StandardScaler().set_output(transform="polars")),
        ("model", LinearRegression()),
    ]
)

raw_train = pl.DataFrame({"price": [10.0, 20.0, 30.0], "quantity": [2, 3, 4]})
pipeline.fit(raw_train, np.array([2.0, 6.0, 12.0]))

# The caller supplies raw observations; the fitted pipeline prepares them.
raw_predict = pl.DataFrame({"price": [15.0], "quantity": [2]})
predictions = pipeline.predict(raw_predict)
assert np.allclose(predictions, [3.0])
assert pipeline[:-1].get_feature_names_out().tolist() == ["price", "notional"]
```

`fit` applies the declared preparation and fits the scaler and regressor. `predict` validates the raw input, recreates `notional`, selects the fitted model layout, and applies the fitted scaler before predicting. Save the **whole fitted pipeline** to carry its feature definitions and fitted components into deployment. See [saving and loading](persistence.md) and [cross-validation](../examples/cross-validation.md).

The benefit is a shared, inspectable preparation path. Schema checks detect structural mismatches; they do not prove that values have the right business meaning or automatically prevent leakage. Polars expressions do not learn and freeze statistics. Keep learned preprocessing such as scaling, imputation, or training-derived aggregates in appropriate fitted components, and validate the pipeline before deployment.

### How each class addresses a concrete risk

| Class | Example issue in training or deployment | How the component helps |
| --- | --- | --- |
| [`FeatureSpec`](../classes/feature-spec.md) | A transaction-value expression, output name, and column drops are scattered across separate scripts. | Define `price * quantity`, the `notional` alias, and optional drops together as one reusable specification. Construction validates the definition; `FeatureEngineer` executes it. |
| [`FeatureEngineer`](../classes/feature-engineer.md) | Training computes `notional` and then `log_notional`, but serving forgets a stage or applies them in a different order. | Store both specs in entry order inside the pipeline. Fit records the feature schema; prediction replays the stages and checks output names, order, and dtypes. |
| [`SchemaSpec`](../classes/schema-spec.md) | An application requires `price: Float64` and `quantity: Int64`, but an upstream producer changes `quantity` to strings. | Declare that fixed contract before feature calculation. Exact dtype checks reject the changed input instead of implicitly accepting a new representation; no casting occurs. |
| [`SchemaValidator`](../classes/schema-validator.md) | The schema is determined by the training dataset, and a deployment batch arrives with missing columns or changed dtypes. | Learn the required names, order, and dtypes during fit. Transform rejects missing columns and handles dtype changes according to the chosen strict, semi-strict, or permissive policy. |
| [`ColumnSelector`](../classes/column-selector.md) | A helper column reaches the estimator, or callers supply model features in a different order. | Select `['price', 'notional']` in that exact order, omit other columns, and optionally rename selected outputs. Fitted output names are checked; add schema validation for dtype checks. |
| [`FeaturePassthrough`](../classes/feature-passthrough.md) | A preprocessing branch needs to retain numeric flags unchanged alongside separately scaled numeric features. | Explicitly select the unchanged flags in a `ColumnTransformer` branch and combine them with the scaled branch. Native Polars values and dtypes are retained; downstream model compatibility still needs checking. |

Use the classes together according to your contract. Choose `SchemaSpec` for a known schema or `SchemaValidator` for a learned one; choose `ColumnSelector` when renaming is needed or `FeaturePassthrough` for unchanged selected features. The [parallel branch example](selection.md) shows how to combine transformed and unchanged columns.

### Schema drift: training inputs change in deployment

Suppose training used `price: Float64` and `quantity: Int64`, but a deployment starts sending `quantity` as strings. Names alone still look correct. A fitted `SchemaValidator` catches the dtype change before preparation reaches the model:

```python
import polars as pl

from polars_learn import SchemaValidationError, SchemaValidator

train = pl.DataFrame({"price": [10.0], "quantity": [2]})
validator = SchemaValidator().fit(train)
incoming = pl.DataFrame({"price": [15.0], "quantity": ["2"]})

try:
    validator.transform(incoming)
except SchemaValidationError as error:
    print(f"Schema drift rejected: {error}")
else:
    raise AssertionError("Strict validation should reject the changed dtype")
```

Choose an exact declared contract with `SchemaSpec`, or learn the training contract with `SchemaValidator`. Configure acceptable casts, order changes, and extras explicitly. See the [schema-drift examples](schemas.md#schema-drift-at-prediction-time) for rejected batches, safe normalization, and validation inside a prediction pipeline. Schema checks address structure and dtypes; business meaning and distribution drift need separate checks.


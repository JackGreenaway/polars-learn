<div class="docs-hero" markdown="1">

# polars-learn

<p class="lead">Feature engineering in Polars. Prediction pipelines in scikit-learn.</p>

The power comes from the Polars API: its expressive column operations, selectors, and eager and lazy execution. `polars-learn` packages those capabilities into reusable scikit-learn-style components, so you can define data preparation once, fit it with your model, and apply the same transformations at prediction time.

- Powered by the Polars expression API for feature engineering
- Compatible with scikit-learn estimators and pipelines
- Explicit feature names, column order, and schema policies
- Pandas input support through behind-the-scenes conversion to Polars
- Open source under the MIT license

<div class="docs-actions" markdown="1">

[Getting started](getting-started.md)
[API reference](api.md)

</div>
</div>

Pandas DataFrames are supported when you install `polars-learn[pandas]` and enable `convert_pandas=True` on each transformer receiving pandas input. Conversion happens behind the scenes; transformations use the Polars API and return Polars frames by default. Pandas output is available through `set_output(transform="pandas")` for compatibility. See [pandas interoperability](user-guide/frames.md#optional-pandas-input) for details.

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

`fit` applies the declared preparation and fits the scaler and regressor. `predict` validates the raw input, recreates `notional`, selects the fitted model layout, and applies the fitted scaler before predicting. Save the **whole fitted pipeline** to carry its feature definitions and fitted components into deployment. See [saving and loading](user-guide/persistence.md) and [cross-validation](examples/cross-validation.md).

The benefit is a shared, inspectable preparation path. Schema checks detect structural mismatches; they do not prove that values have the right business meaning or automatically prevent leakage. Polars expressions do not learn and freeze statistics. Keep learned preprocessing such as scaling, imputation, or training-derived aggregates in appropriate fitted components, and validate the pipeline before deployment.

### How each class addresses a concrete risk

| Class | Example issue in training or deployment | How the component helps |
| --- | --- | --- |
| [`FeatureSpec`](classes/feature-spec.md) | A transaction-value expression, output name, and column drops are scattered across separate scripts. | Define `price * quantity`, the `notional` alias, and optional drops together as one reusable specification. Construction validates the definition; `FeatureEngineer` executes it. |
| [`FeatureEngineer`](classes/feature-engineer.md) | Training computes `notional` and then `log_notional`, but serving forgets a stage or applies them in a different order. | Store both specs in entry order inside the pipeline. Fit records the feature schema; prediction replays the stages and checks output names, order, and dtypes. |
| [`SchemaSpec`](classes/schema-spec.md) | An application requires `price: Float64` and `quantity: Int64`, but an upstream producer changes `quantity` to strings. | Declare that fixed contract before feature calculation. Exact dtype checks reject the changed input instead of implicitly accepting a new representation; no casting occurs. |
| [`SchemaValidator`](classes/schema-validator.md) | The schema is determined by the training dataset, and a deployment batch arrives with missing columns or changed dtypes. | Learn the required names, order, and dtypes during fit. Transform rejects missing columns and handles dtype changes according to the chosen strict, semi-strict, or permissive policy. |
| [`ColumnSelector`](classes/column-selector.md) | A helper column reaches the estimator, or callers supply model features in a different order. | Select `['price', 'notional']` in that exact order, omit other columns, and optionally rename selected outputs. Fitted output names are checked; add schema validation for dtype checks. |
| [`FeaturePassthrough`](classes/feature-passthrough.md) | A preprocessing branch needs to retain numeric flags unchanged alongside separately scaled numeric features. | Explicitly select the unchanged flags in a `ColumnTransformer` branch and combine them with the scaled branch. Native Polars values and dtypes are retained; downstream model compatibility still needs checking. |

Use the classes together according to your contract. Choose `SchemaSpec` for a known schema or `SchemaValidator` for a learned one; choose `ColumnSelector` when renaming is needed or `FeaturePassthrough` for unchanged selected features. The [parallel branch example](user-guide/selection.md) shows how to combine transformed and unchanged columns.

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

Choose an exact declared contract with `SchemaSpec`, or learn the training contract with `SchemaValidator`. Configure acceptable casts, order changes, and extras explicitly. See the [schema-drift examples](user-guide/schemas.md#schema-drift-at-prediction-time) for rejected batches, safe normalization, and validation inside a prediction pipeline. Schema checks address structure and dtypes; business meaning and distribution drift need separate checks.

## Explore the documentation

<div class="docs-topics" markdown="1">
<div class="docs-topic" markdown="1">

### Feature engineering

Build model features from ordered Polars expressions.

**Components:** [FeatureEngineer](classes/feature-engineer.md), [FeatureSpec](classes/feature-spec.md).

[Feature engineering guide](user-guide/features.md)

</div>
<div class="docs-topic" markdown="1">

### Schema validation

Declare input contracts or learn a schema during training.

**Components:** [SchemaSpec](classes/schema-spec.md), [SchemaValidator](classes/schema-validator.md).

[Schema contracts guide](user-guide/schemas.md)

</div>
<div class="docs-topic" markdown="1">

### Column selection

Select, order, rename, and retain model inputs.

**Components:** [ColumnSelector](classes/column-selector.md), [FeaturePassthrough](classes/feature-passthrough.md).

[Selection and passthrough guide](user-guide/selection.md)

</div>
<div class="docs-topic" markdown="1">

### Pipelines and persistence

Compose estimators, validate models, and save fitted pipelines.

**Workflows:** [Cross-validation](examples/cross-validation.md), [lazy prediction](examples/lazy-pipeline.md).

[Production guide](production.md)

</div>
</div>

## Why use it?

**Turn the power of the Polars API into part of your model.** `polars-learn` lets you express feature preparation with Polars, compose it with scikit-learn-style estimators, and carry that preparation through training, validation, saving, and prediction. Your model artifact can contain both the feature definitions and the estimator that consumes them.

### Rich feature logic through the Polars API

The transformation language is Polars itself. Combine column arithmetic, conditional expressions, string and datetime operations, casts, null handling, and selectors to describe the features your model needs. For example, calculate transaction value from price and quantity, extract calendar features from a timestamp, or build a flag from several input conditions.

A `FeatureSpec` accepts native Polars expressions, including expressions that expand to multiple columns. This makes one reusable pipeline component capable of many different transformations: you supply the expressions rather than writing a new transformer class for each feature. Polars provides the expression engine; `polars-learn` supplies the estimator interface, ordered execution, and schema checks. See [defining features](user-guide/features.md#defining-features).

### Build dependent features in a clear sequence

Real preparation often has several stages. Compute `notional` from `price * quantity`, derive `log_notional` from the new column, then remove raw inputs that the model should not consume. Specifications are applied in exactly their entry order in the sequence passed to `FeatureEngineer`. Each stage adds its expressions and applies its drops before the next stage receives the result, so later stages can use earlier outputs that have been retained. Lazy input builds the same ordered dependencies into a deferred query plan.

Keep related independent expressions together, give outputs explicit names, and place drops after the stage that needs those columns. The feature definitions describe both the calculations and the final model input layout. The [sibling feature example](user-guide/features.md#sibling-features-and-dependent-stages) shows independent expressions grouped in one stage, their outputs combined in later stages, and the resulting feature engineer connected to a model. See [the complete prediction pipeline](getting-started.md#your-first-raw-data-prediction-pipeline).

### Train and predict with the same preparation

Put the feature engineer before the estimator in a `Pipeline`. Call `pipeline.fit(raw_train, y)` to train and `pipeline.predict(raw_predict)` to prepare new observations and predict. Callers can supply raw columns while the pipeline computes the required model features in the configured order.

This is especially useful when a model moves from a notebook into an application: its preparation travels with it, and prediction code uses the same definitions as training. The components follow scikit-learn cloning and parameter conventions, so they can participate in cross-validation and model selection. See [estimators and pipelines](user-guide/estimators.md) and [cross-validation](examples/cross-validation.md).

### Make the model's input contract explicit

Feature values are only part of the contract. Names, order, and dtypes also matter. Declare an expected schema with `SchemaSpec`, learn the training schema with `SchemaValidator`, and use selectors to establish the columns and order passed to the estimator.

`FeatureEngineer` records the resulting feature schema during fitting and checks it during transformation. These checks make missing columns and structural changes visible at the preparation boundary. Choose the schema and casting policies that match your application, then inspect fitted schemas and feature names when reviewing a model. See [schema contracts](user-guide/schemas.md).

### Keep lazy preparation under your control

Use an eager Polars DataFrame for in-memory workflows or a LazyFrame to construct a deferred preparation plan. With lazy input, structural checks use schema metadata and row evaluation can remain deferred until you collect the result or request eager output at the model boundary.

The package preserves access to Polars' lazy execution model while giving preparation an estimator interface. You decide where data becomes materialized according to the receiving model's requirements. See [lazy execution](user-guide/frames.md).

### Bring pandas inputs into the same workflow

You can start with pandas DataFrames and still define your feature logic with the Polars API. Install `polars-learn[pandas]` and enable `convert_pandas=True` on transformers receiving pandas input. The package converts those frames to Polars behind the scenes, then applies the same preparation logic used for native Polars input.

Use `set_output` to choose eager or lazy Polars output, or pandas output for compatibility. Prefer native Polars for speed and to avoid conversion overhead. This provides a path for existing pandas callers to use Polars-based preparation. See [pandas input support](user-guide/frames.md#optional-pandas-input).

### Save preparation alongside the fitted model

Save the fitted pipeline with its feature definitions, schema expectations, selected columns, and estimator. Load that artifact to reuse the preparation when predicting from new raw observations. Fitted attributes, feature names, and configurable diagnostics make the preparation inspectable as well as reusable. See [saving and loading](user-guide/persistence.md) and [production guidance](production.md).

Expressions replay the logic you define. For preprocessing that must learn and retain training statistics, such as scaling or statistical imputation, add an appropriate fitted transformer to the pipeline. See [deterministic expressions versus learned preprocessing](user-guide/features.md#deterministic-expressions-versus-learned-preprocessing).

## Find the right component

| Component | Purpose | Typical production use |
| --- | --- | --- |
| [`FeatureEngineer`](classes/feature-engineer.md) | Execute ordered feature definitions and validate their resulting schema. | Derive the same model features during training and prediction. |
| [`FeatureSpec`](classes/feature-spec.md) | Describe one expression or a group of expressions, aliases, and drops. | Make feature logic explicit and easy to review. |
| [`SchemaSpec`](classes/schema-spec.md) | Enforce a schema you declare. | Validate a known input contract before feature calculation. |
| [`SchemaValidator`](classes/schema-validator.md) | Learn the training schema and enforce a chosen cast policy. | Detect structural or dtype drift at prediction time. |
| [`ColumnSelector`](classes/column-selector.md) | Select, order, and optionally rename columns. | Establish the exact model input names and order. |
| [`FeaturePassthrough`](classes/feature-passthrough.md) | Select named columns without changing their values or dtypes. | Explicitly retain features that need no transformation. |

Each class page includes its purpose, when to use it, a runnable example with input/output tables, production considerations, and its scikit-learn-style reference with typed Parameters, Attributes, and individually documented Methods. The [complete API reference](api.md) collects the shared estimator contract and all exceptions in one place.

## Read by task

New to the package? Start with [installation and a complete prediction pipeline](getting-started.md). For implementation details, the [API reference](api.md) covers every constructor, method, fitted attribute, exception, and warning. The [user guide](user-guide/estimators.md) explains estimator composition, feature dependencies, schema policies, lazy execution, and pandas interoperability. The [examples](examples/index.md) show complete workflows, and the [production guide](production.md) explains where validation happens and how to manage saved models.

## Scope

These classes prepare tabular data; the final estimator learns the predictive model. Feature expressions do not automatically learn training statistics, impute missing values, encode categories, or prevent leakage. Add appropriate fitted preprocessing steps for those tasks. Model compatibility depends on the receiving estimator's supported containers and dtypes; the shared fit/transform/predict interface does not make every model accept a LazyFrame.

Python 3.11 or newer is required. See [installation](getting-started.md#installation) for dependencies and [development](development.md) for the documentation build and Pages setup.

The package is distributed under the [MIT license](https://github.com/JackGreenaway/polars-learn/blob/main/LICENSE).

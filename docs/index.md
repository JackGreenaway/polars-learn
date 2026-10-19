<div class="docs-hero" markdown="1">

# polars-learn

<p class="lead">Train on raw data. Deploy the whole preparation pipeline.</p>

**Your model needs its feature logic as well as its weights.** A model trained on derived features cannot safely consume raw observations unless the same preparation runs first. `polars-learn` puts feature generation, schema validation, and column selection inside a scikit-learn-style pipeline, so `fit` and `predict` share one preparation path.

The power comes from the **Polars API**: compose expressions, use selectors, build dependent features in entry order, and choose eager or lazy execution. The package gives those transformations a reusable estimator interface and fitted schema checks.

- **Keep training and serving together:** save the fitted pipeline with its feature definitions and model, then predict from raw observations.
- **Catch invalid input before prediction:** check required columns and dtypes, enforce row-level value rules, control permitted casts, and make the model's feature layout explicit.
- **Express your actual feature logic:** use native Polars operations or package custom Python callbacks when needed.

<div class="docs-actions" markdown="1">

[Getting started](getting-started.md)
[API reference](api.md)

</div>
</div>

Pandas DataFrames are supported when you install `polars-learn[pandas]` and enable `convert_pandas=True` on each transformer receiving pandas input. Conversion happens behind the scenes; transformations use the Polars API and return Polars frames by default. Pandas output is available through `set_output(transform="pandas")` for compatibility. See [pandas interoperability](user-guide/frames.md#optional-pandas-input) for details.

## The problems it helps solve

| A common failure point | What you put in the pipeline | What changes for training and deployment |
| --- | --- | --- |
| Serving forgets the preparation function or uses a different version. | Ordered `FeatureSpec` definitions executed by `FeatureEngineer`. | The saved pipeline carries the preparation; `predict` accepts raw observations and applies those definitions. |
| A producer changes column names, order, or dtypes. | A declared or learned schema contract. | Structural mismatches fail at validation, or are normalized only through the policies you choose. |
| A helper column accidentally reaches the model. | Explicit selection and output naming. | The estimator receives the intended columns in the intended order. |
| Domain logic needs a Python function alongside Polars expressions. | A `map_elements` or `map_batches` expression with an explicit return dtype. | Serializable callback logic can travel with the fitted artifact and run when preparation is replayed. |
| Evaluation fits learned preparation outside each training fold. | The complete preparation and estimator pipeline. | Cross-validation can clone and fit included learned components within each fold. |

See the runnable [schema-drift examples](user-guide/schemas.md#schema-drift-at-prediction-time), [custom-function and pickle example](user-guide/features.md#custom-python-functions-with-map_elements), and [cross-validation workflow](examples/cross-validation.md). Prefer native Polars expressions for speed; Python callbacks are available for custom logic and must have serializable state and deployment dependencies available.

Use this package when you want Polars-based preparation to be part of the model lifecycle. Expressions replay your logic; learned statistics still belong in fitted preprocessing steps. Schema checks validate structure and dtypes, while business rules and distribution monitoring need separate checks.

See [the complete pipeline workflow](user-guide/pipeline-workflow.md) for training and serving examples.

<figure class="pipeline-figure" aria-labelledby="pipeline-flow-caption">
  <div class="pipeline-flow" role="img" aria-label="Raw Polars input passes through schema validation, feature engineering, and column selection before the fitted estimator predicts.">
    <div class="pipeline-step"><span>Input</span><code>raw Polars frame</code></div>
    <span class="pipeline-arrow" aria-hidden="true">→</span>
    <div class="pipeline-step"><span>Validate</span><code>SchemaValidator</code></div>
    <span class="pipeline-arrow" aria-hidden="true">→</span>
    <div class="pipeline-step"><span>Prepare</span><code>FeatureEngineer</code></div>
    <span class="pipeline-arrow" aria-hidden="true">→</span>
    <div class="pipeline-step"><span>Select</span><code>ColumnSelector</code></div>
    <span class="pipeline-arrow" aria-hidden="true">→</span>
    <div class="pipeline-step"><span>Predict</span><code>fitted estimator</code></div>
  </div>
  <figcaption id="pipeline-flow-caption">At prediction time, raw input passes through the fitted preparation steps before reaching the estimator. Training fits these steps on training data as part of the same pipeline.</figcaption>
</figure>

## Explore the documentation

<div class="docs-topics" markdown="1">
<div class="docs-topic" markdown="1">

### Feature engineering

Build model features from ordered Polars expressions.

**Components:** [FeatureEngineer](classes/feature-engineer.md), [FeatureSpec](classes/feature-spec.md).

[Feature engineering guide](user-guide/features.md)

</div>
<div class="docs-topic" markdown="1">

### Input validation

Declare structural contracts, learn a training schema, and check row values.

**Components:** [SchemaSpec](classes/schema-spec.md), [SchemaValidator](classes/schema-validator.md), [ValueValidator](classes/value-validator.md).

[Schema contracts guide](user-guide/schemas.md) · [Value rules guide](user-guide/value-validation.md)

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

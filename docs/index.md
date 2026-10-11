<div class="docs-hero" markdown="1">

# polars-learn

<p class="lead">Feature engineering in Polars. Prediction pipelines in scikit-learn.</p>

Prepare tabular data with native Polars expressions and explicit schema contracts. Fit preparation with your model and reuse the same steps for every prediction.

- Native Polars expressions for feature engineering
- Compatible with scikit-learn estimators and pipelines
- Explicit feature names, column order, and schema policies
- Open source under the MIT license

<div class="docs-actions" markdown="1">

[Getting started](getting-started.md)
[API reference](api.md)

</div>
</div>

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

- **One preparation path:** fit on raw training observations and predict from new raw observations using the same ordered feature definitions.
- **Native expressions:** use Polars arithmetic, conditionals, selectors, and aliases instead of writing another transformer for every deterministic feature.
- **Explicit contracts:** choose a declared schema or learn a training schema, then decide how prediction-time changes are handled.
- **Named outputs:** preserve feature names and control eager or lazy containers at the model boundary.
- **Reviewable artifacts:** inspect fitted schemas, configure diagnostics, and save the fitted pipeline with version-aware persistence.

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

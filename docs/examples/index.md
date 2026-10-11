# Examples

These examples are small enough to run locally and use real package interfaces. They illustrate composition and validation rather than benchmark performance or model accuracy.

| Workflow | What it demonstrates |
| --- | --- |
| [Raw input to prediction](../getting-started.md#your-first-raw-data-prediction-pipeline) | Fit preprocessing and a model together; predict without supplying derived columns. |
| [Sibling features and ordered dependencies](../user-guide/features.md#sibling-features-and-dependent-stages) | Build features in successive specifications and drop temporary columns. |
| [Parallel transformed and unchanged features](../user-guide/selection.md) | Combine a scaler and FeaturePassthrough in a ColumnTransformer. |
| [Lazy feature preparation](lazy-pipeline.md) | Defer feature evaluation and collect explicitly at the model boundary. |
| [Custom Python feature functions](../user-guide/features.md#custom-python-functions-with-map_elements) | Define map_elements callbacks, compose stages, and pickle/restore fitted preparation. |
| [Schema drift](../user-guide/schemas.md#schema-drift-at-prediction-time) | Reject missing columns and dtype changes, normalize explicitly allowed changes, and guard prediction in a pipeline. |
| [Cross-validation](cross-validation.md) | Clone and fit the full pipeline per fold. |
| [Persistence and warning filters](../user-guide/persistence.md) | Save a fitted pipeline and choose whether mismatches warn, fail, or are silenced. |

## Prediction and evaluation gallery

| Question | Workflow | Data |
| --- | --- | --- |
| [Can order value predict a fee?](held-out-orders.md) | Held-out split, schema validation, feature selection, scaling, regression | Generated locally, 80 orders |
| [How stable is a full pipeline across folds?](cross-validation.md) | Clone and fit the entire preparation path per fold | Local order fixture |
| [Where should a lazy query become eager?](lazy-pipeline.md) | Deferred Polars preparation and estimator boundary | Local order fixture |

Copy each complete Python block into a file and run it in the installed environment.
Assertions and marked displayed output are validated by the documentation checker.

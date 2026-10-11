# Examples

These examples are small enough to run locally and use real package interfaces. They illustrate composition and validation rather than benchmark performance or model accuracy.

| Workflow | What it demonstrates |
| --- | --- |
| [Raw input to prediction](../getting-started.md#your-first-raw-data-prediction-pipeline) | Fit preprocessing and a model together; predict without supplying derived columns. |
| [Sibling features and ordered dependencies](../user-guide/features.md#sibling-features-and-dependent-stages) | Build features in successive specifications and drop temporary columns. |
| [Parallel transformed and unchanged features](../user-guide/selection.md) | Combine a scaler and FeaturePassthrough in a ColumnTransformer. |
| [Lazy feature preparation](lazy-pipeline.md) | Defer feature evaluation and collect explicitly at the model boundary. |
| [Cross-validation](cross-validation.md) | Clone and fit the full pipeline per fold. |
| [Persistence and warning filters](../user-guide/persistence.md) | Save a fitted pipeline and choose whether mismatches warn, fail, or are silenced. |

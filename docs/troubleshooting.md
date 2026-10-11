# Troubleshooting

| Symptom | Cause and remedy |
| --- | --- |
| `NotFittedError` | Fit the transformer or full pipeline on training data before calling transform/predict. |
| Missing columns at prediction | Supply the fitted raw input contract, including columns needed by expressions. Do not send only precomputed features. |
| Feature output schema changed | Check output names, order, dtypes, aliases, and input types. Use schema validation before engineering. |
| A feature cannot see a newly created column | Sibling expressions execute together. Put the dependent expression in a later FeatureSpec. |
| `expressions` is rejected | The single required argument is `expression`; it accepts one Expr or a nonempty sequence. There is no compatibility alias. |
| Passthrough removed other features | A sequential passthrough selects only its columns. Use parallel ColumnTransformer branches to combine transformed and unchanged outputs. |
| Model rejects a LazyFrame | Request `set_output(transform="polars")` at the last native step, then confirm the estimator accepts its eager dtypes. |
| Cast failure appears during collect | Lazy plans defer value checks until execution. Handle errors around collect as well as transform. |
| Small integers still cannot narrow in semi-strict mode | Integer widening is based on the entire source dtype range, not only observed values. |
| Version warnings during artifact loading | Compare the recorded creating/current versions. Validate compatibility, then use an error or ignore warning filter as appropriate. |
| Warnings still print at verbosity 0 | Verbosity controls progress only. Use warning filters or `warn_on_conversion=False` for conversion warnings. |
| Captured callback will not pickle | Remove captured locks/open resources or use native Polars expressions; keep callback dependencies available. |
| Pandas input is rejected | Install the pandas extra and set `convert_pandas=True`; pandas output is unsupported. |
| A parameter change did not rebuild fitted state | Refit after set_params; build a fresh instance if previous fitted state must be discarded. |

See the [API exceptions](api.md#exceptions-and-warnings) for exact failure categories and the [production guide](production.md) for validation practices.

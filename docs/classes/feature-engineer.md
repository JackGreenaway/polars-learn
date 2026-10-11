# FeatureEngineer

## About

Feature preparation becomes part of the model pipeline. Define derived columns once, fit the raw-input and output-schema contract, and let prediction replay the same definitions before the estimator receives its features.

## Why use it?

Use it when training and serving must share ordered, deterministic feature logic. It is especially useful for dependent features: calculate an intermediate column in one specification, consume it in the next, and drop it before the model boundary.

## Example: input to output

### Input

| price | quantity |
| --- | --- |
| 10.0 | 2 |
| 20.0 | 3 |

### Code

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

raw = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
engineer = FeatureEngineer(
    [
        FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
        FeatureSpec(
            pl.col("notional").log1p(), alias="log_notional", drop=["quantity"]
        ),
    ]
)
prepared = engineer.fit_transform(raw)
assert prepared.columns == ["price", "notional", "log_notional"]
assert prepared["notional"].to_list() == [20.0, 60.0]
assert prepared["log_notional"].equals(pl.Series("log_notional", [20.0, 60.0]).log1p())
assert raw.columns == ["price", "quantity"]
```

### Output

| price | notional | log_notional |
| --- | --- | --- |
| 10.0 | 20.0 | 3.044522437723423 |
| 20.0 | 60.0 | 4.110873864173311 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Production considerations

Put the engineer before the predictive estimator and pass raw data to both pipeline.fit and pipeline.predict. It records schemas rather than learned feature statistics; use fitted statistical preprocessing for training-derived means, scales, or vocabularies. Eager fit_transform evaluates expressions twice. See the [complete prediction pipeline](../getting-started.md#your-first-raw-data-prediction-pipeline).

## Reference

Use `FeatureEngineer` as a preprocessing step before a predictive estimator. Define feature expressions once, fit the full pipeline on raw training data, and pass new raw observations to `pipeline.predict`. The pipeline calls the engineer's `transform` before the estimator's prediction method, generating the same named features in the same specification order. This keeps preparation inside the pipeline instead of requiring a separate feature-building function at each call site. See the [raw-data prediction example](../getting-started.md#your-first-raw-data-prediction-pipeline).

The fitted raw input names/order and resulting feature schema remain the contract. Prediction callers provide required raw columns, not precomputed derived columns. `FeatureEngineer` replays expressions and validates schemas; learned preprocessing statistics must come from other fitted pipeline steps.

```text
FeatureEngineer(
    features,
    *,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

`features` accepts a single `FeatureSpec`, a dict of `FeatureSpec` keyword
arguments, or an ordered sequence mixing these two forms. Single definitions are
normalized to one-element tuples for execution. Dicts are unpacked with
`FeatureSpec(**config)`. Generators and bare Polars expressions are unsupported.
An empty sequence is valid and produces an identity transformation. All
specifications are checked before processing input. The original parameter
object is retained unchanged for estimator cloning and `get_params`.
`set_params(features=...)` accepts the same forms; refit after changes.

Config keys are `expression`, `alias`, `drop`, `name`, and
`verbosity`, with the same types and validation as the FeatureSpec constructor.
The required `expression` accepts a single expression or a nonempty sequence.
Missing arguments, unknown keys, and invalid expression types fail with
`TypeError`; an empty expression sequence fails with `ValueError`. Config errors identify `features` or `features[index]` and preserve
the original exception as their cause. Real Polars expressions are required;
strings are not parsed or evaluated.

Configs are validated at construction and converted again once per fit or
transform call. This picks up parameter changes without mutating caller-owned
dicts. A config's own verbosity can emit specification construction summaries
on those conversions, independently of the engineer's verbosity. For optional
static annotations, `FeatureSpecConfig` is available from `polars_learn.types`;
it is a TypedDict describing an ordinary dict, with validation applied by the
engineer at runtime.

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

X = pl.DataFrame({"x": [1, 2]})
spec = FeatureSpec(pl.col("x") * 2, alias="doubled")
engineer = FeatureEngineer(spec)
assert engineer.get_params()["features"] is spec
assert engineer.fit_transform(X)["doubled"].to_list() == [2, 4]
```

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec
from polars_learn.types import FeatureSpecConfig

X = pl.DataFrame({"x": [1, 2]})
config: FeatureSpecConfig = {"expression": pl.col("x") * 2, "alias": "doubled"}
engineer = FeatureEngineer(
    [
        config,
        FeatureSpec(pl.col("doubled") + 1, alias="next", drop=["doubled"]),
    ]
)
assert engineer.fit_transform(X.lazy()).collect()["next"].to_list() == [3, 5]
```

Additional fitted attributes are `input_schema_` and `output_schema_`, dicts
containing input and resulting output names/dtypes in order. `output_features_`
lists all resulting columns, including retained input columns and new features.

Fit builds the ordered plan, checking each specification's expression
schema and drops. With eager input it executes the plan before recording state;
with lazy input it records schemas without evaluating rows. Transform requires
exact fitted input names and order, and an exact resulting output schema. Input
dtypes can differ only if that final schema still matches.

`fit_transform` currently calls `fit` then `transform`; eager feature expressions
are therefore evaluated twice, and pandas conversion can happen twice. Lazy
input remains unevaluated during those calls unless eager output is requested.
Feature expressions do not learn training statistics. Avoid nondeterministic
expressions when consistent training and inference are required.

Plan dependency, drop, and output-schema failures raise `FeatureEngineeringError`.
Eager execution failures are wrapped in that exception with their causes.
Execution errors from a returned lazy plan can be native Polars exceptions;
an explicit `"polars"` output request wraps them within `transform`.

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

X = pl.DataFrame({"x": [1, 2]})
engineer = FeatureEngineer(
    [
        FeatureSpec(pl.col("x") + 1, alias="a", name="increment"),
        FeatureSpec(pl.col("a") * 2, alias="b", drop=["a"], name="double"),
    ]
)
plan = engineer.fit_transform(X.lazy())
assert plan.collect().to_dict(as_series=False) == {"x": [1, 2], "b": [4, 6]}
```

## Shared methods and configuration

All transformers share fit, transform, fit_transform, get_feature_names_out, get_params, set_params, and set_output. Their common constructor options are convert_pandas, warn_on_conversion, and verbosity. Consult the [shared reference](../api.md#shared-transformer-behavior) for signatures, return types, fitted attributes, cloning, and failure behavior.

See [persistence](../user-guide/persistence.md) for serialization and warning controls, and [diagnostics](../user-guide/diagnostics.md) for verbosity and error timing.

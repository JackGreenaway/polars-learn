# Using polars-learn in production

## Why preparation belongs with the model

A prediction endpoint should accept the raw observations the fitted pipeline expects. When feature logic lives in a separate helper, training and serving callers must remember to invoke the same helper in the same order. `FeatureEngineer` makes those expressions part of the pipeline: training fits the preparation contract and prediction reuses it before the model executes. Keeping schema checks, feature construction, and selection together makes changes reviewable and model artifacts easier to operate.

This is a practical building block for production workflows, not a guarantee of application correctness or a performance benchmark. Your data contracts, model tests, monitoring, and deployment process remain part of the system.

## Establish a clear input contract

Use `SchemaSpec` for an externally declared contract or `SchemaValidator` for a contract learned from training. Decide how extras, order changes, and dtype changes should behave. Keep target columns outside the feature frame. Select the exact model-facing features after engineering if raw columns should not reach the estimator. Inspect fitted input/output names and schemas as part of model review.

A schema validates names and dtypes, not business meaning. Add `ValueValidator` after the structural contract to enforce required non-null values, numeric bounds, allowed categories, and other row-level rules with named Polars Boolean expressions. False and null check results raise `ValueValidationError`. A permissive cast may lose information even when it succeeds; use strict or semi-strict policies where that tradeoff is unacceptable.

For a training-defined contract, strict mode rejects an unexpected dtype with a
typed package error. `ValueValidator` then checks row-level rules before the
model sees the batch:

```python
import polars as pl

from polars_learn import (
    SchemaValidationError,
    SchemaValidator,
    ValueValidationError,
    ValueValidator,
)

training = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
contract = SchemaValidator(mode="strict").fit(training)
values = ValueValidator(
    {
        "positive_price": pl.col("price") > 0,
        "quantity_present": pl.col("quantity").is_not_null(),
    }
).fit(training)
incoming = pl.DataFrame({"price": [15.0], "quantity": ["2"]})

try:
    contract.transform(incoming)
except SchemaValidationError as error:
    assert "quantity" in str(error)
else:
    raise AssertionError("Unexpected input dtype was accepted")

wrong_value = pl.DataFrame({"price": [-1.0, 15.0], "quantity": [2, 1]})
try:
    values.transform(wrong_value)
except ValueValidationError as error:
    assert "positive_price" in str(error)
else:
    raise AssertionError("Invalid row value was accepted")
```

Use application checks for rules that need external state, cross-row aggregation,
or custom remediation. `ValueValidator` covers row-wise conditions that can be
expressed as Polars Boolean expressions.

## Keep prediction behavior stable

Prefer deterministic native expressions. Put features with dependencies in successive specifications. Use fitted preprocessing for statistics learned on training data; recomputing batch statistics during prediction can change results with request composition. Fit a complete pipeline on training data and reuse it; do not refit transformers on prediction input. Refit after changing constructor parameters or feature definitions.

## Make failures observable

Configuration mistakes fail during construction or revalidation. Fit resolves schemas without evaluating input rows. Eager execution and value checks happen during transform; lazy plans may defer value-dependent errors until collection. Handle these stages, retain chained exceptions, and record useful metadata without dumping raw rows. Verbosity 1 provides progress; verbosity 2 adds schema and operation details. See [diagnostics](user-guide/diagnostics.md) for exact exception types and timing.

Failed refitting may preserve previous successful state. Build and validate a fresh candidate pipeline, then swap it into service after its checks pass, rather than modifying a serving object's configuration in place.

## Control the data and model boundary

Use lazy inputs to defer row execution and explicit eager output where your final estimator needs it. The package does not automatically convert model input to NumPy, encode categorical values, or make every estimator accept Polars. Validate the real model/container/dtype combination. `FeatureEngineer.fit_transform` resolves the feature schema and executes eager feature expressions once during transform. Schema resolution can access source metadata; measure your own workload before making latency or scan-count assumptions.

## Save a complete, verifiable artifact

Save the fitted pipeline together with a pinned environment and representative raw-data fixtures. Test that loading preserves transformed names, order, dtypes, values, and predictions. Version mismatches warn before class state restoration; accepted compatible loads are attempted, and incompatible state raises a contextual error. You can silence known warnings with standard filters or promote them to errors. A successful cross-version load does not establish equivalent semantics, so validate outputs before rollout and keep the original environment available.

Only load trusted artifacts. Python callbacks require serializable captured state and available modules or resources. Prefer native expressions when possible. See [persistence](user-guide/persistence.md) for save/load examples and format limits.

## Upgrade and deployment checklist

1. Fit a fresh candidate pipeline on the intended training data and inspect its schemas and selected model features.
2. Exercise accepted and rejected inputs: missing columns, extras, reordered columns, incompatible dtypes, nulls, invalid values, and boundary values. Assert the expected exception type at the stage where eager or lazy execution occurs.
3. Round-trip the complete artifact in the receiving environment and compare feature names, order, dtypes, values, and predictions.
4. Measure latency and memory with representative batch sizes and sources. Include query collection and pandas conversion if your service uses them; lazy plans defer work but do not guarantee low memory or latency.
5. Record dependency versions with the release, deploy the validated candidate, and monitor input failures, prediction behavior, and resource use.

The repository's tests exercise fitted/unfitted serialization, callback closures, eager/lazy behavior, scikit-learn composition, and real cross-version Polars loading. These checks support the package contract; application-specific validation is still needed.

# Using polars-learn in production

## Why preparation belongs with the model

A prediction endpoint should accept the raw observations the fitted pipeline expects. When feature logic lives in a separate helper, training and serving callers must remember to invoke the same helper in the same order. `FeatureEngineer` makes those expressions part of the pipeline: training fits the preparation contract and prediction reuses it before the model executes. Keeping schema checks, feature construction, and selection together makes changes reviewable and model artifacts easier to operate.

This is a practical building block for production workflows, not a guarantee of application correctness or a performance benchmark. Your data contracts, model tests, monitoring, and deployment process remain part of the system.

## Establish a clear input contract

Use `SchemaSpec` for an externally declared contract or `SchemaValidator` for a contract learned from training. Decide how extras, order changes, and dtype changes should behave. Keep target columns outside the feature frame. Select the exact model-facing features after engineering if raw columns should not reach the estimator. Inspect fitted input/output names and schemas as part of model review.

A schema validates names and dtypes, not business meaning. Check required non-null values, value ranges, category availability, and other domain rules separately. A permissive cast may lose information even when it succeeds; use strict or semi-strict policies where that tradeoff is unacceptable.

## Keep prediction behavior stable

Prefer deterministic native expressions. Put features with dependencies in successive specifications. Use fitted preprocessing for statistics learned on training data; recomputing batch statistics during prediction can change results with request composition. Fit a complete pipeline on training data and reuse it; do not refit transformers on prediction input. Refit after changing constructor parameters or feature definitions.

## Make failures observable

Configuration mistakes fail during construction or revalidation. Eager schema and execution checks happen during fit/transform. Lazy value-dependent errors happen when the returned query executes. Handle both stages, retain chained exceptions, and record useful metadata without dumping raw rows. Verbosity 1 provides progress; verbosity 2 adds schema and operation details. See [diagnostics](user-guide/diagnostics.md) for exact exception types and timing.

Failed refitting may preserve previous successful state. Build and validate a fresh candidate pipeline, then swap it into service after its checks pass, rather than modifying a serving object's configuration in place.

## Control the data and model boundary

Use lazy inputs to defer row execution and explicit eager output where your final estimator needs it. The package does not automatically convert model input to NumPy, encode categorical values, or make every estimator accept Polars. Validate the real model/container/dtype combination. `fit_transform` can evaluate eager features twice, and schema resolution can access source metadata; measure your own workload before making latency or scan-count assumptions.

## Save a complete, verifiable artifact

Save the fitted pipeline together with a pinned environment and representative raw-data fixtures. Test that loading preserves transformed names, order, dtypes, values, and predictions. Version mismatches warn before class state restoration; accepted compatible loads are attempted, and incompatible state raises a contextual error. You can silence known warnings with standard filters or promote them to errors. A successful cross-version load does not establish equivalent semantics, so validate outputs before rollout and keep the original environment available.

Only load trusted artifacts. Python callbacks require serializable captured state and available modules or resources. Prefer native expressions when possible. See [persistence](user-guide/persistence.md) for save/load examples and format limits.

## Upgrade and deployment checklist

1. Fit a fresh candidate pipeline on the intended training data and inspect its schemas and selected model features.
2. Run representative prediction fixtures, including missing columns, wrong dtypes, nulls, and lazy execution if used.
3. Round-trip the complete artifact in the receiving environment and compare features and predictions.
4. Check runtime cost with realistic data and record environment versions with the release.
5. Deploy the validated candidate and monitor input failures, prediction behavior, and resource use.

The repository's tests exercise fitted/unfitted serialization, callback closures, eager/lazy behavior, scikit-learn composition, and real cross-version Polars loading. These checks support the package contract; application-specific validation is still needed.

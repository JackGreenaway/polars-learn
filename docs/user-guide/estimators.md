# Estimator conventions and pipelines

The preprocessing steps use `fit` and `transform`; the final estimator uses `fit(X, y)` and its prediction method. Pass raw data to the complete pipeline for both fitting and prediction. `Pipeline` fits preprocessing on training input and, during `predict`, reapplies the fitted schema policies and feature expressions before invoking the model. The transformers ignore `y`; the model uses it to learn predictions.

A pipeline built from these transformers can remain lazy. Set the last native transformer to eager output before a downstream estimator that needs materialized data. `"polars-lazy"` is a package-specific output option; do not pass it to scikit-learn's standard transformers.

```python
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polars_learn import FeatureEngineer, FeatureSpec, SchemaValidator

X = pl.DataFrame({"price": [10.0, 20.0, 30.0], "quantity": [2, 3, 4]})
y = np.array([1.0, 2.0, 3.0])
pipeline = Pipeline(
    [
        ("schema", SchemaValidator(mode="semi-strict")),
        (
            "features",
            FeatureEngineer(
                [FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")]
            ).set_output(transform="polars"),
        ),
        ("scale", StandardScaler().set_output(transform="polars")),
        ("model", RandomForestRegressor(n_estimators=20, random_state=0)),
    ]
)
pipeline.fit(X.lazy(), y)
predictions = pipeline.predict(X.lazy())
assert predictions.shape == (3,)
```

Supported input is Polars DataFrames and LazyFrames, plus explicitly enabled pandas DataFrames. NumPy arrays, Series, Python lists, and sparse arrays are not accepted. These transformers bypass scikit-learn's automatic output wrapping and global output-container settings; configure each native transformer with `set_output`. There is no automatic conversion to NumPy. Request pandas compatibility output explicitly with `set_output(transform="pandas")`.

Cloning keeps constructor parameters and explicit output settings, and discards fitted state. `y` is accepted for estimator compatibility and ignored. Additional `fit_transform` keyword parameters, including `sample_weight`, are unsupported. Refit after changing parameters that affect the fitted contract; `set_params` does not reset fitted state. Invalid settings are rechecked when processing input.

### The Estimator Pattern

These transformers follow the same estimator conventions used by scikit-learn and libraries such as XGBoost:

| Operation | Preprocessing transformer | Final predictive model |
| --- | --- | --- |
| `fit` | Validate or learn the training input contract; return the fitted transformer. | Learn model parameters from training features and targets. |
| Apply fitted state | `transform(X)` returns prepared features. | `predict(X)` returns predictions. |
| Fit and apply | `fit_transform(X)` fits and returns prepared training features. | Fit first, then call the model's prediction method. |
| Configure | Constructor parameters, `get_params`, and `set_params`. | The same parameter conventions for estimator-style models. |

In a `Pipeline`, each preprocessing step prepares the input for the next step, and the final estimator learns from that prepared input. Calling `predict` on the fitted pipeline reapplies the preprocessing steps before predicting. XGBoost is an example of a library with a [scikit-learn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html); its regression/classification models use `fit` and `predict`. The `transform` operation belongs to the preprocessing steps.

This shared interface describes how steps compose. The final estimator must also accept the prepared feature container and dtypes. Set the last native transformer to `set_output(transform="polars")` when the model needs eager data. If a model requires a different container, add an explicit conversion step at that boundary. Keep preprocessing fitted on training input and reuse it for validation and prediction.

## Fitting, cloning, and changing parameters

All six transformers expose `get_params`, `set_params`, `fit`, `transform`, `fit_transform`, `get_feature_names_out`, and `set_output`. `FeatureSpec` is an operation description rather than an estimator. `fit` returns the transformer, while transformation returns a Polars frame by default or a pandas DataFrame when requested. Calling `transform` before fitting raises `NotFittedError`.

Scikit-learn cloning copies constructor configuration and explicit output settings, then discards learned state. Use complete pipelines in cross-validation so each fold fits its own training-dependent preprocessing. Keep the target out of the feature frame. `set_params` changes configuration without immediately validating it or clearing learned state; refit after changing the contract. A failed refit may leave earlier fitted state available, so use a fresh instance when old state must be discarded.

## Inspecting model inputs

`feature_names_in_` and `n_features_in_` describe the whole fitted input. `output_features_` and `get_feature_names_out()` describe the transformer output, subject to the class-specific extra-column policy. Returned name arrays are copies. `FeatureEngineer` also exposes `input_schema_` and `output_schema_`; schema classes expose `schema_`. See the [complete shared API](../api.md#shared-transformer-behavior).

## Compatible interfaces and model boundaries

Libraries such as XGBoost offer scikit-learn-style estimators, but preprocessing uses `transform` and predictive models use `predict`. Choose the output container and feature dtypes that your specific estimator accepts. These docs demonstrate RandomForestRegressor; they do not promise universal model compatibility.

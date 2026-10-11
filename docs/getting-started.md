# Getting started

## Installation

Install the package into your project:

```bash
uv add polars-learn
# Or use pip:
pip install polars-learn
```

For optional pandas input support:

```bash
uv add 'polars-learn[pandas]'
# Or use pip:
pip install 'polars-learn[pandas]'
```

Python **3.11 or newer** is required. Runtime dependencies are Polars >=1.20, scikit-learn >=1.6, NumPy >=1.24, and cloudpickle >=3. The pandas extra adds pandas >=2 and PyArrow >=14. These are minimum supported dependency declarations, rather than a guarantee of cross-version artifact compatibility; retain a lockfile when saving models.

See [Development](development.md) for running this repository from a checkout.

## Your first raw-data prediction pipeline

Define the preparation once and train the complete pipeline on raw input. This example creates `notional` and `log_notional` inside the pipeline, removes `quantity` from the model's feature set, and retains `price`.

```python
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from polars_learn import FeatureEngineer, FeatureSpec, SchemaValidator

raw_train = pl.DataFrame({"price": [10.0, 20.0, 30.0, 40.0], "quantity": [2, 3, 4, 5]})
y = np.array([1.0, 2.0, 3.0, 4.0])

pipeline = Pipeline(
    [
        ("raw_schema", SchemaValidator()),
        (
            "features",
            FeatureEngineer(
                [
                    FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
                    FeatureSpec(
                        pl.col("notional").log1p(),
                        alias="log_notional",
                        drop=["quantity"],
                    ),
                ]
            ).set_output(transform="polars"),
        ),
        ("model", RandomForestRegressor(n_estimators=20, random_state=0)),
    ]
)

pipeline.fit(raw_train, y)

# Prediction takes new raw observations; feature preparation runs inside the pipeline.
raw_predict = pl.DataFrame({"price": [15.0, 25.0], "quantity": [2, 4]})
predictions = pipeline.predict(raw_predict)
assert predictions.shape == (2,)
assert np.isfinite(predictions).all()

# Inspect the prepared feature names without building features outside the pipeline.
assert pipeline[:-1].get_feature_names_out().tolist() == [
    "price",
    "notional",
    "log_notional",
]
assert raw_predict.columns == ["price", "quantity"]
```

The prediction input contains `price` and `quantity`; you supply neither `notional` nor `log_notional`. `pipeline.predict(raw_predict)` validates the raw schema, computes the derived columns, drops `quantity` from the model input, and then predicts. Keep the target `y` separate from the raw feature frame. Fit the complete pipeline on training input and reuse that fitted pipeline for validation and prediction.

The pipeline expects the fitted raw input contract at prediction time. Do not feed it a frame that has already been feature-engineered, or refit its preprocessing on prediction data. Missing required raw columns and incompatible schemas raise errors according to the configured policies.

The individual transformers also work on their own: `fit(X)` returns the transformer, `transform(X)` applies it, and `fit_transform(X)` fits and prepares the same input. Calling `transform` or `get_feature_names_out` before a successful fit raises scikit-learn's `NotFittedError`. Input frames are never mutated.

## Next steps

Read [feature engineering](user-guide/features.md) to define dependent features, [schema policies](user-guide/schemas.md) to choose an input contract, and [production guidance](production.md) before deploying a fitted pipeline.

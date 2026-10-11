# A lazy pipeline with an eager model boundary

Start with raw lazy observations and make collection explicit before the final estimator. This example uses an in-memory LazyFrame; the same transformer interface accepts a scan query when its source is available.

```python
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from polars_learn import FeatureEngineer, FeatureSpec, SchemaValidator

raw = pl.DataFrame({"price": [10.0, 20.0, 30.0, 40.0], "quantity": [2, 3, 4, 5]})
pipeline = Pipeline(
    [
        ("schema", SchemaValidator()),
        (
            "features",
            FeatureEngineer(
                FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
            ).set_output(transform="polars"),
        ),
        ("model", RandomForestRegressor(n_estimators=10, random_state=0)),
    ]
)
pipeline.fit(raw.lazy(), np.array([1.0, 2.0, 3.0, 4.0]))
predictions = pipeline.predict(raw.head(2).lazy())
assert predictions.shape == (2,)
assert pipeline[:-1].get_feature_names_out().tolist() == [
    "price",
    "quantity",
    "notional",
]
```

The schema step preserves lazy input. The feature step requests eager output, so its returned result is collected before reaching the model. Eager fitting and fit_transform behavior can still evaluate preparation more than once; this is not a promise of a single source scan. Keep source access and execution errors within your request's error handling. See [lazy execution](../user-guide/frames.md).

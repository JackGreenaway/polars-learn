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
incoming = pl.DataFrame({"price": [15.0, 25.0], "quantity": [2, 3]})
predictions = pipeline.predict(incoming.lazy())
assert predictions.shape == (2,)
print(f"Predictions returned: {len(predictions)}")
assert pipeline[:-1].get_feature_names_out().tolist() == [
    "price",
    "quantity",
    "notional",
]
```
<!-- verified-output -->
```text
Predictions returned: 2
```

The schema step preserves lazy input. The feature step requests eager output, so its returned result is collected before reaching the model. Eager fitting and fit_transform behavior can still evaluate preparation more than once; this is not a promise of a single source scan. Keep source access and execution errors within your request's error handling. See [lazy execution](../user-guide/frames.md).


## Interpretation and next steps

The four local training orders have illustrative targets 1 through 4. Two new
orders produce two predictions; this tiny fixture demonstrates the collection
boundary, not predictive accuracy. `set_output(transform="polars")` materializes
the feature step because the forest needs eager values. Use the
[held-out workflow](held-out-orders.md) to evaluate predictions before deploying.
For file scans, keep sources available through collection and monitor memory use.

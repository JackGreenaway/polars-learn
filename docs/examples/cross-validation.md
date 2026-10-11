# Cross-validation with a complete pipeline

Fit the full pipeline inside each fold instead of fitting training-dependent preprocessing on the full dataset before splitting. The expressions here are deterministic; a statistical preprocessing step added later should learn only from its fold's training data.

```python
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from polars_learn import FeatureEngineer, SchemaValidator

raw = pl.DataFrame(
    {"price": [float(i) for i in range(1, 13)], "quantity": [1, 2, 3] * 4}
)
y = np.arange(12, dtype=float)
pipeline = Pipeline(
    [
        ("schema", SchemaValidator()),
        (
            "features",
            FeatureEngineer(
                {
                    "expression": pl.col("price") * pl.col("quantity"),
                    "alias": "notional",
                }
            ).set_output(transform="polars"),
        ),
        ("model", RandomForestRegressor(n_estimators=10, random_state=0)),
    ]
)
scores = cross_val_score(pipeline, raw, y, cv=3, scoring="neg_mean_absolute_error")
assert scores.shape == (3,)
assert np.isfinite(scores).all()
```

Choose a splitter appropriate for your data; time-ordered or grouped observations often need a different split policy. `FeatureEngineer` cannot detect target leakage in an expression or input column. Inspect selected model features and keep the target separate from the raw frame.

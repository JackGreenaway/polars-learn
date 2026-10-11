# Cross-validation with a complete pipeline

Fit the full pipeline inside each fold instead of fitting training-dependent preprocessing on the full dataset before splitting. The expressions here are deterministic; a statistical preprocessing step added later should learn only from its fold's training data.

```python
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from polars_learn import FeatureEngineer, SchemaValidator

# Local fixture: twelve independent orders, with an illustrative fee target.
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
print(f"Mean fold MAE: {-scores.mean():.3f}")
```
<!-- verified-output -->
```text
Mean fold MAE: 2.575
```

Choose a splitter appropriate for your data; time-ordered or grouped observations often need a different split policy. `FeatureEngineer` cannot detect target leakage in an expression or input column. Inspect selected model features and keep the target separate from the raw frame.

## Interpretation and next steps

The target here is an illustrative increasing fee rather than a real business
measurement. With `cv=3`, regression uses three unshuffled folds; each training
fold excludes its validation orders. Negative MAE is scikit-learn's scoring
convention: negate it to report error in target units, where smaller is better.
A forest cannot extrapolate beyond its training target range, which matters for
this ordered fixture. Compare a baseline and choose shuffled, grouped, or
chronological folds according to how future data will arrive. See the
[held-out example](held-out-orders.md) for a known generating relationship.

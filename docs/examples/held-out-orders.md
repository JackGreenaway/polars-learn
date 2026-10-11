# Can transaction value predict a service fee?

## Question and data

Can a pipeline predict a fee from an order's price and quantity while accepting
raw orders at prediction time? We generate 80 independent synthetic orders
locally, with a fee equal to 1.5 plus 2% of transaction value. This deliberately
noiseless relationship demonstrates pipeline wiring, not evidence about real fees.
The target is kept outside the feature frame; no downloads are needed.

## Configuration and complete example

Split raw observations before fitting. The schema validator learns only training
structure. Feature engineering computes a row-local product; explicit selection
removes raw and helper columns before the scaler and regression model. The scaler
learns its mean and variance only from training data within the pipeline.

```python
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from polars_learn import ColumnSelector, FeatureEngineer, FeatureSpec, SchemaValidator

rng = np.random.default_rng(42)
prices = rng.uniform(5, 100, 80)
quantities = rng.integers(1, 10, 80)
raw = pl.DataFrame({"price": prices, "quantity": quantities})
fees = 1.5 + 0.02 * prices * quantities
X_train, X_test, y_train, y_test = train_test_split(
    raw, fees, test_size=0.25, random_state=42
)
pipeline = make_pipeline(
    SchemaValidator(),
    FeatureEngineer(FeatureSpec(pl.col("price") * pl.col("quantity"), alias="value")),
    ColumnSelector(["value"]),
    StandardScaler(),
    LinearRegression(),
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(f"Training orders: {X_train.height}; test orders: {X_test.height}")
print(f"Held-out MAE: {mean_absolute_error(y_test, predictions):.6f}")
assert np.allclose(predictions, y_test)
```
<!-- verified-output -->
```text
Training orders: 60; test orders: 20
Held-out MAE: 0.000000
```

## Results and interpretation

Mean absolute error is the average absolute difference between the predicted
and observed fees, in fee currency units. Zero at the displayed precision is
expected because the model class contains the exact generating relationship.
It does not imply the same accuracy for noisy real orders. Prediction runs the
same schema check, product, selection, and fitted scaling as training.

## Next steps

Add realistic noise and compare against a constant baseline. Use chronological
or grouped splits if orders are related by time or customer. Evaluate the
[complete pipeline in cross-validation](cross-validation.md), then
[save the fitted pipeline](../user-guide/persistence.md) and test schema drift.

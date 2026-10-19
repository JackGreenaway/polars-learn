# polars-learn

Native Polars feature engineering, schema contracts, and row-level value checks inside scikit-learn pipelines.
For Python users who want the same preparation path during training and prediction:
compose ordered expressions, validate declared or learned schemas, and select model
columns. Supports eager and lazy Polars frames and optional pandas conversion.

## Installation

Requires Python 3.11+, Polars 1.20+, scikit-learn 1.6+, NumPy 1.24+, and
cloudpickle 3+. Dependencies are installed automatically.

```bash
pip install polars-learn
# Optional pandas input/output support:
pip install 'polars-learn[pandas]'
```

## Quick start

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

orders = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
features = FeatureEngineer(
    FeatureSpec(pl.col("price") * pl.col("quantity"), alias="total")
)
prepared = features.fit_transform(orders)
assert prepared["total"].to_list() == [20.0, 60.0]
```

Expressions replay on each batch; they do not freeze learned statistics. Schema
checks cover structure and dtypes, not business rules or distribution drift.
Lazy queries must be collected before an estimator that requires eager input.

Read the [documentation](https://jackgreenaway.github.io/polars-learn/),
[installation and first pipeline](docs/getting-started.md),
[user guide](docs/user-guide/index.md), [examples](docs/examples/index.md),
[API reference](docs/api.md), and [development guide](docs/development.md).
Licensed under [MIT](LICENSE).

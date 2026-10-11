# FeatureEngineer workflows

## About

Feature preparation becomes part of the model pipeline. Define derived columns once, fit the raw-input and output-schema contract, and let prediction replay the same definitions before the estimator receives its features.

## Why use it?

Use it when training and serving must share ordered, deterministic feature logic. It is especially useful for dependent features: calculate an intermediate column in one specification, consume it in the next, and drop it before the model boundary.

## Examples

### Input

| price | quantity |
| --- | --- |
| 10.0 | 2 |
| 20.0 | 3 |

### Code

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

raw = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
engineer = FeatureEngineer(
    [
        FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
        FeatureSpec(
            pl.col("notional").log1p(), alias="log_notional", drop=["quantity"]
        ),
    ]
)
prepared = engineer.fit_transform(raw)
assert prepared.columns == ["price", "notional", "log_notional"]
assert prepared["notional"].to_list() == [20.0, 60.0]
assert prepared["log_notional"].equals(pl.Series("log_notional", [20.0, 60.0]).log1p())
assert raw.columns == ["price", "quantity"]
```

### Output

| price | notional | log_notional |
| --- | --- | --- |
| 10.0 | 20.0 | 3.044522437723423 |
| 20.0 | 60.0 | 4.110873864173311 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

Put the engineer before the predictive estimator and pass raw data to both pipeline.fit and pipeline.predict. It records schemas rather than learned feature statistics; use fitted statistical preprocessing for training-derived means, scales, or vocabularies. Eager fit_transform evaluates expressions twice. See the [complete prediction pipeline](../getting-started.md#your-first-raw-data-prediction-pipeline).
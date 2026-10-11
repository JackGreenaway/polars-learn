# FeatureSpec workflows

## About

A feature specification makes an operation explicit: the Polars expressions to evaluate together, their output names, any columns to drop, and an optional label for diagnostics. It is a lightweight description consumed by FeatureEngineer, rather than an estimator.

## Why use it?

Use it to organize a readable feature plan without creating a transformer class for each arithmetic operation. One required expression argument accepts either a single Expr or a nonempty sequence, normalized safely to a tuple.

## Examples

### Input

| ask | bid |
| --- | --- |
| 11.0 | 9.0 |
| 22.0 | 18.0 |

### Code

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

quotes = pl.DataFrame({"ask": [11.0, 22.0], "bid": [9.0, 18.0]})
spec = FeatureSpec(
    [
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
    ],
    drop=["ask", "bid"],
    name="quote_features",
)
prepared = FeatureEngineer(spec).fit_transform(quotes)
assert prepared.to_dict(as_series=False) == {"spread": [2.0, 4.0], "mid": [10.0, 20.0]}
assert isinstance(spec.expression, tuple)
```

### Output

| spread | mid |
| --- | --- |
| 2.0 | 10.0 |
| 4.0 | 20.0 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

Sibling expressions execute together; they cannot consume columns created by siblings. Put dependent expressions in later specifications. An explicit alias requires one expression with one output. Empty sequences, invalid types, and detectable duplicate output names fail during construction. Dict configs accept the same constructor keys; there is no expressions alias.
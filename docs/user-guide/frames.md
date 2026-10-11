# Lazy execution and interoperability

All five transformers accept a `pl.LazyFrame`, including a query from `pl.scan_csv(...)`. Default output follows the input to **each call**, independently of whether fitting used eager or lazy input. Pandas input, when conversion is allowed, is treated as an eager DataFrame.

| `set_output(transform=...)`    | DataFrame input          | LazyFrame input                |
| ------------------------------ | ------------------------ | ------------------------------ |
| `"default"` (initial behavior) | DataFrame                | LazyFrame                      |
| `"polars"`                     | DataFrame                | DataFrame; collects the result |
| `"polars-lazy"`                | LazyFrame                | LazyFrame                      |
| `None`                         | Keep the existing choice | Keep the existing choice       |

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

X = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
engineer = FeatureEngineer(
    [FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")]
)
plan = engineer.fit_transform(X.lazy())  # Build a plan; do not evaluate rows
assert isinstance(plan, pl.LazyFrame)
result = plan.collect()  # Execute when ready
assert result["notional"].to_list() == [20.0, 60.0]

engineer.set_output(transform="polars")
eager_result = engineer.transform(X.lazy())  # Collect inside transform
assert isinstance(eager_result, pl.DataFrame)

engineer.set_output(transform="polars-lazy")
lazy_result = engineer.transform(X)
assert isinstance(lazy_result, pl.LazyFrame)
engineer.set_output(transform="default")
```

Output settings control `transform` and `fit_transform` returns. `fit` always returns the transformer and does not collect lazy rows just because eager output is configured. Forcing a lazy return from eager input can wrap an already computed result; pass lazy input to keep row evaluation deferred throughout.

Lazy operations check column names, order, dtypes, expression dependencies, and drops using schema metadata. Resolving metadata can access a source's metadata; it does not collect all its rows. Lazy fit validates the plan schema, not every input value. Value-dependent errors surface at collection. A returned LazyFrame is a query plan: collecting it again may execute the query again.

Inspect lazy output with `plan.collect_schema()` for names and dtypes, and `plan.collect()` for data. Verbosity reports lazy row counts as unknown rather than evaluating rows to count them.

## Optional pandas input

Install the `pandas` extra shown above, then explicitly allow conversion:

```python
import pandas as pd

from polars_learn import SchemaValidator

X = pd.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
validator = SchemaValidator(convert_pandas=True, warn_on_conversion=False)
result = validator.fit_transform(X)
assert result.columns == ["price", "quantity"]
```

All five transformers expose `convert_pandas` and `warn_on_conversion`. Conversion is disabled by default. When enabled, it emits `PandasConversionWarning` unless `warn_on_conversion=False`. Use Polars directly to avoid conversion overhead; `fit_transform` may convert input more than once. Pandas and PyArrow are imported only when needed. The extra includes PyArrow for nullable, extension, and other dtypes that require it. Conversion follows Polars conventions, including NaN to null conversion and ignoring the pandas index. Unsupported dtypes raise; conversion does not fall back to row operations. An explicit `"polars-lazy"` output choice returns a lazy wrapper after conversion.

## Choosing an execution boundary

For a lazy data source, construct the query and validate structural metadata before collecting at the final model-facing step. Schema resolution can access source metadata; lazy does not mean that no file or network access occurs. Value errors are deferred until execution, so place error handling around collection as well as plan construction. Many estimators require eager input; an explicit `set_output(transform="polars")` makes that boundary visible.

Requesting lazy output after an eager input does not undo work already performed. Supply a LazyFrame from the beginning when row execution must remain deferred. Saved transformers retain configuration and fitted schemas, not a snapshot of external input files; provide available data sources at prediction time.

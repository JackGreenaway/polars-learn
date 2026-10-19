# Lazy execution and interoperability

All six transformers accept a `pl.LazyFrame`, including a query from `pl.scan_csv(...)`. Default output follows the input to **each call**, independently of whether fitting used eager or lazy input. Pandas input, when conversion is allowed, is treated as an eager DataFrame. `ValueValidator` is the exception to deferred execution: it materializes lazy input to check values, then returns an in-memory LazyFrame by default.

| `set_output(transform=...)`    | DataFrame input          | LazyFrame input                |
| ------------------------------ | ------------------------ | ------------------------------ |
| `"default"` (initial behavior) | DataFrame                | LazyFrame                      |
| `"polars"`                     | DataFrame                | DataFrame; collects the result |
| `"polars-lazy"`                | LazyFrame                | LazyFrame                      |
| `"pandas"`                    | pandas DataFrame         | pandas DataFrame; collects and converts |
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

Lazy operations check column names, order, dtypes, expression dependencies, and drops using schema metadata. Resolving metadata can access a source's metadata; it does not collect all its rows. Fits validate schemas, not input values. `ValueValidator` resolves check schemas during fit, then materializes and checks the batch during transform; its default LazyFrame output wraps the checked in-memory data, so later collection does not rerun the source. A returned LazyFrame from other transformers is a query plan: collecting it again may execute the query again.

Inspect lazy output with `plan.collect_schema()` for names and dtypes, and `plan.collect()` for data. Verbosity reports lazy row counts as unknown rather than evaluating rows to count them.

## Optional pandas input

Pandas DataFrames are supported through behind-the-scenes conversion to a Polars DataFrame. Feature transformations then use the same Polars API as native input, and the output is a Polars frame by default. Pandas output is available explicitly for compatibility. Enable conversion on each transformer that receives pandas input.

Install the optional pandas dependencies:

```bash
uv add 'polars-learn[pandas]'
# Or use pip:
pip install 'polars-learn[pandas]'
```

See [installation](../getting-started.md#installation) for the package requirements. Then explicitly allow conversion on the transformer:

```python
import pandas as pd

from polars_learn import SchemaValidator

X = pd.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
validator = SchemaValidator(convert_pandas=True, warn_on_conversion=False)
result = validator.fit_transform(X)
assert result.columns == ["price", "quantity"]
```

All six transformers expose `convert_pandas` and `warn_on_conversion`. Conversion is disabled by default. When enabled, it emits `PandasConversionWarning` unless `warn_on_conversion=False`. Use Polars directly to avoid conversion overhead; `fit_transform` may convert input more than once. Pandas and PyArrow are imported only when needed. The extra includes PyArrow for nullable, extension, and other dtypes that require it. Conversion follows Polars conventions, including NaN to null conversion and ignoring the pandas index. Unsupported dtypes raise; conversion does not fall back to row operations. An explicit `"polars-lazy"` output choice returns a lazy wrapper after conversion.

## Pandas output for compatibility

Prefer native Polars input and output for speed: it keeps preparation in Polars and avoids conversion overhead. Use `set_output(transform="pandas")` when an existing consumer needs a pandas DataFrame. All six transformers support this option on `transform` and `fit_transform`; feature calculations and schema checks still happen in Polars.

Install `polars-learn[pandas]` using the commands in [Optional pandas input](#optional-pandas-input). Pandas output requires pandas and PyArrow. You do not need `convert_pandas=True` when your input is already Polars; that flag controls pandas **input** independently.

```python
import pandas as pd
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

raw = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
engineer = FeatureEngineer(
    FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
).set_output(transform="pandas")

result = engineer.fit_transform(raw.lazy())
assert isinstance(result, pd.DataFrame)
assert result["notional"].tolist() == [20.0, 60.0]
assert result.columns.tolist() == ["price", "quantity", "notional"]
```

Lazy input is collected before conversion to pandas. `fit` still returns the transformer and does not collect lazy rows merely because pandas output is configured. Conversion uses [Polars `to_pandas()` defaults](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_pandas.html): names and row order are retained, the result has a fresh RangeIndex, and dtypes follow pandas conversion conventions. For example, a nullable integer column may become a floating-point column with NaN values. The original pandas index is not restored.

Keep intermediate pipeline steps in Polars and request pandas at the final preparation step when possible. If you configure an entire multi-step pipeline to produce pandas, enable `convert_pandas=True` on each native transformer receiving the previous step's pandas output; repeated conversions add overhead. Global scikit-learn output settings remain bypassed; configure these transformers or their pipeline explicitly with `set_output`.

## Choosing an execution boundary

For a lazy data source, construct the query and validate structural metadata before collecting at the final model-facing step. Schema resolution can access source metadata; lazy does not mean that no file or network access occurs. Value errors are deferred until execution, so place error handling around collection as well as plan construction. Many estimators require eager input; an explicit `set_output(transform="polars")` makes that boundary visible.

Requesting lazy output after an eager input does not undo work already performed. Supply a LazyFrame from the beginning when row execution must remain deferred. Saved transformers retain configuration and fitted schemas, not a snapshot of external input files; provide available data sources at prediction time.

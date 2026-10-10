# polars-learn

High-performance feature engineering for scikit-learn, powered by Polars.

## Features

- Composable native Polars expressions, evaluated in dependency order.
- Declarative schema contracts and column selection for reusable data preparation.
- Learned schemas with strict, semi-strict, and permissive validation.
- scikit-learn estimators with cloning, parameters, and feature names.
- Native Polars output and optional, explicit pandas conversion.
- Python 3.11+, Polars 1.20+, and scikit-learn 1.6+.

## Installation

```bash
uv add polars-learn
# From a local checkout:
uv pip install .
# Optional pandas and PyArrow support:
uv pip install '.[pandas]'
```

## Quick Start

```python
import polars as pl
from polars_learn import (
    ColumnSelector,
    FeatureEngineer,
    FeatureSpec,
    SchemaSpec,
    SchemaValidator,
)

X = pl.DataFrame({"price": [10.0, 20.0, 30.0], "quantity": [2, 3, 4]})

schema = SchemaSpec({"price": pl.Float64, "quantity": pl.Int64})
validated = schema.fit_transform(X)

selector = ColumnSelector(["quantity", "price"], rename={"quantity": "qty"})
selected = selector.fit_transform(validated)
print(selected.columns)
# ['qty', 'price']

validator = SchemaValidator(mode="strict").fit(X)
validated = validator.transform(X)

engineer = FeatureEngineer(
    [
        FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
        FeatureSpec(
            pl.col("notional").log1p(),
            alias="log_notional",
            drop=["quantity"],
        ),
    ]
)
result = engineer.fit_transform(validated)
print(result)
print(engineer.get_feature_names_out())
# ['price' 'notional' 'log_notional']
```

## Feature Engineering

`FeatureSpec` accepts exactly one of `expression` (one `pl.Expr`) or `expressions` (a nonempty sequence of expressions). It stores a single canonical expression tuple. Names follow Polars conventions: an existing alias or a native output name needs no additional alias. An explicit `alias` overrides the name of one output; it is rejected for multiple outputs. Selectors and expression expansion are supported, with their output names resolved against the frame.

```python
spec = FeatureSpec(
    expressions=[
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
    ],
    drop=["ask", "bid"],
    name="quote_features",
)
quotes = pl.DataFrame({"ask": [11.0], "bid": [9.0]})
print(FeatureEngineer([spec]).fit_transform(quotes))
```

Expressions within one specification are evaluated together and cannot reference new columns created by siblings. Separate specifications execute sequentially; later ones can use earlier outputs. Drops happen after each specification. Missing expression inputs, missing drop columns, and duplicate output names raise `FeatureEngineeringError` (duplicate names detectable at construction raise `ValueError`). Overwriting existing columns is allowed and keeps Polars' column position. For example, `FeatureSpec(pl.col("price") + 1)` replaces `price`.

`fit` executes the training transformation to validate expressions and learn output names and dtypes. `transform` requires the fitted input column names and order and the exact fitted output schema. Input dtype differences are acceptable only when the resulting schema remains identical; use `SchemaValidator` first for explicit input type policies. Empty feature lists are identity transforms. The public API accepts eager DataFrames; internally, lazy stages preserve sequential semantics and collect once per execution. Expressions are stateless; training transformations do not learn statistics. Use deterministic expressions to keep training and inference consistent.

## Schema Contracts and Column Selection

`SchemaSpec` is a lightweight, declarative contract for required columns and dtypes. It validates DataFrames without needing a fitted estimator state and is useful when you want a reusable schema assertion in training or validation code.

```python
schema = SchemaSpec({"price": pl.Float64, "quantity": pl.Int64})
print(schema.fit_transform(X).schema)
```

`ColumnSelector` keeps or reorders a chosen subset of columns, optionally renaming them to a final output schema. It suits preprocessing steps such as separating feature blocks, keeping a fixed input order for downstream estimators, or producing a tidy column set before feature engineering.

```python
selector = ColumnSelector(["quantity", "price"], rename={"quantity": "qty"})
print(selector.fit_transform(X).columns)
# ['qty', 'price']
```

## Schema Validation

```python
validator = SchemaValidator(
    mode="semi-strict",
    allow_extra_columns=False,
    enforce_column_order=True,
)
validator.fit(X)  # Always learns the raw input schema
print(validator.schema_)
validated = validator.transform(X)  # Always validates against that schema
```

| Mode               | Policy for differing dtypes                             |
| ------------------ | ------------------------------------------------------- |
| `strict` (default) | Reject every dtype difference.                          |
| `semi-strict`      | Allow only the explicit lossless widening policy below. |
| `permissive`       | Use Polars `cast(strict=True)` and report failures.     |

Semi-strict allows integer widening when the entire source dtype's range fits inside the target integer dtype (including compatible unsigned-to-signed widening), Float32 → Float64, and integer → Float64 only when **every non-null value round-trips exactly**. For example, Int64 `2**53` and `2**54` are accepted, but `2**53 + 1` is rejected. It rejects narrowing, strings to numbers, and Null dtype to a different dtype. Null values in already compatible columns are preserved; nullability constraints are not part of this version.

Permissive casts may be lossy according to Polars semantics, such as floats to integers. Failed casts never silently replace invalid values with nulls. Missing required columns always fail; multiple schema mismatches are reported in `SchemaValidationError`. Strict mode does not cast or rename columns.

Extra columns fail by default. Set `allow_extra_columns=True` to preserve them, or also set `extra_columns="drop"` to explicitly remove them. Column order checks compare the relative order of required columns; permitted extras may appear anywhere. With `enforce_column_order=False`, output is normalised to fitted column order, followed by preserved extras in their input order. `get_feature_names_out()` describes fitted required columns; inference-only extras are not included. Avoid preserving inference-only extras before a fixed-width downstream estimator or `FeatureEngineer`.

The shorthand `strict=True` means strict mode, and `strict=False` means permissive mode. Conflicting `strict` and `mode` arguments are rejected. Schemas learned by the validator and feature engineer are independent.

## scikit-learn Pipelines

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline(
    [
        ("schema", SchemaValidator(mode="semi-strict")),
        ("features", engineer),
        ("scale", StandardScaler().set_output(transform="polars")),
        ("model", RandomForestRegressor(n_estimators=20, random_state=0)),
    ]
)
pipeline.fit(X, np.array([1.0, 2.0, 3.0]))
print(pipeline.predict(X))
print(pipeline[:-1].get_feature_names_out())
```

These transformers implement `fit`, `transform`, `fit_transform`, `get_params`, `set_params`, and `get_feature_names_out`. Fitted attributes include `feature_names_in_`, `n_features_in_`, and `output_features_`. Cloning discards fitted state. Supplied `input_features` must exactly match fitted input names.

Data stays in Polars between our transformers. Both support `set_output(transform="default")` and `set_output(transform="polars")`, which return native Polars frames. They reject `"pandas"` output and deliberately bypass scikit-learn's automatic output wrapping, including global output settings, to preserve native names and allowed extras. No automatic NumPy conversion occurs. Downstream estimator support for Polars depends on the estimator and installed scikit-learn version; arbitrary string columns do not work with random forests.

Tests run applicable scikit-learn constructor, parameter, and cloning checks. Generic checks requiring NumPy array input do not apply to this deliberately DataFrame-only API. No LazyFrame, sparse-array, or NumPy input is accepted.

## Optional Pandas Support

```bash
uv sync --extra pandas
```

```python
# Requires the pandas extra; both transformers expose these options.
import pandas as pd

pandas_X = pd.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
validator = SchemaValidator(convert_pandas=True, warn_on_conversion=False)
polars_X = validator.fit_transform(pandas_X)
```

Pandas input is rejected unless `convert_pandas=True`. Conversion returns Polars and emits `PandasConversionWarning` about memory and performance overhead unless `warn_on_conversion=False`. Imports are lazy; core functionality works without pandas or PyArrow. The extra includes PyArrow because nullable, extension, and other non-NumPy-backed pandas dtypes can require it for `pl.from_pandas`. Conversion follows Polars conventions (including NaN-to-null and ignoring the pandas index). Unsupported conversion dtypes raise rather than falling back to row-wise operations.

## Development

```bash
uv sync
uv sync --extra pandas     # Optional interoperability test coverage
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

`uv.lock` pins reproducible development dependencies. Pyright includes its Node runtime through the development extra, so no system Node installation is needed. Pandas tests use `pytest.importorskip("pandas")`; core tests also run without the optional extra. The package ships `py.typed` for downstream typing tools.

Library requirements use minimum versions without speculative upper bounds. Add an upper bound or exclude a release only when a known incompatibility requires it; the lockfile pins the development environment without constraining downstream applications.

### Releases

Configure the repository or `pypi` environment secret `PYPI_API_TOKEN` with a PyPI upload token for `polars-learn`. The publishing job uses the `pypi` environment.

Update `project.version` in `pyproject.toml` and `__version__` in `src/polars_learn/__init__.py`, run `uv lock`, and push the changes. Publish a GitHub release from the GitHub Releases page to trigger **Upload Python Package**. It builds distributions, transfers them to a separate publishing job, and uploads them to PyPI.

**Create GitHub Release** remains available to create a release with validated distributions. Releases created by that workflow with GitHub's workflow token do not trigger **Upload Python Package**; use the GitHub Releases page for the publishing flow.

## License

MIT. See [LICENSE](LICENSE).

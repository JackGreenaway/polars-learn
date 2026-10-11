# Feature engineering

Feature engineering is part of the model's input preparation. Keeping it in a fitted pipeline gives training and prediction one shared preparation path: validate the raw schema, compute dependent features in order, select or drop columns, and pass the prepared frame to the estimator. New observations arrive as raw data; the pipeline prepares them at prediction time.

| You define once | The fitted pipeline applies it |
| --- | --- |
| Raw column and dtype requirements | Validate incoming observations before feature preparation. |
| Feature expressions and their dependencies | Generate the same named features during training and prediction. |
| Columns to retain or drop | Provide the fitted feature layout to the model. |
| Additional learned preprocessing | Reuse fitted scaling or other estimator-based preprocessing steps. |

This keeps the preparation logic visible alongside the estimator and reduces the chance of omitting a step or using a different feature definition in prediction code. `FeatureEngineer` executes your expressions and records schema expectations; it does not learn feature statistics. Put transformations that need learned training statistics in additional fitted pipeline steps.

## Defining features

`FeatureEngineer` turns an ordered set of Polars expressions into a reusable pipeline step. Each `transform` evaluates those definitions against the incoming raw frame, so the fitted pipeline can prepare new observations whenever you call `predict`. The standalone examples below show how to define and inspect that step; in a predictive workflow, include it before the final estimator as shown in [your first prediction pipeline](../getting-started.md#your-first-raw-data-prediction-pipeline).

Create a `FeatureSpec` with one required `expression` argument: pass a single `pl.Expr` or a nonempty sequence of expressions. The specification normalizes either form to a tuple without changing your input sequence. Use native expression names, an expression's `.alias(...)`, or the specification's `alias` argument to name its outputs. An explicit `alias` requires exactly one expression with one output.

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

quotes = pl.DataFrame({"ask": [11.0], "bid": [9.0]})
spec = FeatureSpec(
    expression=[
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
    ],
    drop=["ask", "bid"],
    name="quote_features",
)
result = FeatureEngineer(spec).fit_transform(quotes)
assert result.to_dict(as_series=False) == {"spread": [2.0], "mid": [10.0]}
```

`FeatureEngineer` accepts a single `FeatureSpec` directly, as in the example above, a dict of `FeatureSpec` keyword arguments, or an ordered sequence mixing those forms. Dicts are unpacked with `FeatureSpec(**config)` and undergo the same validation. For example:

```python
import polars as pl

from polars_learn import FeatureEngineer

X = pl.DataFrame({"price": [10.0, 20.0]})
engineer = FeatureEngineer({"expression": pl.col("price") * 2, "alias": "doubled"})
assert engineer.fit_transform(X)["doubled"].to_list() == [20.0, 40.0]
```

A dict uses the exact keys `expression`, `alias`, `drop`, `name`, and `verbosity` accepted by `FeatureSpec`. The required `expression` value accepts one Polars expression or a nonempty sequence; strings, generators, and empty sequences are rejected. Expressions must be actual Polars expression objects; expression strings are not evaluated. Unknown keys and invalid values fail at construction, with the configuration's sequence index included when applicable. An empty dict is invalid; use `FeatureEngineer([])` for an identity operation.

Single definitions are treated as one-element sequences internally; constructor parameters, including original dicts, are retained unchanged for scikit-learn cloning and `get_params`. The same forms are accepted by `set_params(features=...)`; refit after changes.

Expressions within one specification run together. They can reference columns already present, but cannot reference columns newly created by another expression in that same specification. Use separate specifications for dependent features. Specifications run in list order; drops occur after each specification's expressions. Later specifications cannot use dropped columns.

Existing columns can be overwritten: `FeatureSpec(pl.col("price") + 1)` replaces `price`. Selectors such as `pl.all().name.suffix("_copy")` can expand to multiple outputs. Duplicate output names and missing expression or drop inputs fail during schema checking; duplicates detectable when constructing a specification fail there. An empty `FeatureEngineer([])` leaves the columns unchanged.

`FeatureEngineer.transform` requires the fitted input names in exactly the same order, and the resulting output names, order, and dtypes must match the fitted output schema. Input dtypes may differ if the final output schema still matches. Use a `SchemaValidator` first when input types need a defined conversion policy.

Fitting eager input evaluates the feature expressions. Fitting lazy input checks the plan's schema without evaluating rows. Features do not learn statistics; use deterministic expressions. Currently `fit_transform` calls both `fit` and `transform`, so eager feature expressions are evaluated twice.

## Sibling features and dependent stages

**Feature specifications are applied in exactly the order you supply them.** In `FeatureEngineer([first, second, third])`, the first specification adds its features and applies its drops, the second receives that result, and the third receives the second's result. This order is preserved during both fitting and transformation, so later specifications can build on columns created earlier. A column dropped by an earlier stage is no longer available.

For a LazyFrame, the specifications build a query plan in that dependency order; row evaluation remains deferred until collection. Polars may optimize execution while preserving the query's meaning. Specification order describes how the transformations compose, rather than requiring each stage to materialize a separate frame.

Sibling features are expressions in the same `FeatureSpec`. They use the columns available at the start of that stage. Group independent calculations together, then add another specification when a calculation needs their newly created outputs.

Here, `spread` and `mid` are siblings: both use the raw `ask` and `bid` columns. The next stage combines them into `relative_spread`. The final stage derives `spread_bps` from that result.

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

quotes = pl.DataFrame({"ask": [11.0, 22.0], "bid": [9.0, 18.0]})

# Stage 1: sibling expressions share the same raw input columns.
quote_features = FeatureSpec(
    expression=[
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
    ],
    drop=["ask", "bid"],
    name="quote_features",
)

# Stage 2: both sibling outputs now exist and can be combined.
relative_spread = FeatureSpec(
    pl.col("spread") / pl.col("mid"),
    alias="relative_spread",
    name="relative_spread",
)

# Stage 3: use the previous stage's output and drop intermediates afterward.
spread_bps = FeatureSpec(
    pl.col("relative_spread") * 10_000,
    alias="spread_bps",
    drop=["spread", "mid", "relative_spread"],
    name="spread_bps",
)

# Join the stages together in their dependency order.
engineer = FeatureEngineer([quote_features, relative_spread, spread_bps])
result = engineer.fit_transform(quotes)
assert result.to_dict(as_series=False) == {"spread_bps": [2000.0, 2000.0]}
assert engineer.get_feature_names_out().tolist() == ["spread_bps"]
```

| Stage | Reads | Creates | Drops after calculation |
| --- | --- | --- | --- |
| Sibling quote features | `ask`, `bid` | `spread`, `mid` | `ask`, `bid` |
| Relative spread | `spread`, `mid` | `relative_spread` | None |
| Basis-point spread | `relative_spread` | `spread_bps` | `spread`, `mid`, `relative_spread` |

The input rows `(ask=11, bid=9)` and `(ask=22, bid=18)` both have a spread equal to 20% of their midpoint, producing 2,000 basis points. Only `spread_bps` remains in the output because the final stage drops the intermediate features. Omit those drops if your model should also receive the intermediates.

!!! note "A sibling cannot read a new sibling output"
    Put `spread` and `mid` in one specification, then compute `relative_spread` in the next. Adding the ratio as a third expression in the first specification would try to read columns that do not exist yet. Expressions in a single specification do not execute as sequential assignments.

### Choosing one expression or several stages

You can also compute the same final feature in one specification. Compose the calculations directly from the raw columns instead of referring to newly created sibling names:

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

quotes = pl.DataFrame({"ask": [11.0, 22.0], "bid": [9.0, 18.0]})
spread = pl.col("ask") - pl.col("bid")
mid = (pl.col("ask") + pl.col("bid")) / 2

engineer = FeatureEngineer(
    FeatureSpec(
        (spread / mid) * 10_000,
        alias="spread_bps",
        drop=["ask", "bid"],
    )
)
result = engineer.fit_transform(quotes)
assert result.to_dict(as_series=False) == {"spread_bps": [2000.0, 2000.0]}
```

Here, `spread` and `mid` are Python variables holding Polars expressions, not newly created DataFrame columns. Polars composes them into one expression that reads `ask` and `bid`. You could also include these expressions as separate sibling outputs in the same specification while using the composed expression for `spread_bps`.

Choose the form that suits your workflow. One specification is compact when you only need the final result. Separate stages make intermediate columns explicit when you want to retain them as model features, inspect their values, or refer to their names in later transformations. The restriction is only on reading a column newly created by a sibling; composing expressions from existing input columns works within one specification. These examples illustrate equivalent feature values, not a performance comparison.

### Connecting the stages to a model

To include the joined stages in a model, use a feature engineer containing the joined stages as a pipeline step. This standalone example keeps the relative spread as the model feature:

```python
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from polars_learn import FeatureEngineer, FeatureSpec

quote_features = FeatureSpec(
    [
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
    ]
)
relative_spread = FeatureSpec(
    pl.col("spread") / pl.col("mid"),
    alias="relative_spread",
    drop=["ask", "bid", "spread", "mid"],
)
engineer = FeatureEngineer([quote_features, relative_spread])
pipeline = Pipeline([("features", engineer), ("model", LinearRegression())])

raw_train = pl.DataFrame({"ask": [11.0, 12.0, 14.0], "bid": [9.0, 8.0, 6.0]})
pipeline.fit(raw_train, np.array([1.0, 2.0, 4.0]))

# Supply raw quotes; the pipeline computes siblings, combines them, and predicts.
raw_predict = pl.DataFrame({"ask": [13.0], "bid": [7.0]})
predictions = pipeline.predict(raw_predict)
assert np.allclose(predictions, [3.0])
```

This pipeline uses two stages and retains `relative_spread` as its model input. Each prediction runs both stages on the new raw quotes before calling the fitted regressor.

## Deterministic expressions versus learned preprocessing

`FeatureEngineer` replays expressions; it does not learn expression-specific statistics. An expression such as subtracting the current batch mean recomputes that mean on every transform call, which can make a prediction depend on the other rows in the request. Use a fitted statistical transformer when a quantity must be learned from training data and held fixed at inference. Avoid random or time-dependent expressions when reproducible predictions are required.

## Evaluation and mutation

Input frames are never mutated. Expressions within one specification execute together and cannot consume new sibling outputs; split dependencies into ordered specifications. Existing names may be deliberately overwritten. `fit_transform` calls `fit` and then `transform`, so eager feature expressions execute twice. Lazy inputs defer row execution unless eager output is requested. Prefer native expressions to Python callbacks for portable, reviewable feature logic.

See [`FeatureSpec`](../api.md#featurespec) and [`FeatureEngineer`](../api.md#featureengineer) for exact validation, fitted schema, and exception behavior.

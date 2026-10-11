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

`FeatureEngineer` turns an ordered set of Polars expressions into a reusable pipeline step. Each `transform` evaluates those definitions against the incoming raw frame, so the fitted pipeline can prepare new observations whenever you call `predict`. The standalone examples below show how to define and inspect that step; in a predictive workflow, include it before the final estimator as in the quick start.

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

## Deterministic expressions versus learned preprocessing

`FeatureEngineer` replays expressions; it does not learn expression-specific statistics. An expression such as subtracting the current batch mean recomputes that mean on every transform call, which can make a prediction depend on the other rows in the request. Use a fitted statistical transformer when a quantity must be learned from training data and held fixed at inference. Avoid random or time-dependent expressions when reproducible predictions are required.

## Evaluation and mutation

Input frames are never mutated. Expressions within one specification execute together and cannot consume new sibling outputs; split dependencies into ordered specifications. Existing names may be deliberately overwritten. `fit_transform` calls `fit` and then `transform`, so eager feature expressions execute twice. Lazy inputs defer row execution unless eager output is requested. Prefer native expressions to Python callbacks for portable, reviewable feature logic.

See [`FeatureSpec`](../api.md#featurespec) and [`FeatureEngineer`](../api.md#featureengineer) for exact validation, fitted schema, and exception behavior.

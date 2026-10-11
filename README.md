<div align="center">

# 🐻‍❄️ polars-learn

**Raw data in. Model-ready features out.**

Composable expressions · Clear schema contracts · Eager and lazy execution

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Polars 1.20+](https://img.shields.io/badge/Polars-1.20%2B-CD792C?style=flat-square)](https://pola.rs/)
[![scikit-learn 1.6+](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

[![Tests](https://github.com/JackGreenaway/polars-learn/actions/workflows/tests.yml/badge.svg)](https://github.com/JackGreenaway/polars-learn/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/polars-learn?style=flat-square&color=0F766E)](https://pypi.org/project/polars-learn/)
[![MIT license](https://img.shields.io/badge/License-MIT-0F766E?style=flat-square)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/Code%20style-Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)

[Quick start](#quick-start) · [API reference](docs/api.md) · [LazyFrames](#lazyframes-and-return-types) · [Persistence](#saving-and-loading-a-fitted-pipeline) · [Releases](#releases)

</div>

---

Prepare Polars data for estimators that implement the scikit-learn interface, with reusable feature expressions, column selection, and schema validation. Use these transformers in a `sklearn.pipeline.Pipeline`, following the same fit-and-apply conventions used by scikit-learn and estimator-style libraries such as XGBoost. Work with eager `DataFrame` inputs or keep `LazyFrame` queries deferred until the model needs materialized features.

The power of these transformations comes from the **Polars API**: its column expressions, selectors, and eager and lazy execution. `polars-learn` wraps those capabilities in reusable estimator-style components, keeping your Polars feature logic inside the fitted pipeline.

**Pandas input and output are supported for compatibility.** Install `polars-learn[pandas]` and set `convert_pandas=True` on each transformer receiving pandas input. Pandas DataFrames are converted to Polars behind the scenes; transformations run through the Polars API and return Polars frames by default. Request `set_output(transform="pandas")` for pandas compatibility output. Conversion is opt-in and disabled by default. Prefer native Polars for speed and to avoid conversion overhead. See [pandas output](docs/user-guide/frames.md#pandas-output-for-compatibility) for dependencies, dtype behavior, and pipeline guidance.

**Put feature preparation inside the pipeline.** Define your Polars expressions once with `FeatureEngineer`, train with `pipeline.fit(raw_X, y)`, and predict with `pipeline.predict(new_raw_X)`. The pipeline generates the features before the model runs, using the same definitions and execution order in both paths. You can keep feature preparation together with the model instead of maintaining separate feature-building functions and remembering to call them before each prediction.

Requires Python 3.11+, Polars 1.20+, and scikit-learn 1.6+. CI runs on Python **3.11 and 3.12**, including pandas interoperability tests.

## Documentation

Read the [complete documentation](https://jackgreenaway.github.io/polars-learn/) for the user guide, API reference, runnable examples, troubleshooting, and production guidance. The sources live in [docs/index.md](docs/index.md); [docs/development.md](docs/development.md) explains local preview and GitHub Pages publishing. The site becomes available after the Documentation workflow deploys successfully.

## Why use polars-learn?

If standard scaling and polynomial features belong inside a fitted pipeline, why leave full feature generation outside it? The [homepage walkthrough](docs/index.md#why-pipeline-scaling-but-leave-feature-generation-outside) compares a separate preparation function with a pipeline that carries its feature definitions into prediction, and shows [the issue each class addresses](docs/index.md#how-each-class-addresses-a-concrete-risk).

**Bring the power of the Polars API into the fitted model pipeline.** Define feature logic with native Polars expressions, compose it with an estimator, and reuse the same preparation during training, validation, and prediction.

- **Express rich transformations without a new class for every feature.** Combine arithmetic, conditions, string and datetime operations, casts, null handling, and selectors in reusable `FeatureSpec` definitions. Polars supplies the expression engine; `polars-learn` makes that logic a pipeline component.
- **Build features in stages.** Calculate transaction value from price and quantity, derive a logarithm from that new feature, then drop raw columns. Specifications are applied in the order you supply them: each stage adds its features and applies its drops before the next stage receives the result. Later stages can therefore use retained outputs from earlier stages.
- **Predict directly from raw observations.** Train with `pipeline.fit(raw_X, y)` and predict with `pipeline.predict(new_raw_X)`. The pipeline applies the configured feature definitions before the estimator runs.
- **Enforce names, order, and dtypes.** Declare or learn schema contracts, select model inputs, and check the transformed schema against the fitted expectations.
- **Use eager or lazy Polars workflows.** Prepare in-memory frames or build deferred plans, then choose where to materialize data for your estimator.
- **Accept pandas input.** Enable `convert_pandas=True` with the pandas extra installed. Frames are converted to Polars behind the scenes, transformations use the Polars API, and outputs are Polars frames by default. Use `set_output(transform="pandas")` for compatibility with pandas consumers.
- **Carry preparation with the model.** Clone components for cross-validation, inspect fitted feature names and schemas, and save the fitted pipeline with its preparation logic and estimator.

`FeatureEngineer` executes expressions and records schema expectations. Add fitted preprocessing steps for transformations that must learn and retain training statistics. Read [why use it?](docs/index.md#why-use-it) for the full breakdown and workflow links.

## Installation

Install the package into your project:

```bash
uv add polars-learn
# Or use pip:
pip install polars-learn
```

For optional pandas input support:

```bash
uv add 'polars-learn[pandas]'
# Or use pip:
pip install 'polars-learn[pandas]'
```

See [Development](#development) for running this repository from a checkout.

## Choose a Class

| Class                | Use it when                                                    | What fitting does                                              |
| -------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| `SchemaSpec`         | You already know the required column names and dtypes.         | Checks your explicit contract and records output names.        |
| `SchemaValidator`    | Training data should define the schema for later input.        | Learns every training column's name, order, and dtype.         |
| `FeatureSpec`        | You want to describe one feature operation.                    | This is a description, not a transformer; there is no `fit`.   |
| `FeatureEngineer`    | You want to execute feature operations in order.               | Validates the transformation and records input/output schemas. |
| `ColumnSelector`     | You want to keep, reorder, or rename selected columns.         | Checks selected columns and records their output names.        |
| `FeaturePassthrough` | You want an explicit unchanged-column pipeline step or branch. | Checks selected columns and records their unchanged names.     |

The five transformers implement the scikit-learn estimator interface: `fit`, `transform`, `fit_transform`, `get_feature_names_out`, `get_params`, `set_params`, and `set_output`. They support cloning and named pipeline parameters, such as `features__features` or `model__max_depth`. `FeatureSpec` is an operation description, not an estimator. All six classes accept `verbosity`. See the [API reference](docs/api.md) for every constructor option, fitted attribute, and public method.

## Quick Start

Define the preparation once and train the complete pipeline on raw input. This example creates `notional` and `log_notional` inside the pipeline, removes `quantity` from the model's feature set, and retains `price`.

```python
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from polars_learn import FeatureEngineer, FeatureSpec, SchemaValidator

raw_train = pl.DataFrame({"price": [10.0, 20.0, 30.0, 40.0], "quantity": [2, 3, 4, 5]})
y = np.array([1.0, 2.0, 3.0, 4.0])

pipeline = Pipeline(
    [
        ("raw_schema", SchemaValidator()),
        (
            "features",
            FeatureEngineer(
                [
                    FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
                    FeatureSpec(
                        pl.col("notional").log1p(),
                        alias="log_notional",
                        drop=["quantity"],
                    ),
                ]
            ).set_output(transform="polars"),
        ),
        ("model", RandomForestRegressor(n_estimators=20, random_state=0)),
    ]
)

pipeline.fit(raw_train, y)

# Prediction takes new raw observations; feature preparation runs inside the pipeline.
raw_predict = pl.DataFrame({"price": [15.0, 25.0], "quantity": [2, 4]})
predictions = pipeline.predict(raw_predict)
assert predictions.shape == (2,)
assert np.isfinite(predictions).all()

# Inspect the prepared feature names without building features outside the pipeline.
assert pipeline[:-1].get_feature_names_out().tolist() == [
    "price",
    "notional",
    "log_notional",
]
assert raw_predict.columns == ["price", "quantity"]
```

The prediction input contains `price` and `quantity`; you supply neither `notional` nor `log_notional`. `pipeline.predict(raw_predict)` validates the raw schema, computes the derived columns, drops `quantity` from the model input, and then predicts. Keep the target `y` separate from the raw feature frame. Fit the complete pipeline on training input and reuse that fitted pipeline for validation and prediction.

The pipeline expects the fitted raw input contract at prediction time. Do not feed it a frame that has already been feature-engineered, or refit its preprocessing on prediction data. Missing required raw columns and incompatible schemas raise errors according to the configured policies.

The individual transformers also work on their own: `fit(X)` returns the transformer, `transform(X)` applies it, and `fit_transform(X)` fits and prepares the same input. Calling `transform` or `get_feature_names_out` before a successful fit raises scikit-learn's `NotFittedError`. Input frames are never mutated.

## Feature Engineering

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

## Fixed Contracts and Column Selection

`SchemaSpec` checks an explicit, nonempty schema. Required columns must have exactly the specified dtypes; this class does not cast them. Both its `fit` and `transform` validate input, and `transform` requires a successful fit.

`ColumnSelector` returns just the columns you choose, in the requested order. Unselected columns are ignored, input order may differ, and dtypes are not validated. Selected and renamed output names must be unique.

```python
import polars as pl

from polars_learn import ColumnSelector, SchemaSpec

X = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
contract = SchemaSpec({"price": pl.Float64, "quantity": pl.Int64})
validated = contract.fit_transform(X)

selector = ColumnSelector(["quantity", "price"], rename={"quantity": "qty"})
selected = selector.fit_transform(validated)
assert selected.columns == ["qty", "price"]
```

For `SchemaSpec`, extra columns are rejected by default. If extras are allowed, `extra_columns="drop"` produces only the required columns; `"preserve"` puts required columns first and then extras in their input order. Preserved extra column names become part of the fitted output names. Later transforms must produce those same names in the same order; their dtypes are not checked. Use `SchemaValidator` if inference may introduce new extra columns.

## Explicit Feature Passthrough

Use `FeaturePassthrough(columns)` when a pipeline layer should simply carry named columns forward. It returns just those columns in the specified order, with their Polars values, names, dtypes, nulls, and row order preserved. There is no renaming or casting option. Selected columns must exist; input order and dtypes may differ between calls. It supports the same verbosity, pandas conversion, fitted feature names, and output controls as the other transformers.

```python
import polars as pl
from sklearn.pipeline import Pipeline

from polars_learn import FeaturePassthrough

X = pl.DataFrame({"id": [7, 8], "flag": [True, False], "unused": [1, 2]})
pipeline = Pipeline([("unchanged", FeaturePassthrough(["id", "flag"]))])
result = pipeline.fit_transform(X.lazy())
assert isinstance(result, pl.LazyFrame)
assert result.collect().equals(X.select("id", "flag"))
```

A sequential `Pipeline` passes each step's output to the next step, so a passthrough step selecting two columns omits all the others. To keep unchanged columns **alongside** transformed columns, put the operations in parallel `ColumnTransformer` branches:

```python
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from polars_learn import FeaturePassthrough

X = pl.DataFrame({"value": [10.0, 20.0], "id": [7, 8], "flag": [True, False]})
preprocessing = ColumnTransformer(
    [
        ("scaled", StandardScaler(), ["value"]),
        ("unchanged", FeaturePassthrough(["id", "flag"]), ["id", "flag"]),
    ],
    verbose_feature_names_out=False,
).set_output(transform="polars")
result = preprocessing.fit_transform(X)
assert result["value"].to_list() == [-1.0, 1.0]
assert result.select("id", "flag").equals(X.select("id", "flag"))
assert result.columns == ["value", "id", "flag"]
```

The outer column lists tell `ColumnTransformer` which input to send to each branch; `FeaturePassthrough` declares its own selected-column contract. Use eager input at this boundary (`lazy_X.collect()` for a lazy query). Polars output keeps the separate column dtypes. `verbose_feature_names_out=False` prevents branch prefixes on output names; final names must then be unique. For branch behavior, see the [scikit-learn ColumnTransformer reference](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html). Use `ColumnSelector` when you also need optional renaming.

## Learned Schema Validation

`SchemaValidator.fit` learns all training columns as required columns. Transform checks them against the learned dtypes according to `mode`:

| Mode                 | Treatment of a different input dtype                                                |
| -------------------- | ----------------------------------------------------------------------------------- |
| `"strict"` (default) | Reject it; no casting.                                                              |
| `"semi-strict"`      | Allow only the lossless widening policy below.                                      |
| `"permissive"`       | Attempt a Polars cast with `strict=True`; some successful casts may lose precision. |

Semi-strict permits integer widening when the **entire source dtype's range** fits inside the fitted integer dtype's range, including compatible unsigned to signed widening. Small observed values alone do not make narrowing acceptable. It also permits Float32 to Float64, and integer to Float64 only when every non-null value round-trips exactly. For example, `2**53` and `2**54` are accepted, but `2**53 + 1` is rejected. String to numeric conversion, narrowing, and Null dtype to another dtype are rejected in this mode.

```python
import polars as pl

from polars_learn import SchemaValidator

train = pl.DataFrame({"x": pl.Series([1], dtype=pl.Int64)})
incoming = pl.DataFrame({"x": pl.Series([2, None], dtype=pl.Int32)})
validator = SchemaValidator(mode="semi-strict").fit(train)
result = validator.transform(incoming)
assert result.schema["x"] == pl.Int64
assert result["x"].to_list() == [2, None]
```

Required columns must always be present. Extra columns fail unless `allow_extra_columns=True`; allowing extras does not make fitted columns optional. `extra_columns="drop"` removes allowed extras, while the default `"preserve"` keeps them. The drop setting alone does not allow extras.

`enforce_column_order=True` checks the relative order of required columns, so allowed extras can appear between them. With `enforce_column_order=False`, output puts fitted columns first, followed by preserved extras in their input order. `get_feature_names_out()` lists required fitted columns, excluding inference-only extras. Preserving extras can change the output width; drop them before an estimator that expects a fixed number of features.

Null values are preserved when the dtype is compatible. Nullability constraints are not learned. In permissive mode, a column with Null dtype can be cast to the fitted dtype. Failed strict casts raise; they do not silently replace invalid values with nulls. Choose the validation policy through `mode`: `"strict"`, `"semi-strict"`, or `"permissive"`. The default is `"strict"`.

## LazyFrames and Return Types

All five transformers accept a `pl.LazyFrame`, including a query from `pl.scan_csv(...)`. Default output follows the input to **each call**, independently of whether fitting used eager or lazy input. Pandas input, when conversion is allowed, is treated as an eager DataFrame.

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

Lazy operations check column names, order, dtypes, expression dependencies, and drops using schema metadata. Resolving metadata can access a source's metadata; it does not collect all its rows. Lazy fit validates the plan schema, not every input value. Value-dependent errors surface at collection. A returned LazyFrame is a query plan: collecting it again may execute the query again.

Inspect lazy output with `plan.collect_schema()` for names and dtypes, and `plan.collect()` for data. Verbosity reports lazy row counts as unknown rather than evaluating rows to count them.

## Pipelines with scikit-learn-style Estimators

The preprocessing steps use `fit` and `transform`; the final estimator uses `fit(X, y)` and its prediction method. Pass raw data to the complete pipeline for both fitting and prediction. `Pipeline` fits preprocessing on training input and, during `predict`, reapplies the fitted schema policies and feature expressions before invoking the model. The transformers ignore `y`; the model uses it to learn predictions.

A pipeline built from these transformers can remain lazy. Set the last native transformer to eager output before a downstream estimator that needs materialized data. `"polars-lazy"` is a package-specific output option; do not pass it to scikit-learn's standard transformers.

```python
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polars_learn import FeatureEngineer, FeatureSpec, SchemaValidator

X = pl.DataFrame({"price": [10.0, 20.0, 30.0], "quantity": [2, 3, 4]})
y = np.array([1.0, 2.0, 3.0])
pipeline = Pipeline(
    [
        ("schema", SchemaValidator(mode="semi-strict")),
        (
            "features",
            FeatureEngineer(
                [FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")]
            ).set_output(transform="polars"),
        ),
        ("scale", StandardScaler().set_output(transform="polars")),
        ("model", RandomForestRegressor(n_estimators=20, random_state=0)),
    ]
)
pipeline.fit(X.lazy(), y)
predictions = pipeline.predict(X.lazy())
assert predictions.shape == (3,)
```

Supported input is Polars DataFrames and LazyFrames, plus explicitly enabled pandas DataFrames. NumPy arrays, Series, Python lists, and sparse arrays are not accepted. These transformers bypass scikit-learn's automatic output wrapping and global output-container settings; configure each native transformer with `set_output`. There is no automatic conversion to NumPy. Request pandas compatibility output explicitly with `set_output(transform="pandas")`.

Cloning keeps constructor parameters and explicit output settings, and discards fitted state. `y` is accepted for estimator compatibility and ignored. Additional `fit_transform` keyword parameters, including `sample_weight`, are unsupported. Refit after changing parameters that affect the fitted contract; `set_params` does not reset fitted state. Invalid settings are rechecked when processing input.

### The Estimator Pattern

These transformers follow the same estimator conventions used by scikit-learn and libraries such as XGBoost:

| Operation          | Preprocessing transformer                                                     | Final predictive model                                     |
| ------------------ | ----------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `fit`              | Validate or learn the training input contract; return the fitted transformer. | Learn model parameters from training features and targets. |
| Apply fitted state | `transform(X)` returns prepared features.                                     | `predict(X)` returns predictions.                          |
| Fit and apply      | `fit_transform(X)` fits and returns prepared training features.               | Fit first, then call the model's prediction method.        |
| Configure          | Constructor parameters, `get_params`, and `set_params`.                       | The same parameter conventions for estimator-style models. |

In a `Pipeline`, each preprocessing step prepares the input for the next step, and the final estimator learns from that prepared input. Calling `predict` on the fitted pipeline reapplies the preprocessing steps before predicting. XGBoost is an example of a library with a [scikit-learn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html); its regression/classification models use `fit` and `predict`. The `transform` operation belongs to the preprocessing steps.

This shared interface describes how steps compose. The final estimator must also accept the prepared feature container and dtypes. Set the last native transformer to `set_output(transform="polars")` when the model needs eager data. If a model requires a different container, add an explicit conversion step at that boundary. Keep preprocessing fitted on training input and reuse it for validation and prediction.

## Saving and Loading a Fitted Pipeline

Save the complete fitted pipeline so the feature expressions, raw-input contract, fitted feature names, output settings, and model stay together. After loading it, keep passing raw observations to `predict`; you do not need to recreate the Polars expressions or prepare features separately.

`FeatureSpec`, `FeatureEngineer`, `ColumnSelector`, `FeaturePassthrough`, `SchemaSpec`, and `SchemaValidator` support Python pickling in fitted and unfitted forms where applicable. Ordinary `pl.Expr` objects, single-spec inputs, dict configs, and mixed specification sequences are preserved using Polars' native expression serialization. Fitted schemas and output-container choices survive loading. Pickling an unfitted transformer does not fit it.

```python
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from polars_learn import FeatureEngineer, FeatureSpec, SchemaValidator

raw_train = pl.DataFrame({"price": [10.0, 20.0, 30.0], "quantity": [2, 3, 4]})
y = np.array([1.0, 2.0, 3.0])
pipeline = Pipeline(
    [
        ("schema", SchemaValidator()),
        (
            "features",
            FeatureEngineer(
                FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
            ).set_output(transform="polars"),
        ),
        ("model", RandomForestRegressor(n_estimators=10, random_state=0)),
    ]
)
pipeline.fit(raw_train, y)

# Use your persistent model path in an application; this example cleans up its files.
with TemporaryDirectory() as directory:
    model_path = Path(directory) / "pipeline.pkl"
    with model_path.open("wb") as destination:
        pickle.dump(pipeline, destination, protocol=pickle.HIGHEST_PROTOCOL)
    with model_path.open("rb") as source:
        restored = pickle.load(source)

raw_predict = pl.DataFrame({"price": [15.0, 25.0], "quantity": [2, 4]})
np.testing.assert_array_equal(
    restored.predict(raw_predict), pipeline.predict(raw_predict)
)
assert restored[:-1].get_feature_names_out().tolist() == [
    "price",
    "quantity",
    "notional",
]
```

`joblib.dump`/`joblib.load` and `cloudpickle.dumps`/`cloudpickle.loads` also work with these objects. `cloudpickle` is a runtime dependency because Python callbacks embedded in Polars expressions need it for serialization. Tests include local lambda/closure callbacks and loading a complete fitted pipeline in a fresh interpreter. Prefer native Polars expressions when possible; custom callbacks must have serializable captured state and their required modules, plugins, and external resources available when they execute.

Saved objects record their creating versions of Python, `polars-learn`, Polars, scikit-learn, NumPy, and cloudpickle. Loading under a different version emits `polars_learn.InconsistentVersionWarning` for each changed dependency before restoring that object's state, then attempts loading. The warning exposes `class_name`, `dependency`, `original_version`, and `current_version`. Older artifacts without version metadata cannot report their creating environment.

Same-version expression loading uses Polars binary serialization to preserve native values, including NaN and infinity. Cross-version expression loading tries Polars' JSON representation, provided the expression operations, dtypes, callbacks, and other required objects are available and compatible in the receiving environment. If that representation is unavailable or rejected, loading raises `SerializationError` with version context and the original exception as its cause; it does not fall back to the old binary layout. CI verifies both directions between Polars 1.35.2 and the locked version, currently 1.44.2, for all six main classes and a fitted prediction pipeline.

To reject a version mismatch before restoring state, promote the warning to an error:

```python
import pickle
import warnings

import polars as pl

from polars_learn import FeatureSpec, InconsistentVersionWarning

saved = pickle.dumps(FeatureSpec(pl.col("price") * 2))
with warnings.catch_warnings():
    warnings.simplefilter("error", InconsistentVersionWarning)
    restored_spec = pickle.loads(saved)
```

If you accept a version mismatch, silence this package's version warnings around loading with the standard Python warning filter. The context restores your previous filters afterward and leaves other warning categories active. This works with pickle, joblib, and cloudpickle; incompatible state still raises `SerializationError`.

```python
import pickle
import warnings

import polars as pl

from polars_learn import FeatureSpec, InconsistentVersionWarning

saved = pickle.dumps(FeatureSpec(pl.col("price") * 2))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", InconsistentVersionWarning)
    restored_spec = pickle.loads(saved)
```

To silence this category application-wide, call `warnings.filterwarnings("ignore", category=InconsistentVersionWarning)` once in your application. Final estimators may issue their own version warnings, controlled by their own warning categories.

**Load only artifacts you trust.** Pickle-based loading can execute Python code, including through serialized expressions containing Python callbacks. Cross-version loading is a compatibility attempt, and a successful load does not guarantee identical numerical results. Validate transformed features and predictions after upgrading, and retain the original environment lockfile for recovery. Arbitrary future version transitions and unpicklable captured resources cannot be guaranteed portable. See [Polars expression persistence](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.deserialize.html) and [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).

## Verbosity

All six classes accept an integer `verbosity`:

| Level         | Output to standard output                                                          |
| ------------- | ---------------------------------------------------------------------------------- |
| `0` (default) | No progress diagnostics.                                                           |
| `1`           | Input dimensions and fit/transform progress; specification construction summaries. |
| `2`           | Level 1 plus schemas and feature operation details.                                |

```python
import polars as pl

from polars_learn import SchemaValidator

X = pl.DataFrame({"x": [1, 2]})
validator = SchemaValidator(verbosity=1)
validated = validator.fit_transform(X)
# [SchemaValidator] Validating input
# [SchemaValidator] Input: 2 rows, 1 columns
# [SchemaValidator] Fit complete; output columns: ['x']
# [SchemaValidator] Validating input
# [SchemaValidator] Input: 2 rows, 1 columns
# [SchemaValidator] Transform complete: 2 rows, 1 columns
```

`FeatureSpec` reports its construction separately. An engineer's verbosity does not change its specifications' verbosity. A dict config's `verbosity` applies when that config is converted into a specification, during construction and each fit/transform call. Frame summaries show metadata rather than printing data rows; feature details can include expression representations. Levels must be `0`, `1`, or `2`; booleans and strings are rejected. Verbosity does not suppress exceptions or pandas conversion warnings.

## Errors and When They Occur

| Situation                                                                               | Exception or warning                | Timing                                      |
| --------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------- |
| Invalid argument type or unsupported input                                              | `TypeError`                         | Construction or the next relevant operation |
| Invalid settings, conflicting policies, or duplicate configured names                   | `ValueError`                        | Construction or revalidation                |
| Transform or feature-name access before fitting                                         | `sklearn.exceptions.NotFittedError` | Method call                                 |
| `SchemaSpec` contract or fitted output-name mismatch                                    | `ValueError`                        | Fit/transform schema check                  |
| `ColumnSelector`/`FeaturePassthrough` missing columns or fitted output-name mismatch    | `ValueError`                        | Fit/transform schema check                  |
| `SchemaValidator` missing columns, disallowed extras/order/types, or eager cast failure | `SchemaValidationError`             | Transform                                   |
| Feature dependency, drop, or schema failure; eager feature execution failure            | `FeatureEngineeringError`           | Fit/transform                               |
| Value-dependent feature failure or permissive cast failure in a returned lazy plan      | Native Polars exception             | `collect()` or another execution operation  |
| Lossy semi-strict integer to Float64 cast in a lazy plan                                | `SchemaValidationError`             | Lazy execution                              |
| Enabled pandas conversion with warnings enabled                                         | `PandasConversionWarning`           | Conversion, even at verbosity 0             |

`PolarsLearnError` is the base of `SchemaValidationError` and `FeatureEngineeringError`. Plain `TypeError`/`ValueError`, `NotFittedError`, and native Polars errors do not inherit from it. Input query and source failures can also propagate as native Polars or source exceptions during schema resolution or collection. Wrapped errors preserve their original exception as `__cause__`.

To force value evaluation within `transform`, request `"polars"` output. `FeatureEngineer` wraps resulting execution errors in `FeatureEngineeringError`; `SchemaValidator` wraps Polars collection errors in `SchemaValidationError`. `SchemaSpec`, `ColumnSelector`, and `FeaturePassthrough` do not wrap source/collection errors.

## Optional Pandas Support

Install `polars-learn[pandas]` as described in [Installation](#installation), then explicitly allow conversion:

```python
import pandas as pd

from polars_learn import SchemaValidator

X = pd.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
validator = SchemaValidator(convert_pandas=True, warn_on_conversion=False)
result = validator.fit_transform(X)
assert result.columns == ["price", "quantity"]
```

All five transformers expose `convert_pandas` and `warn_on_conversion`. Conversion is disabled by default. When enabled, it emits `PandasConversionWarning` unless `warn_on_conversion=False`. Use Polars directly to avoid conversion overhead; `fit_transform` may convert input more than once. Pandas and PyArrow are imported only when needed. The extra includes PyArrow for nullable, extension, and other dtypes that require it. Conversion follows Polars conventions, including NaN to null conversion and ignoring the pandas index. Unsupported dtypes raise; conversion does not fall back to row operations. An explicit `"polars-lazy"` output choice returns a lazy wrapper after conversion.

## Development

Run from a repository checkout:

```bash
uv sync  # Includes pandas and PyArrow for the full test suite
uv run pytest --doctest-modules src tests
uv run ruff check .
uv run ruff format .
uv run ruff format --check .
uv run pyright
uv build
```

**Ruff is the formatter for all Python code and documentation examples.** CI checks `ruff format --check .`, linting, types, tests, and API doctests. Ruff's directory checks cover Python files, not Markdown code fences; format a Markdown example with `ruff format --stdin-filename example.py` before adding it. The test matrix covers Python 3.11 and 3.12 with all development dependencies, including pandas and PyArrow. Every job runs the full interoperability suite. A subprocess test separately verifies native usage with pandas and PyArrow blocked.

`uv.lock` pins development dependencies. Pyright includes its Node runtime via the development group. Pandas and PyArrow are required for development; missing pandas causes test collection to fail rather than silently skip coverage. Run `uv sync` after changing requirements. These remain optional runtime dependencies for users of the package. The package includes `py.typed`; transformer return annotations cover both DataFrame and LazyFrame because `set_output` can change the container. Narrow with `isinstance(result, pl.LazyFrame)` when your type checker needs the specific type. Library dependencies use minimum versions; the lockfile does not constrain downstream applications to those exact versions.

### Releases

Run **Actions → Create GitHub Release → Run workflow** to release the version in `pyproject.toml` from the selected branch or tag. The workflow validates the package, builds a fresh wheel and source archive, and creates a GitHub release named `vX.Y.Z` with those files attached. Once that workflow succeeds, **Upload Python Package** starts automatically and publishes the **same built files** to PyPI. Existing files in a local `dist/` directory are not used.

Before the first release:

1. Push both [release](.github/workflows/create-release.yml) and [publishing](.github/workflows/python-publish.yml) workflows to the repository's default branch. The automatic publisher uses GitHub's `workflow_run` event, which requires the publishing workflow on that branch.
2. Create the GitHub Actions environment `pypi` if it does not already exist.
3. Configure the `polars-learn` PyPI project's trusted publisher with your GitHub repository owner/name, workflow filename **`python-publish.yml`**, and environment **`pypi`**. See [PyPI's setup instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/). Publishing uses OIDC and does not read a `PYPI_API_TOKEN` secret.

For each new version:

1. Set `project.version` in `pyproject.toml` and `__version__` in `src/polars_learn/__init__.py` to the same new `X.Y.Z` version, then run `uv lock`.
2. Commit and push the release changes. The runner builds from the selected committed revision, so uncommitted local changes are not included.
3. Run **Create GitHub Release** from Actions, selecting the release branch or tag.
4. Check both workflow runs. The second run uses the `pypi` environment; any protection rules configured for that environment still apply.

The release workflow fails on mismatched versions, an existing `vX.Y.Z` tag, failed tests/checks, or invalid build metadata. The automatic publisher only accepts successful manual release-workflow runs from this repository and downloads artifacts from that exact run. Failed or cancelled release runs do not publish. It works with `GITHUB_TOKEN` without relying on its release event to start another workflow. See [GitHub's workflow_run documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run).

Publishing a GitHub release manually through the Releases page is also supported. Use a `vX.Y.Z` tag matching the code's package version. That path builds and validates the tagged source before uploading to PyPI.

If PyPI publishing fails after the GitHub release was created, fix the reported setup or upload problem and rerun **Upload Python Package**, rather than creating the same release tag again. Its automatic path reuses the original release-run artifacts, which must still be available. PyPI versions cannot be overwritten; use a new version for changed distributions.

## License

MIT. See [LICENSE](LICENSE).

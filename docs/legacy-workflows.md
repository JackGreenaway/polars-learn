# Additional workflows and release operations

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

Install `polars-learn[pandas]` as described in [Installation](getting-started.md), then explicitly allow conversion:

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

1. Push both [release](https://github.com/JackGreenaway/polars-learn/blob/main/.github/workflows/create-release.yml) and [publishing](https://github.com/JackGreenaway/polars-learn/blob/main/.github/workflows/python-publish.yml) workflows to the repository's default branch. The automatic publisher uses GitHub's `workflow_run` event, which requires the publishing workflow on that branch.
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

MIT. See [LICENSE](https://github.com/JackGreenaway/polars-learn/blob/main/LICENSE).

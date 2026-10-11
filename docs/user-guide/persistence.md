# Saving and loading

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

## Artifact lifecycle

Save the complete fitted preprocessing-and-model pipeline, its environment lockfile, and an input/output fixture that can verify features and predictions after loading. Choose version-warning filters in the loading application rather than persisting a global suppression policy in your model. See [production guidance](../production.md) for rollout checks.

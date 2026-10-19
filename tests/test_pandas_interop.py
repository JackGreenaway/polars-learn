import importlib
import pickle
import subprocess
import sys

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polars_learn import (
    ColumnSelector,
    FeatureEngineer,
    FeaturePassthrough,
    FeatureSpec,
    PandasConversionWarning,
    SchemaSpec,
    SchemaValidator,
    ValueValidator,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: FeaturePassthrough(["price", "quantity"], **kw),
        lambda **kw: FeatureEngineer([], **kw),
        SchemaValidator,
        lambda **kw: ValueValidator({"price_positive": pl.col("price") > 0}, **kw),
    ],
)
def test_conversion_options(frame, factory):
    incoming = pd.DataFrame(frame.to_dict(as_series=False))
    with pytest.raises(TypeError, match="convert_pandas"):
        factory().fit(incoming)
    est = factory(convert_pandas=True)
    with pytest.warns(PandasConversionWarning, match="overhead"):
        est.fit(incoming)
    with pytest.warns(PandasConversionWarning):
        assert_frame_equal(est.transform(incoming), frame)
    quiet = factory(convert_pandas=True, warn_on_conversion=False)
    assert_frame_equal(quiet.fit_transform(incoming), frame)


def test_nullable_pandas():
    incoming = pd.DataFrame(
        {
            "i": pd.Series([1, None], dtype="Int64"),
            "s": pd.Series(["a", None], dtype="string"),
        }
    )
    est = SchemaValidator(convert_pandas=True, warn_on_conversion=False)
    out = est.fit_transform(incoming)
    assert out.schema == {"i": pl.Int64, "s": pl.String}
    assert out.to_dict(as_series=False) == {"i": [1, None], "s": ["a", None]}


def test_pandas_unavailable(monkeypatch):
    class FakeFrame:
        pass

    FakeFrame.__module__ = "pandas.core.frame"
    real_import = importlib.import_module

    def guarded(name):
        if name == "pandas":
            raise ImportError("pandas unavailable")
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", guarded)
    with pytest.raises(TypeError, match="extra"):
        SchemaValidator(convert_pandas=True).fit(FakeFrame())


def test_native_import_never_requires_pandas():
    code = """
import sys
import importlib.util
real_find_spec = importlib.util.find_spec
def find_spec(name, *args, **kwargs):
    if name.split('.')[0] in ('pandas', 'pyarrow'):
        return None
    return real_find_spec(name, *args, **kwargs)
importlib.util.find_spec = find_spec
class BlockPandas:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pandas', 'pyarrow'):
            raise ModuleNotFoundError('optional dependency blocked')
sys.meta_path.insert(0, BlockPandas())
import polars as pl
from polars_learn import FeatureEngineer, SchemaValidator
frame = pl.DataFrame({'x': [1, 2]})
assert FeatureEngineer([]).fit_transform(frame).equals(frame)
assert SchemaValidator().fit_transform(frame).equals(frame)
assert 'pandas' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "factory",
    [
        lambda: FeaturePassthrough(["quantity", "price"]),
        lambda: ColumnSelector(["quantity", "price"], rename={"price": "value"}),
        lambda: FeatureEngineer(
            FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
        ),
        lambda: SchemaSpec({"price": pl.Float64, "quantity": pl.Int64}),
        SchemaValidator,
        lambda: ValueValidator({"price_positive": pl.col("price") > 0}),
    ],
)
def test_pandas_output_all_transformers(frame, factory, lazy):
    incoming = frame.lazy() if lazy else frame
    native = factory().fit_transform(incoming)
    expected = (native.collect() if lazy else native).to_pandas()
    estimator = factory().set_output(transform="pandas")
    assert estimator.fit(incoming) is estimator
    pd.testing.assert_frame_equal(estimator.transform(incoming), expected)
    pd.testing.assert_frame_equal(estimator.fit_transform(incoming), expected)
    pd.testing.assert_frame_equal(clone(estimator).fit_transform(incoming), expected)
    restored = pickle.loads(pickle.dumps(estimator))
    pd.testing.assert_frame_equal(restored.transform(incoming), expected)
    estimator.set_output(transform=None)
    assert isinstance(estimator.transform(incoming), pd.DataFrame)
    estimator.set_output(transform="default")
    assert isinstance(estimator.transform(incoming), type(incoming))


def test_pandas_output_nullable_values_and_index():
    incoming = pd.DataFrame(
        {"i": pd.Series([1, None], dtype="Int64"), "s": ["a", None]}
    )
    incoming.index = [10, 20]
    est = SchemaValidator(convert_pandas=True, warn_on_conversion=False)
    result = est.set_output(transform="pandas").fit_transform(incoming)
    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(pd.RangeIndex(2))
    assert result["i"].iloc[0] == 1
    assert pd.isna(result["i"].iloc[1])
    assert result["s"].iloc[0] == "a"
    assert pd.isna(result["s"].iloc[1])


@pytest.mark.parametrize("dependency", ["pandas", "pyarrow"])
def test_pandas_output_missing_dependency(frame, monkeypatch, dependency):
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] == dependency:
            raise ImportError(f"{dependency} unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    estimator = SchemaValidator().set_output(transform="pandas").fit(frame)
    with pytest.raises(ImportError, match=r"install polars-learn\[pandas\]"):
        estimator.transform(frame)


def test_pandas_output_at_pipeline_boundary(frame):
    pipeline = Pipeline(
        [
            ("schema", SchemaValidator()),
            ("features", FeatureEngineer([]).set_output(transform="pandas")),
            ("scale", StandardScaler().set_output(transform="pandas")),
        ]
    )
    result = pipeline.fit_transform(frame.lazy())
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == frame.columns
    assert pipeline.get_feature_names_out().tolist() == frame.columns

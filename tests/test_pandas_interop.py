import importlib
import subprocess
import sys

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_learn import (
    FeatureEngineer,
    FeaturePassthrough,
    PandasConversionWarning,
    SchemaValidator,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: FeaturePassthrough(["price", "quantity"], **kw),
        lambda **kw: FeatureEngineer([], **kw),
        SchemaValidator,
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

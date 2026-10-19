"""Persistence must preserve executable expressions and fitted pipeline behavior."""

import io
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime

import cloudpickle
import joblib
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from polars_learn import (
    ColumnSelector,
    FeatureEngineer,
    FeatureEngineeringError,
    FeaturePassthrough,
    FeatureSpec,
    PandasConversionWarning,
    PolarsLearnError,
    SchemaSpec,
    SchemaValidationError,
    SchemaValidator,
    ValueValidationError,
    ValueValidator,
)

PROTOCOLS = range(pickle.HIGHEST_PROTOCOL + 1)
TRANSFORMERS = [
    lambda: ColumnSelector(["quantity", "price"], rename={"quantity": "qty"}),
    lambda: FeaturePassthrough(["quantity", "price"]),
    lambda: SchemaSpec({"price": pl.Float64, "quantity": pl.Int64}),
    lambda: SchemaValidator(mode="permissive"),
    lambda: ValueValidator(
        {
            "positive_price": pl.col("price") > 0,
            "quantity_present": pl.col("quantity").is_not_null(),
        }
    ),
    lambda: FeatureEngineer(
        [
            FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
            {
                "expression": pl.col("notional") + 1,
                "alias": "adjusted",
                "drop": ["quantity"],
            },
        ]
    ),
]


@pytest.mark.parametrize("factory", TRANSFORMERS)
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("fit_lazy", [False, True])
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_transformer_pickle_round_trip(frame, factory, fitted, fit_lazy, protocol):
    original = factory().set_output(transform="polars-lazy")
    if fitted:
        original.fit(frame.lazy() if fit_lazy else frame)
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is type(original)
    assert restored.verbosity == original.verbosity
    assert restored.convert_pandas == original.convert_pandas
    if not fitted:
        with pytest.raises(NotFittedError):
            restored.transform(frame)
        expected = original.fit_transform(frame)
        result = restored.fit_transform(frame)
    else:
        expected = original.transform(frame)
        result = restored.transform(frame)
        np.testing.assert_array_equal(
            restored.feature_names_in_, original.feature_names_in_
        )
        assert restored.n_features_in_ == original.n_features_in_
    assert isinstance(result, pl.LazyFrame)
    assert_frame_equal(result.collect(), expected.collect())
    np.testing.assert_array_equal(
        restored.get_feature_names_out(), original.get_feature_names_out()
    )
    # Native-default output must still track the next input's kind after loading.
    restored.set_output(transform="default")
    assert isinstance(restored.transform(frame), pl.DataFrame)
    assert isinstance(restored.transform(frame.lazy()), pl.LazyFrame)
    assert not hasattr(clone(restored), "output_features_")


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "spec",
    [
        FeatureSpec(
            pl.col("price") * pl.col("quantity"), alias="notional", name="product"
        ),
        FeatureSpec(
            expression=[pl.col("price") + 1, pl.col("quantity") * 2], drop=["quantity"]
        ),
        FeatureSpec(pl.all().name.suffix("_copy")),
        FeatureSpec(
            pl.when(pl.col("price") > 2).then(pl.lit("high")).otherwise(pl.lit("low")),
            alias="band",
        ),
        FeatureSpec(pl.lit(pl.Series("offsets", [1, 2, 3, 4])), alias="offsets"),
    ],
)
def test_feature_spec_pickle_preserves_native_expression_execution(
    frame, spec, protocol
):
    restored = pickle.loads(pickle.dumps(spec, protocol=protocol))
    assert restored.drop == spec.drop
    assert restored.name == spec.name
    assert restored.verbosity == spec.verbosity
    assert all(isinstance(expr, pl.Expr) for expr in restored.expression)
    assert_frame_equal(
        FeatureEngineer(restored).fit_transform(frame),
        FeatureEngineer(spec).fit_transform(frame),
    )


def callback_spec(offset, kind="batches"):
    if kind == "elements":

        def adjust(value):
            lookup = {1.0: 2.0, 2.0: 4.0, 3.0: 6.0, 4.0: 8.0, 5.0: 10.0}
            return lookup[value] + offset

        return FeatureSpec(
            pl.col("price").map_elements(adjust, return_dtype=pl.Float64),
            alias="adjusted",
            name="element_callback",
        )

    # A local closure cannot be recovered by importing a module-level function.
    return FeatureSpec(
        pl.col("price").map_batches(
            lambda series: series * 2 + offset,
            return_dtype=pl.Float64,
            is_elementwise=True,
        ),
        alias="adjusted",
        name="captured_offset",
    )


@pytest.mark.parametrize("kind", ["batches", "elements"])
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_pickle_python_callback_closures(frame, protocol, kind):
    spec = callback_spec(3.0, kind)
    configured = {"expression": spec.expression[0], "name": spec.name}
    for original in (spec, FeatureEngineer(spec), FeatureEngineer(configured)):
        restored = pickle.loads(pickle.dumps(original, protocol=protocol))
        engineer = (
            FeatureEngineer(restored) if isinstance(restored, FeatureSpec) else restored
        )
        result = engineer.fit_transform(frame)
        assert result["adjusted"].to_list() == [5.0, 7.0, 9.0, 11.0]


@pytest.mark.parametrize("serializer", [pickle, cloudpickle, joblib])
def test_nested_schema_and_pandas_settings_round_trip(serializer):
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series([datetime(2026, 1, 1)], dtype=pl.Datetime("ms")),
            "values": pl.Series([[1, 2]], dtype=pl.List(pl.Int64)),
            "state": pl.Series(["active"], dtype=pl.Enum(["active", "inactive"])),
        }
    )
    original = SchemaSpec(
        dict(frame.schema), convert_pandas=True, warn_on_conversion=False
    ).fit(frame)
    if serializer is joblib:
        buffer = io.BytesIO()
        serializer.dump(original, buffer)
        buffer.seek(0)
        restored = serializer.load(buffer)
    else:
        restored = serializer.loads(serializer.dumps(original))
    assert restored.schema_ == original.schema_
    assert restored.convert_pandas is True
    assert restored.warn_on_conversion is False
    assert_frame_equal(restored.transform(frame), frame)


@pytest.mark.parametrize("kind", ["batches", "elements"])
@pytest.mark.parametrize("serializer", ["pickle", "cloudpickle", "joblib"])
def test_complete_pipeline_loads_in_a_fresh_process(frame, tmp_path, serializer, kind):
    names = ["price", "notional", "adjusted"]
    pipeline = Pipeline(
        [
            ("contract", SchemaSpec(dict(frame.schema))),
            ("schema", SchemaValidator(mode="semi-strict")),
            (
                "features",
                FeatureEngineer(
                    [
                        FeatureSpec(
                            pl.col("price") * pl.col("quantity"), alias="notional"
                        ),
                        callback_spec(3.0, kind),
                    ]
                ),
            ),
            ("select", ColumnSelector(names)),
            ("unchanged", FeaturePassthrough(names).set_output(transform="polars")),
            ("model", RandomForestRegressor(n_estimators=5, random_state=0, n_jobs=1)),
        ]
    )
    pipeline.fit(frame.lazy(), np.array([1.0, 2.0, 3.0, 4.0]))
    incoming = pl.DataFrame({"price": [4.0, 5.0], "quantity": [2, 3]})
    path = tmp_path / "pipeline.pkl"
    if serializer == "joblib":
        joblib.dump(pipeline, path, compress=3)
    else:
        module = pickle if serializer == "pickle" else cloudpickle
        path.write_bytes(module.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL))
    expected = pipeline.predict(incoming.lazy()).tolist()
    script = """
import json
import pickle
import sys
import joblib
import numpy as np
import polars as pl
from sklearn.base import clone

path, serializer, expected = sys.argv[1:]
if serializer == "joblib":
    restored = joblib.load(path)
else:
    # cloudpickle output is also readable with ordinary pickle.load.
    with open(path, "rb") as source:
        restored = pickle.load(source)
incoming = pl.DataFrame({"price": [4.0, 5.0], "quantity": [2, 3]})
np.testing.assert_array_equal(
    restored.predict(incoming.lazy()), np.array(json.loads(expected))
)
prepared = restored[:-1].transform(incoming.lazy())
assert prepared.columns == ["price", "notional", "adjusted"]
assert prepared["adjusted"].to_list() == [11.0, 13.0]
assert not hasattr(clone(restored)["features"], "output_schema_")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), serializer, json.dumps(expected)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "POLARS_MAX_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "error_type",
    [
        PolarsLearnError,
        SchemaValidationError,
        FeatureEngineeringError,
        PandasConversionWarning,
        ValueValidationError,
    ],
)
def test_public_errors_and_warning_pickle(error_type):
    original = error_type("example message")
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is error_type
    assert restored.args == original.args


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_pandas_conversion_and_warnings_settings_survive_pickle(frame, protocol):
    incoming = frame.to_pandas()
    original = SchemaValidator(convert_pandas=True, warn_on_conversion=False)
    expected = original.fit_transform(incoming)
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert_frame_equal(restored.transform(incoming), expected)


def test_pickle_fitted_scan_does_not_require_the_training_source(frame, tmp_path):
    source = tmp_path / "training.csv"
    frame.write_csv(source)
    original = FeatureEngineer(FeatureSpec(pl.col("price") * 2, alias="doubled"))
    original.fit(pl.scan_csv(source))
    payload = pickle.dumps(original)
    source.unlink()
    restored = pickle.loads(payload)
    assert restored.transform(frame.lazy()).collect()["doubled"].to_list() == [
        2.0,
        4.0,
        6.0,
        8.0,
    ]


@pytest.mark.parametrize(
    "factory", [*TRANSFORMERS, lambda: FeatureSpec(pl.col("price") + 1)]
)
def test_version_warning_for_each_class(frame, factory, monkeypatch):
    import warnings

    from polars_learn import InconsistentVersionWarning, _serialization

    original = factory()
    if not isinstance(original, FeatureSpec):
        original.fit(frame)
    creating = _serialization._versions()
    monkeypatch.setattr(
        _serialization, "_versions", lambda: {**creating, "polars": "older"}
    )
    payload = pickle.dumps(original)
    monkeypatch.setattr(_serialization, "_versions", lambda: creating)
    with pytest.warns(InconsistentVersionWarning) as captured:
        restored = pickle.loads(payload)
    matching = [
        item.message
        for item in captured
        if item.message.class_name == type(original).__name__
    ]
    assert matching
    assert matching[0].dependency == "polars"
    assert matching[0].original_version == "older"
    assert matching[0].current_version == pl.__version__
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        with pytest.raises(InconsistentVersionWarning):
            pickle.loads(payload)
    if isinstance(restored, FeatureSpec):
        assert_frame_equal(
            FeatureEngineer(restored).fit_transform(frame),
            FeatureEngineer(original).fit_transform(frame),
        )
    else:
        assert_frame_equal(restored.transform(frame), original.transform(frame))


def test_warning_can_stop_loading_before_expression_deserialization(monkeypatch):
    import warnings

    from polars_learn import InconsistentVersionWarning, _serialization

    original = FeatureEngineer({"expression": pl.col("price") + 1})
    creating = _serialization._versions()
    monkeypatch.setattr(
        _serialization, "_versions", lambda: {**creating, "polars": "older"}
    )
    payload = pickle.dumps(original)
    monkeypatch.setattr(_serialization, "_versions", lambda: creating)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Expression deserialized before its version warning was handled")

    monkeypatch.setattr(pl.Expr, "deserialize", unexpected_decode)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        with pytest.raises(InconsistentVersionWarning):
            pickle.loads(payload)


def test_expression_restore_failure_has_versions_and_original_cause(monkeypatch):
    from polars_learn import SerializationError

    payload = pickle.dumps(FeatureSpec(pl.col("price") + 1))

    def unsupported(*args, **kwargs):
        raise pl.exceptions.InvalidOperationError("expression is unavailable")

    monkeypatch.setattr(pl.Expr, "deserialize", unsupported)
    with pytest.raises(
        SerializationError, match=f"saved with Polars {pl.__version__}"
    ) as captured:
        pickle.loads(payload)
    assert isinstance(captured.value.__cause__, pl.exceptions.InvalidOperationError)
    assert "expression is unavailable" in str(captured.value)


def test_unpicklable_callback_state_fails_with_class_context():
    import threading

    from polars_learn import SerializationError

    lock = threading.Lock()
    spec = FeatureSpec(
        pl.col("price").map_batches(
            lambda series: series if lock.locked() else series + 1,
            return_dtype=pl.Float64,
        )
    )
    with pytest.raises(
        SerializationError, match="Cannot serialize FeatureSpec"
    ) as captured:
        pickle.dumps(spec)
    assert captured.value.__cause__ is not None


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_object_self_references_and_warning_state_are_preserved(protocol):
    from polars_learn import InconsistentVersionWarning

    original = FeatureEngineer(FeatureSpec(pl.col("price") + 1))
    original.reference = original
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert restored.reference is restored
    warning = InconsistentVersionWarning("FeatureSpec", "polars", "1.x", "1.y")
    restored_warning = pickle.loads(pickle.dumps(warning, protocol=protocol))
    assert str(restored_warning) == str(warning)
    assert restored_warning.original_version == "1.x"
    assert restored_warning.current_version == "1.y"


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), None, 2**63 - 1]
)
def test_expression_literals_are_lossless_in_the_creating_version(value):
    original = FeatureSpec(pl.lit(value), alias="literal")
    restored = pickle.loads(pickle.dumps(original))
    frame = pl.DataFrame({"x": [1, 2]})
    assert_frame_equal(
        FeatureEngineer(restored).fit_transform(frame),
        FeatureEngineer(original).fit_transform(frame),
    )


def test_cross_version_expression_failure_does_not_try_binary(monkeypatch):
    from polars_learn import SerializationError, _serialization

    formats = []

    def unsupported(source, *, format):
        formats.append(format)
        raise pl.exceptions.InvalidOperationError("unsupported JSON expression")

    monkeypatch.setattr(pl.Expr, "deserialize", unsupported)
    with pytest.raises(SerializationError, match="saved with Polars older"):
        _serialization._restore_expression(b"old binary", "{}", "older")
    assert formats == ["json"]
    with pytest.raises(SerializationError, match="no JSON representation"):
        _serialization._restore_expression(b"old binary", None, "older")
    assert formats == ["json"]


@pytest.mark.parametrize("serializer", [pickle, cloudpickle])
def test_version_warnings_can_be_silenced_without_hiding_other_warnings(
    monkeypatch, frame, serializer
):
    import warnings

    from polars_learn import InconsistentVersionWarning, _serialization

    original = FeatureEngineer(FeatureSpec(pl.col("price") + 1)).fit(frame)
    creating = _serialization._versions()
    monkeypatch.setattr(
        _serialization, "_versions", lambda: {**creating, "polars": "older"}
    )
    payload = serializer.dumps(original)
    monkeypatch.setattr(_serialization, "_versions", lambda: creating)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            restored = serializer.loads(payload)
            warnings.warn("unrelated warning", UserWarning, stacklevel=2)
        assert [str(item.message) for item in captured] == ["unrelated warning"]
        serializer.loads(payload)
        assert any(
            isinstance(item.message, InconsistentVersionWarning) for item in captured
        )
    assert_frame_equal(restored.transform(frame), original.transform(frame))


@pytest.mark.parametrize("lazy", [False, True])
def test_map_elements_nulls_dependencies_and_pickle(lazy):
    def score(code):
        return {"standard": 1.0, "priority": 3.0}.get(code.strip().lower(), 0.0)

    raw = pl.DataFrame(
        {"service": [" Priority ", "standard", None], "quantity": [2, 3, 1]}
    )
    incoming = raw.lazy() if lazy else raw
    engineer = FeatureEngineer(
        [
            FeatureSpec(
                pl.col("service").map_elements(
                    score, return_dtype=pl.Float64, skip_nulls=True
                ),
                alias="score",
                drop=["service"],
            ),
            FeatureSpec(pl.col("score") * pl.col("quantity"), alias="weighted"),
        ]
    ).fit(incoming)
    restored = pickle.loads(pickle.dumps(engineer))
    output = restored.transform(incoming)
    result = output.collect() if lazy else output
    assert result["weighted"].to_list() == [6.0, 3.0, None]
    assert result.schema["score"] == pl.Float64
    assert result.columns == ["quantity", "score", "weighted"]

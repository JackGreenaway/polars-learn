import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from polars_learn import FeatureEngineer, FeatureEngineeringError, FeatureSpec


def engineer():
    return FeatureEngineer(
        [
            FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
            FeatureSpec(
                pl.col("notional").log1p(), alias="log_notional", drop=["quantity"]
            ),
        ]
    )


def test_fit_transform_names_and_state(frame):
    est = engineer()
    assert est.fit(frame) is est
    assert est.n_features_in_ == 2
    assert est.feature_names_in_.tolist() == frame.columns
    assert est.get_feature_names_out().tolist() == ["price", "notional", "log_notional"]
    assert est.get_feature_names_out().dtype == object
    assert_frame_equal(est.transform(frame), engineer().fit_transform(frame))
    assert est.transform(frame)["notional"].to_list() == [2, 6, 12, 20]
    assert (
        est.get_feature_names_out(frame.columns).tolist()
        == est.output_features_.tolist()
    )
    with pytest.raises(ValueError, match="input_features"):
        est.get_feature_names_out(["bad"])
    names = est.get_feature_names_out()
    names[0] = "changed"
    assert est.get_feature_names_out()[0] == "price"


def test_not_fitted_and_unsupported_input(frame):
    with pytest.raises(NotFittedError):
        engineer().transform(frame)
    with pytest.raises(NotFittedError):
        engineer().get_feature_names_out()
    for value in [np.ones((2, 2)), None]:
        with pytest.raises(TypeError, match="Polars DataFrame"):
            engineer().fit(value)


@pytest.mark.parametrize(
    "frame_change",
    [
        lambda df: df.drop("quantity"),
        lambda df: df.select("quantity", "price"),
        lambda df: df.with_columns(pl.lit(1).alias("extra")),
    ],
)
def test_input_names_checked(frame, frame_change):
    with pytest.raises(FeatureEngineeringError, match="Input columns"):
        engineer().fit(frame).transform(frame_change(frame))


@pytest.mark.parametrize(
    "features",
    [
        [FeatureSpec(pl.col("missing"), name="missing_input")],
        [FeatureSpec(pl.col("price"), drop=["absent"])],
        [FeatureSpec(pl.col("price"), drop=["price"]), FeatureSpec(pl.col("price"))],
        [FeatureSpec(pl.col("price").str.to_uppercase())],
        [FeatureSpec(expressions=[pl.all(), pl.col("price")])],
    ],
)
def test_execution_errors_chained(frame, features):
    with pytest.raises(FeatureEngineeringError) as caught:
        FeatureEngineer(features).fit(frame)
    assert caught.value.__cause__ is not None


def test_expressions_in_one_spec_are_parallel(frame):
    est = FeatureEngineer(
        [
            FeatureSpec(
                expressions=[pl.col("price").alias("new"), pl.col("new").alias("later")]
            )
        ]
    )
    with pytest.raises(FeatureEngineeringError):
        est.fit(frame)


def test_output_schema_inconsistency(frame):
    est = FeatureEngineer([]).fit(frame)
    changed = frame.with_columns(pl.col("quantity").cast(pl.Int32))
    with pytest.raises(FeatureEngineeringError, match="Output schema"):
        est.transform(changed)


def test_empty_clone_and_params(frame):
    features = []
    est = FeatureEngineer(features)
    assert est.features is features
    assert est.get_params()["features"] is features
    assert_frame_equal(est.fit_transform(frame), frame)
    copied = clone(est)
    assert not hasattr(copied, "output_schema_")
    assert copied.set_params(features=engineer().features) is copied
    assert copied.fit_transform(frame).width == 3


def test_clone_expressions(frame):
    est = clone(engineer().fit(frame))
    assert not hasattr(est, "output_schema_")
    assert_frame_equal(est.fit_transform(frame), engineer().fit_transform(frame))


@pytest.mark.parametrize("features", [None, "bad", [1]])
def test_invalid_features(frame, features):
    with pytest.raises(TypeError):
        FeatureEngineer(features).fit(frame)


def test_runtime_failure(frame):
    est = FeatureEngineer([FeatureSpec(pl.col("price").cast(pl.Int8), name="small")])
    est.fit(frame)
    with pytest.raises(FeatureEngineeringError, match="small") as caught:
        est.transform(frame.with_columns(pl.lit(1000.0).alias("price")))
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("lazy", [False, True])
def test_single_spec_matches_sequence_and_preserves_parameter(frame, lazy):
    spec = FeatureSpec(
        expressions=[
            (pl.col("price") * pl.col("quantity")).alias("notional"),
            (pl.col("price") + 1).alias("incremented"),
        ],
        drop=["quantity"],
        name="quote_features",
    )
    incoming = frame.lazy() if lazy else frame
    single = FeatureEngineer(spec)
    sequence = FeatureEngineer([spec])
    assert single.features is spec
    assert single.get_params()["features"] is spec
    result = single.fit_transform(incoming)
    expected = sequence.fit_transform(incoming)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    assert_frame_equal(
        result.collect() if lazy else result,
        expected.collect() if lazy else expected,
    )
    assert single.output_schema_ == sequence.output_schema_
    assert (
        single.get_feature_names_out().tolist()
        == sequence.get_feature_names_out().tolist()
    )
    assert single.features is spec


def test_single_spec_clone_and_set_params(frame):
    spec = FeatureSpec(pl.col("price") + 1, alias="incremented")
    fitted = FeatureEngineer(spec).set_output(transform="polars-lazy").fit(frame)
    copied = clone(fitted)
    assert isinstance(copied.features, FeatureSpec)
    assert not hasattr(copied, "output_schema_")
    result = copied.fit_transform(frame)
    assert isinstance(result, pl.LazyFrame)
    assert result.collect()["incremented"].to_list() == [2.0, 3.0, 4.0, 5.0]
    later = FeatureSpec(pl.col("price") * 2, alias="doubled")
    copied.set_params(features=[spec, later])
    assert copied.fit_transform(frame).collect().columns[-2:] == [
        "incremented",
        "doubled",
    ]
    copied.set_params(features=later)
    assert copied.get_params()["features"] is later
    assert copied.fit_transform(frame).collect().columns[-1] == "doubled"
    copied.set_params(features="bad")
    with pytest.raises(TypeError, match="features must be a FeatureSpec"):
        copied.fit(object())


def test_single_spec_failure_retains_context(frame):
    spec = FeatureSpec(pl.col("missing"), name="single_missing_input")
    with pytest.raises(FeatureEngineeringError, match="single_missing_input") as caught:
        FeatureEngineer(spec).fit(frame.lazy())
    assert isinstance(caught.value.__cause__, pl.exceptions.ColumnNotFoundError)


@pytest.mark.parametrize(
    "features", [pl.col("price"), iter([]), [[FeatureSpec(pl.col("price"))]]]
)
def test_single_spec_support_does_not_accept_other_containers(features):
    with pytest.raises(TypeError, match="features"):
        FeatureEngineer(features)


@pytest.mark.parametrize("lazy", [False, True])
def test_dict_spec_matches_explicit_spec_without_mutating_config(frame, lazy):
    config = {
        "expression": pl.col("price") * pl.col("quantity"),
        "alias": "notional",
        "drop": ["quantity"],
        "name": "quote_features",
    }
    original_expression = config["expression"]
    original_drop = config["drop"]
    incoming = frame.lazy() if lazy else frame
    configured = FeatureEngineer(config)
    result = configured.fit_transform(incoming)
    expected = FeatureEngineer(FeatureSpec(**config)).fit_transform(incoming)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    assert_frame_equal(
        result.collect() if lazy else result,
        expected.collect() if lazy else expected,
    )
    assert configured.features is config
    assert configured.get_params()["features"] is config
    assert config["expression"] is original_expression
    assert original_expression.meta.output_name() == "price"
    assert config["drop"] is original_drop
    assert original_drop == ["quantity"]


def test_mixed_specs_and_dicts_preserve_dependencies_and_clone(frame):
    features = (
        {"expression": pl.col("price") * 2, "alias": "doubled"},
        FeatureSpec(pl.col("doubled") + 1, alias="incremented"),
        {
            "expression": pl.col("incremented") * 2,
            "alias": "final",
            "drop": ["doubled", "incremented"],
        },
    )
    original = FeatureEngineer(features).set_output(transform="polars-lazy")
    original.fit(frame)
    copied = clone(original)
    assert isinstance(copied.features, tuple)
    assert isinstance(copied.features[0], dict)
    assert isinstance(copied.features[1], FeatureSpec)
    assert not hasattr(copied, "output_schema_")
    result = copied.fit_transform(frame)
    assert isinstance(result, pl.LazyFrame)
    assert result.collect()["final"].to_list() == [6.0, 10.0, 14.0, 18.0]
    assert copied.get_feature_names_out().tolist() == ["price", "quantity", "final"]


def test_dict_spec_supports_multiple_expressions_and_revalidation(frame):
    engineer = FeatureEngineer(
        {
            "expressions": [pl.col("price").alias("a"), pl.col("quantity").alias("b")],
            "drop": ["price", "quantity"],
        }
    )
    assert engineer.fit_transform(frame).columns == ["a", "b"]
    config = {"expression": pl.col("price") + 1, "alias": "incremented"}
    engineer.set_params(features=config)
    assert engineer.fit_transform(frame)["incremented"].to_list() == [
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    config["unknown"] = True
    with pytest.raises(TypeError, match="features: invalid FeatureSpec config"):
        engineer.fit(object())


@pytest.mark.parametrize(
    "config,error",
    [
        ({}, ValueError),
        ({"expression": pl.col("price"), "unknown": True}, TypeError),
        ({"expression": "price"}, TypeError),
        (
            {"expression": pl.col("price"), "expressions": [pl.col("quantity")]},
            ValueError,
        ),
        ({"expressions": []}, ValueError),
        ({"expression": pl.col("price"), "drop": "price"}, TypeError),
        ({"expression": pl.col("price"), "verbosity": True}, TypeError),
    ],
)
def test_bad_dict_config_fails_at_construction_with_context(config, error):
    with pytest.raises(error, match="features: invalid FeatureSpec config") as caught:
        FeatureEngineer(config)
    assert isinstance(caught.value.__cause__, error)
    with pytest.raises(error, match=r"features\[1\]: invalid FeatureSpec config"):
        FeatureEngineer([FeatureSpec(pl.col("price")), config])


def test_dict_spec_execution_error_keeps_configured_label(frame):
    with pytest.raises(FeatureEngineeringError, match="configured_missing") as caught:
        FeatureEngineer(
            {"expression": pl.col("missing"), "name": "configured_missing"}
        ).fit(frame.lazy())
    assert isinstance(caught.value.__cause__, pl.exceptions.ColumnNotFoundError)

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
    for value in [np.ones((2, 2)), frame.lazy(), None]:
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

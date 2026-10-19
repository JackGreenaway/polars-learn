import numpy as np
import polars as pl
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import estimator_checks_generator

from polars_learn import (
    FeatureEngineer,
    FeaturePassthrough,
    FeatureSpec,
    SchemaValidator,
    ValueValidator,
)


def preprocessing():
    return Pipeline(
        [
            ("schema", SchemaValidator(mode="semi-strict")),
            (
                "features",
                FeatureEngineer(
                    [
                        FeatureSpec(
                            pl.col("price") * pl.col("quantity"), alias="notional"
                        ),
                        FeatureSpec(
                            pl.col("notional").log1p(),
                            alias="log_notional",
                            drop=["quantity"],
                        ),
                    ]
                ),
            ),
        ]
    )


def test_real_pipeline(frame):
    prep = preprocessing()
    model = Pipeline(
        [
            ("preprocessing", prep),
            ("scale", StandardScaler().set_output(transform="polars")),
            ("model", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]
    )
    y = np.array([1, 2, 3, 4])
    model.fit(frame, y)
    assert isinstance(prep.named_steps["schema"].transform(frame), pl.DataFrame)
    assert isinstance(prep.transform(frame), pl.DataFrame)
    assert isinstance(model[:-1].transform(frame), pl.DataFrame)
    assert model.predict(frame).shape == (4,)
    assert prep.get_feature_names_out().tolist() == [
        "price",
        "notional",
        "log_notional",
    ]
    assert (
        model[:-1].get_feature_names_out().tolist()
        == prep.get_feature_names_out().tolist()
    )
    copied = clone(model)
    assert not hasattr(copied["preprocessing"]["schema"], "schema_")
    copied.fit(frame, y)
    np.testing.assert_array_equal(copied.predict(frame), model.predict(frame))


@pytest.mark.parametrize(
    "estimator",
    [
        FeaturePassthrough(["price"]),
        FeatureEngineer([]),
        FeatureEngineer({"expression": pl.col("price")}),
        SchemaValidator(),
        ValueValidator({"price_positive": pl.col("price") > 0}),
    ],
)
def test_output_protocol(frame, estimator):
    assert estimator.set_output(transform="polars") is estimator
    with config_context(transform_output="pandas"):
        assert isinstance(estimator.fit_transform(frame), pl.DataFrame)
    assert isinstance(
        estimator.set_output(transform="default").transform(frame), pl.DataFrame
    )
    with pytest.raises(ValueError, match="supported"):
        estimator.set_output(transform="unsupported")


@pytest.mark.parametrize(
    "estimator",
    [
        FeaturePassthrough(["price"]),
        FeatureEngineer([]),
        FeatureEngineer({"expression": pl.col("price")}),
        SchemaValidator(),
        ValueValidator({"price_positive": pl.col("price") > 0}),
    ],
)
def test_relevant_estimator_checks(estimator):
    # Array-input checks intentionally don't apply to this DataFrame-only API.
    relevant = {
        "check_no_attributes_set_in_init",
        "check_estimators_get_params",
        "check_estimators_set_params",
        "check_estimator_cloneable",
        "check_do_not_raise_errors_in_init",
    }
    executed = set()
    for candidate, check in estimator_checks_generator(estimator):
        name = getattr(check, "func", check).__name__
        if name in relevant:
            check(candidate)
            executed.add(name)
    assert "check_no_attributes_set_in_init" in executed

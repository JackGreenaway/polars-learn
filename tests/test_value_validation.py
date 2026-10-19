import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from polars_learn import ValueValidationError, ValueValidator


def value_validator(**kwargs):
    return ValueValidator(
        {
            "price_positive": pl.col("price") > 0,
            "quantity_nonnegative": pl.col("quantity") >= 0,
            "quantity_present": pl.col("quantity").is_not_null(),
        },
        **kwargs,
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_valid_values_preserve_frame_and_fit_schema(lazy):
    frame = pl.DataFrame({"price": [1.0, 2.0], "quantity": [0, 3]})
    incoming = frame.lazy() if lazy else frame
    validator = value_validator()
    result = validator.fit_transform(incoming)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    assert_frame_equal(result.collect() if lazy else result, frame)
    assert validator.feature_names_in_.tolist() == frame.columns
    assert validator.get_feature_names_out().tolist() == frame.columns


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "frame, check",
    [
        (pl.DataFrame({"price": [0.0], "quantity": [1]}), "price_positive"),
        (
            pl.DataFrame({"price": [1.0], "quantity": [-1]}),
            "quantity_nonnegative",
        ),
        (
            pl.DataFrame(
                {"price": [1.0], "quantity": pl.Series([None], dtype=pl.Int64)}
            ),
            "quantity_present",
        ),
    ],
)
def test_failed_value_checks_name_each_failed_rule(frame, check, lazy):
    incoming = frame.lazy() if lazy else frame
    with pytest.raises(ValueValidationError, match=check):
        value_validator().fit_transform(incoming)


def test_transform_checks_values_and_exact_fitted_schema():
    frame = pl.DataFrame({"price": [1.0], "quantity": [2]})
    validator = value_validator().fit(frame)
    with pytest.raises(ValueValidationError, match="price_positive"):
        validator.transform(frame.with_columns(pl.lit(-1.0).alias("price")))
    with pytest.raises(ValueValidationError, match="Input schema differs"):
        validator.transform(frame.with_columns(pl.col("quantity").cast(pl.Int32)))
    with pytest.raises(ValueValidationError, match="Input schema differs"):
        validator.transform(frame.select("quantity", "price"))


def test_fit_resolves_check_schema_without_evaluating_values():
    frame = pl.DataFrame({"price": [-1.0], "quantity": [2]})
    validator = value_validator()
    original = frame.clone()
    assert validator.fit(frame) is validator
    assert_frame_equal(frame, original)
    with pytest.raises(ValueValidationError, match="price_positive"):
        validator.transform(frame)


def test_lazy_fit_does_not_execute_source():
    calls = 0

    def count_batches(frame):
        nonlocal calls
        calls += 1
        return frame

    source = (
        pl.DataFrame({"price": [1.0], "quantity": [2]})
        .lazy()
        .map_batches(count_batches, schema={"price": pl.Float64, "quantity": pl.Int64})
    )
    validator = value_validator()
    validator.fit(source)
    assert calls == 0


@pytest.mark.parametrize(
    "checks,error",
    [
        ([], TypeError),
        ({}, ValueError),
        ({"": pl.lit(True)}, ValueError),
        ({"bad": "not an expression"}, TypeError),
        ({"bad": pl.col("price")}, ValueError),
        ({"missing": pl.col("absent") > 0}, ValueValidationError),
    ],
)
def test_bad_checks_fail_clearly(checks, error):
    if isinstance(checks, dict) and (
        "missing" in checks
        or (
            "bad" in checks
            and isinstance(checks["bad"], pl.Expr)
            and checks["bad"].meta.output_name(raise_if_undetermined=False) == "price"
        )
    ):
        validator = ValueValidator(checks)
        message = "Could not resolve" if "missing" in checks else "Boolean"
        with pytest.raises(error, match=message):
            validator.fit(pl.DataFrame({"price": [1.0]}))
    else:
        with pytest.raises(error):
            ValueValidator(checks)


def test_null_check_results_are_violations():
    validator = ValueValidator({"accepted": pl.col("x") > 0})
    incoming = pl.DataFrame({"x": pl.Series([None], dtype=pl.Int64)})
    with pytest.raises(ValueValidationError, match="accepted"):
        validator.fit_transform(incoming)


def test_eager_output_materializes_a_lazy_source_once(monkeypatch):
    frame = pl.DataFrame({"price": [1.0], "quantity": [2]}).lazy()
    real_collect = pl.LazyFrame.collect
    calls = 0

    def count_collect(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return real_collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", count_collect)
    validator = value_validator().set_output(transform="polars")
    result = validator.fit_transform(frame)
    assert isinstance(result, pl.DataFrame)
    assert calls == 1


def test_lazy_default_output_does_not_reread_source():
    calls = 0

    def count_source(frame):
        nonlocal calls
        calls += 1
        return frame

    source = (
        pl.DataFrame({"x": [1, 2]})
        .lazy()
        .map_batches(count_source, schema={"x": pl.Int64})
    )
    validator = ValueValidator({"positive": pl.col("x") > 0})
    result = validator.fit_transform(source)
    assert isinstance(result, pl.LazyFrame)
    assert calls == 1
    assert result.collect()["x"].to_list() == [1, 2]
    assert calls == 1


def test_clone_and_pipeline_use_value_validator():
    frame = pl.DataFrame({"price": [1.0, 2.0, 3.0, 4.0], "quantity": [1, 2, 3, 4]})
    target = [1.0, 2.0, 3.0, 4.0]
    pipeline = Pipeline(
        [
            ("values", value_validator()),
            ("model", RandomForestRegressor(n_estimators=2, random_state=0)),
        ]
    )
    fitted = clone(pipeline).fit(frame, target)
    assert fitted.predict(frame).shape == (4,)
    with pytest.raises(ValueValidationError, match="price_positive"):
        fitted.predict(frame.with_columns(pl.lit(-1.0).alias("price")))


def test_transform_before_fit_raises_not_fitted():
    with pytest.raises(NotFittedError):
        value_validator().transform(pl.DataFrame({"price": [1.0], "quantity": [1]}))

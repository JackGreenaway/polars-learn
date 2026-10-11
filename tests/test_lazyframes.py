import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from polars_learn import (
    ColumnSelector,
    FeatureEngineer,
    FeatureEngineeringError,
    FeaturePassthrough,
    FeatureSpec,
    SchemaSpec,
    SchemaValidationError,
    SchemaValidator,
)

FACTORIES = [
    lambda **kw: FeaturePassthrough(["x"], **kw),
    lambda **kw: ColumnSelector(["x"], rename={"x": "selected"}, **kw),
    lambda **kw: FeatureEngineer(
        [
            FeatureSpec(pl.col("x") + 1, alias="first"),
            {"expression": pl.col("first") * 2, "alias": "second", "drop": ["first"]},
        ],
        **kw,
    ),
    lambda **kw: SchemaSpec({"x": pl.Int64, "y": pl.String}, **kw),
    lambda **kw: SchemaValidator(**kw),
]


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_lazy_fit_and_transform_do_not_collect(factory, verbosity, monkeypatch, capsys):
    frame = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    eager = factory().fit_transform(frame)
    estimator = factory(verbosity=verbosity)
    with monkeypatch.context() as context:

        def fail_collect(*args, **kwargs):
            pytest.fail("Lazy input was unexpectedly collected")

        context.setattr(pl.LazyFrame, "collect", fail_collect)
        result = estimator.fit_transform(frame.lazy())
        assert isinstance(result, pl.LazyFrame)
        transformed = estimator.transform(frame.lazy())
        assert isinstance(transformed, pl.LazyFrame)
        assert estimator.get_feature_names_out().tolist() == list(
            result.collect_schema()
        )
    assert_frame_equal(result.collect(), eager)
    assert_frame_equal(transformed.collect(), eager)
    output = capsys.readouterr().out
    assert ("unknown rows" in output) == (verbosity > 0)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("lazy_input", [False, True])
@pytest.mark.parametrize("mode", ["default", "polars", "polars-lazy"])
def test_output_controls_and_clone(factory, lazy_input, mode):
    frame = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    expected = factory().fit_transform(frame)
    estimator = factory().set_output(transform=mode)
    estimator = clone(estimator)
    estimator.set_output(transform=None)
    result = estimator.fit_transform(frame.lazy() if lazy_input else frame)
    wants_lazy = mode == "polars-lazy" or (mode == "default" and lazy_input)
    assert isinstance(result, pl.LazyFrame if wants_lazy else pl.DataFrame)
    assert_frame_equal(result.collect() if wants_lazy else result, expected)
    estimator.set_output(transform="default")
    assert isinstance(estimator.transform(frame), pl.DataFrame)


def test_lazy_pipeline_and_scan(tmp_path):
    path = tmp_path / "input.csv"
    pl.DataFrame({"x": [1, 2]}).write_csv(path)
    pipeline = Pipeline(
        [
            ("schema", SchemaValidator()),
            (
                "features",
                FeatureEngineer([FeatureSpec(pl.col("x") * 2, alias="doubled")]),
            ),
            ("select", ColumnSelector(["doubled"])),
        ]
    )
    result = pipeline.fit_transform(pl.scan_csv(path))
    assert isinstance(result, pl.LazyFrame)
    assert result.collect()["doubled"].to_list() == [2, 4]
    pipeline.set_output(transform="polars")
    assert isinstance(pipeline.transform(pl.scan_csv(path)), pl.DataFrame)


@pytest.mark.parametrize("lazy_fit", [False, True])
def test_cross_kind_fit_transform(lazy_fit):
    frame = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    for factory in FACTORIES:
        estimator = factory().fit(frame.lazy() if lazy_fit else frame)
        assert isinstance(estimator.transform(frame), pl.DataFrame)
        assert isinstance(estimator.transform(frame.lazy()), pl.LazyFrame)


def test_lazy_schema_errors_fail_before_collect():
    frame = pl.DataFrame({"x": [1]}).lazy()
    with pytest.raises(FeatureEngineeringError, match="missing"):
        FeatureEngineer([FeatureSpec(pl.col("missing"))]).fit(frame)
    with pytest.raises(FeatureEngineeringError, match="Cannot drop"):
        FeatureEngineer([FeatureSpec(pl.col("x"), drop=["missing"])]).fit(frame)
    with pytest.raises(ValueError, match="Missing required"):
        ColumnSelector(["missing"]).fit(frame)
    with pytest.raises(ValueError, match="dtype"):
        SchemaSpec({"x": pl.String}).fit(frame)
    validator = SchemaValidator().fit(frame)
    with pytest.raises(SchemaValidationError, match="Missing required"):
        validator.transform(pl.DataFrame({"y": [1]}).lazy())
    with pytest.raises(SchemaValidationError, match="incompatible"):
        validator.transform(pl.DataFrame({"x": ["1"]}).lazy())


def test_lazy_casts_and_value_errors():
    validator = SchemaValidator(mode="permissive").fit(pl.DataFrame({"x": [1]}).lazy())
    result = validator.transform(pl.DataFrame({"x": ["2", None]}).lazy())
    assert result.collect()["x"].to_list() == [2, None]
    invalid = validator.transform(pl.DataFrame({"x": ["invalid"]}).lazy())
    with pytest.raises(pl.exceptions.InvalidOperationError):
        invalid.collect()
    validator.set_output(transform="polars")
    with pytest.raises(SchemaValidationError, match="Lazy schema validation") as caught:
        validator.transform(pl.DataFrame({"x": ["invalid"]}).lazy())
    assert isinstance(caught.value.__cause__, pl.exceptions.PolarsError)


def test_lazy_semi_strict_lossless_float_cast():
    validator = SchemaValidator(mode="semi-strict").fit(
        pl.DataFrame({"x": [1.0]}).lazy()
    )
    result = validator.transform(pl.DataFrame({"x": [2, None]}).lazy())
    assert result.collect()["x"].to_list() == [2.0, None]
    invalid = validator.transform(pl.DataFrame({"x": [2**53 + 1]}).lazy())
    with pytest.raises(SchemaValidationError, match="losslessly"):
        invalid.collect()
    # Schema-only widening remains lazy and respects the dtype range policy.
    integers = SchemaValidator(mode="semi-strict").fit(pl.DataFrame({"x": [1]}).lazy())
    assert (
        integers.transform(
            pl.DataFrame({"x": pl.Series([2], dtype=pl.Int32)}).lazy()
        ).collect_schema()["x"]
        == pl.Int64
    )
    with pytest.raises(SchemaValidationError, match="semi-strict"):
        integers.transform(pl.DataFrame({"x": ["2"]}).lazy())


def test_lazy_feature_value_error_and_explicit_collection():
    frame = pl.DataFrame({"x": ["invalid"]}).lazy()
    engineer = FeatureEngineer([FeatureSpec(pl.col("x").cast(pl.Int64))])
    result = engineer.fit_transform(frame)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        result.collect()
    engineer.set_output(transform="polars")
    with pytest.raises(FeatureEngineeringError, match="Failed to execute"):
        engineer.transform(frame)


def test_lazy_extras_reordering_and_drop():
    train = pl.DataFrame({"x": [1], "y": [2]}).lazy()
    test = pl.DataFrame({"y": [2], "extra": [3], "x": [1]}).lazy()
    validator = SchemaValidator(
        allow_extra_columns=True, enforce_column_order=False
    ).fit(train)
    assert list(validator.transform(test).collect_schema()) == ["x", "y", "extra"]
    validator.set_params(extra_columns="drop")
    assert list(validator.transform(test).collect_schema()) == ["x", "y"]
    spec = SchemaSpec(
        {"x": pl.Int64},
        allow_extra_columns=True,
        enforce_column_order=False,
        extra_columns="drop",
    )
    assert spec.fit_transform(test).collect().columns == ["x"]

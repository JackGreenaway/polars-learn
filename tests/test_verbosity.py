import polars as pl
import pytest
from sklearn.base import clone

from polars_learn import (
    ColumnSelector,
    FeatureEngineer,
    FeaturePassthrough,
    FeatureSpec,
    SchemaSpec,
    SchemaValidator,
)

FACTORIES = [
    lambda **kw: FeaturePassthrough(["x"], **kw),
    lambda **kw: ColumnSelector(["x"], **kw),
    lambda **kw: FeatureEngineer(FeatureSpec(pl.col("x") + 1), **kw),
    lambda **kw: SchemaSpec({"x": pl.Int64}, **kw),
    lambda **kw: SchemaValidator(**kw),
]


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_diagnostics_and_cloning(factory, verbosity, capsys):
    estimator = factory(verbosity=verbosity)
    assert clone(estimator).verbosity == verbosity
    result = estimator.fit_transform(pl.DataFrame({"x": [1, 2]}))
    assert result.shape == (2, 1)
    output = capsys.readouterr().out
    if verbosity == 0:
        assert output == ""
    else:
        assert "Fit complete" in output
        assert "Transform complete" in output
        assert ("Input schema" in output) == (verbosity == 2)


@pytest.mark.parametrize(
    "factory", FACTORIES + [lambda **kw: FeatureSpec(pl.col("x"), **kw)]
)
@pytest.mark.parametrize(
    "verbosity,error",
    [(True, TypeError), ("1", TypeError), (-1, ValueError), (3, ValueError)],
)
def test_invalid_verbosity_fails_at_construction(factory, verbosity, error):
    with pytest.raises(error, match="verbosity"):
        factory(verbosity=verbosity)


@pytest.mark.parametrize("factory", FACTORIES)
def test_options_revalidated_before_input_conversion(factory):
    estimator = factory().set_params(convert_pandas="yes")
    with pytest.raises(TypeError, match="convert_pandas must be a bool"):
        estimator.fit(object())
    estimator.set_params(convert_pandas=False, verbosity=-1)
    with pytest.raises(ValueError, match="verbosity"):
        estimator.fit(object())


def test_feature_spec_diagnostics(capsys):
    FeatureSpec(pl.col("x"), verbosity=2)
    output = capsys.readouterr().out
    assert "Validated 1 expression(s)" in output
    assert "Outputs: ['x']" in output


def test_fail_fast_feature_and_selector_configuration():
    with pytest.raises(TypeError, match=r"features\[1\]"):
        FeatureEngineer([FeatureSpec(pl.col("x")), object()])
    with pytest.raises(ValueError, match="unique"):
        ColumnSelector(["x", "x"])
    with pytest.raises(ValueError, match="unique"):
        ColumnSelector(["x", "y"], rename={"x": "y"})
    with pytest.raises(ValueError, match="mode must be"):
        SchemaValidator(mode="unknown")


def test_schema_spec_preserves_training_extras():
    frame = pl.DataFrame({"x": [1], "extra": [2]})
    estimator = SchemaSpec({"x": pl.Int64}, allow_extra_columns=True)
    assert estimator.fit_transform(frame).equals(frame)
    assert estimator.get_feature_names_out().tolist() == ["x", "extra"]

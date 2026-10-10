import polars as pl
import pytest

from polars_learn import FeatureEngineer, FeatureSpec


def test_canonical_expressions_and_native_names(frame):
    spec = FeatureSpec(expression=pl.col("price") + 1)
    assert len(spec.expressions) == 1
    result = FeatureEngineer([spec]).fit_transform(frame)
    assert result["price"].to_list() == [2, 3, 4, 5]
    assert "expression=" not in repr(spec)


def test_alias_overrides_native_alias(frame):
    spec = FeatureSpec(expression=pl.col("price").alias("old"), alias="new")
    assert FeatureEngineer([spec]).fit_transform(frame).columns[-1] == "new"


def test_multiple_and_drop(frame):
    spec = FeatureSpec(
        expressions=[pl.col("price").alias("a"), pl.col("quantity").alias("b")],
        drop=["price", "quantity"],
    )
    assert FeatureEngineer([spec]).fit_transform(frame).columns == ["a", "b"]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({}, ValueError),
        ({"expression": pl.col("x"), "expressions": [pl.col("x")]}, ValueError),
        ({"expressions": []}, ValueError),
        ({"expression": "x"}, TypeError),
        ({"expressions": "x"}, TypeError),
        ({"expressions": [pl.col("x"), 1]}, TypeError),
        ({"expression": pl.col("x"), "alias": ""}, ValueError),
        ({"expressions": [pl.col("x"), pl.col("y")], "alias": "a"}, ValueError),
        ({"expression": pl.all(), "alias": "a"}, ValueError),
        ({"expressions": [pl.col("x").alias("a"), pl.col("y").alias("a")]}, ValueError),
        ({"expression": pl.col("x"), "drop": "x"}, TypeError),
        ({"expression": pl.col("x"), "drop": [1]}, TypeError),
        ({"expression": pl.col("x"), "name": ""}, ValueError),
    ],
)
def test_invalid_specs(kwargs, error):
    with pytest.raises(error):
        FeatureSpec(**kwargs)


def test_selector_expansion(frame):
    result = FeatureEngineer([FeatureSpec(expression=pl.all().name.suffix("_copy"))])
    assert result.fit_transform(frame).columns == [
        "price",
        "quantity",
        "price_copy",
        "quantity_copy",
    ]

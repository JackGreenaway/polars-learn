import polars as pl
import pytest

from polars_learn import ColumnSelector, FeatureEngineer, FeatureSpec, SchemaSpec


def test_canonical_expressions_and_native_names(frame):
    spec = FeatureSpec(expression=pl.col("price") + 1)
    assert len(spec.expression) == 1
    result = FeatureEngineer([spec]).fit_transform(frame)
    assert result["price"].to_list() == [2, 3, 4, 5]
    assert "expression=" in repr(spec)


def test_alias_overrides_native_alias(frame):
    spec = FeatureSpec(expression=pl.col("price").alias("old"), alias="new")
    assert FeatureEngineer([spec]).fit_transform(frame).columns[-1] == "new"


def test_multiple_and_drop(frame):
    spec = FeatureSpec(
        expression=[pl.col("price").alias("a"), pl.col("quantity").alias("b")],
        drop=["price", "quantity"],
    )
    assert FeatureEngineer([spec]).fit_transform(frame).columns == ["a", "b"]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({}, TypeError),
        ({"expression": pl.col("x"), "expressions": [pl.col("x")]}, TypeError),
        ({"expression": []}, ValueError),
        ({"expression": "x"}, TypeError),
        ({"expression": None}, TypeError),
        ({"expression": [pl.col("x"), 1]}, TypeError),
        ({"expression": pl.col("x"), "alias": ""}, ValueError),
        ({"expression": [pl.col("x"), pl.col("y")], "alias": "a"}, ValueError),
        ({"expression": pl.all(), "alias": "a"}, ValueError),
        ({"expression": [pl.col("x").alias("a"), pl.col("y").alias("a")]}, ValueError),
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


def test_schema_spec_validates_required_columns_and_dtypes(frame):
    schema = SchemaSpec({"price": pl.Float64, "quantity": pl.Int64})
    validated = schema.fit_transform(frame)
    assert validated.columns == ["price", "quantity"]
    assert dict(validated.schema) == {"price": pl.Float64, "quantity": pl.Int64}

    with pytest.raises(ValueError):
        SchemaSpec({"price": pl.Int64}).transform(pl.DataFrame({"quantity": [1, 2]}))


def test_column_selector_reorders_and_renames(frame):
    selector = ColumnSelector(["quantity", "price"], rename={"quantity": "qty"})
    result = selector.fit_transform(frame)
    assert result.columns == ["qty", "price"]
    assert result["qty"].to_list() == [2, 3, 4, 5]

    with pytest.raises(ValueError):
        ColumnSelector(["missing"]).fit_transform(frame)


@pytest.mark.parametrize("container", [list, tuple])
def test_expression_sequence_normalization_is_independent_of_input(container, frame):
    expressions = container([pl.col("price").alias("a"), pl.col("quantity").alias("b")])
    spec = FeatureSpec(expressions)
    assert isinstance(spec.expression, tuple)
    if isinstance(expressions, list):
        expressions.clear()
    assert len(spec.expression) == 2
    assert FeatureEngineer(spec).fit_transform(frame).columns[-2:] == ["a", "b"]


@pytest.mark.parametrize(
    "expression", [b"price", {"price": pl.col("price")}, iter([pl.col("price")]), 1]
)
def test_unsupported_expression_containers_fail_immediately(expression):
    with pytest.raises(TypeError, match="expression must be"):
        FeatureSpec(expression)

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polars_learn import (
    FeatureEngineer,
    FeaturePassthrough,
    FeatureSpec,
)


def test_values_dtypes_nulls_and_order_are_preserved():
    frame = pl.DataFrame(
        {
            "id": pl.Series([3, None, 1], dtype=pl.UInt32),
            "category": pl.Series(["b", None, "a"], dtype=pl.Categorical),
            "date": [date(2026, 1, 1), None, date(2026, 1, 3)],
            "unused": [1.0, 2.0, 3.0],
        }
    )
    original = frame.clone()
    keep = FeaturePassthrough(["date", "category", "id"])
    result = keep.fit_transform(frame)
    assert_frame_equal(result, frame.select("date", "category", "id"))
    assert_frame_equal(frame, original)
    assert keep.get_feature_names_out().tolist() == ["date", "category", "id"]
    assert keep.n_features_in_ == 4
    assert keep.feature_names_in_.tolist() == frame.columns
    names = keep.get_feature_names_out()
    names[0] = "changed"
    assert keep.get_feature_names_out()[0] == "date"


def test_pipeline_layer_passes_selected_features_unchanged():
    frame = pl.DataFrame({"x": [1, 2], "unused": [3, 4]})
    pipeline = Pipeline(
        [
            ("keep", FeaturePassthrough(["x"])),
            (
                "features",
                FeatureEngineer([FeatureSpec(pl.col("x") * 2, alias="doubled")]),
            ),
        ]
    )
    result = pipeline.fit_transform(frame.lazy())
    assert isinstance(result, pl.LazyFrame)
    assert result.collect().to_dict(as_series=False) == {"x": [1, 2], "doubled": [2, 4]}


def test_column_transformer_combines_scaled_and_unchanged_branches():
    frame = pl.DataFrame({"value": [10.0, 20.0], "id": [7, 8], "flag": [True, False]})
    transformer = ColumnTransformer(
        [
            ("scaled", StandardScaler(), ["value"]),
            ("unchanged", FeaturePassthrough(["id", "flag"]), ["id", "flag"]),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="polars")
    result = transformer.fit_transform(frame)
    assert result.columns == ["value", "id", "flag"]
    assert result["value"].to_list() == [-1.0, 1.0]
    assert_frame_equal(result.select("id", "flag"), frame.select("id", "flag"))
    assert transformer.get_feature_names_out().tolist() == result.columns
    assert_frame_equal(transformer.transform(frame), result)
    assert_frame_equal(clone(transformer).fit_transform(frame), result)


@pytest.mark.parametrize(
    "columns,error",
    [
        ("x", TypeError),
        ([], TypeError),
        ([1], TypeError),
        ([""], TypeError),
        (["x", "x"], ValueError),
    ],
)
def test_bad_selection_fails_at_construction(columns, error):
    with pytest.raises(error):
        FeaturePassthrough(columns)


def test_missing_columns_fitted_state_and_parameter_changes():
    frame = pl.DataFrame({"x": [1]})
    with pytest.raises(NotFittedError):
        FeaturePassthrough(["x"]).transform(frame)
    with pytest.raises(ValueError, match="Missing required"):
        FeaturePassthrough(["missing"]).fit(frame.lazy())
    keep = FeaturePassthrough(["x"]).fit(frame)
    with pytest.raises(ValueError, match="Missing required"):
        keep.transform(pl.DataFrame({"y": [1]}).lazy())
    keep.set_params(columns=["x", "x"])
    with pytest.raises(ValueError, match="unique"):
        keep.transform(frame)
    keep.set_params(columns=["y"])
    assert keep.fit_transform(pl.DataFrame({"y": [2]})).columns == ["y"]
    with pytest.raises(ValueError, match="input_features"):
        keep.get_feature_names_out(["x"])


def test_dtype_changes_and_unselected_columns_are_allowed():
    keep = FeaturePassthrough(["x"]).fit(pl.DataFrame({"x": [1], "unused": [2]}))
    incoming = pl.DataFrame({"new": [3], "x": ["changed dtype"]})
    assert_frame_equal(keep.transform(incoming), incoming.select("x"))


def test_no_rename_parameter():
    keep = FeaturePassthrough(["x"])
    assert "rename" not in keep.get_params()
    with pytest.raises(TypeError, match="rename"):
        FeaturePassthrough(["x"], rename={"x": "y"})
    with pytest.raises(ValueError, match="rename"):
        keep.set_params(rename={"x": "y"})

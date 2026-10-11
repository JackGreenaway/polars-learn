import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from polars_learn import SchemaValidationError, SchemaValidator


def test_schema_inference_and_repeated_transform(frame):
    est = SchemaValidator()
    assert est.fit(frame) is est
    assert est.schema_ == dict(frame.schema)
    assert est.feature_names_in_.tolist() == frame.columns
    assert est.n_features_in_ == 2
    assert est.get_feature_names_out(frame.columns).tolist() == frame.columns
    for _ in range(3):
        assert_frame_equal(est.transform(frame), frame)
    copied = clone(est)
    assert not hasattr(copied, "schema_")
    assert copied.get_params()["mode"] == "strict"
    assert_frame_equal(copied.fit_transform(frame), frame)
    with pytest.raises(ValueError):
        est.get_feature_names_out(["bad"])


def test_before_fit(frame):
    with pytest.raises(NotFittedError):
        SchemaValidator().transform(frame)


def test_multiple_mismatches(frame):
    est = SchemaValidator().fit(frame)
    with pytest.raises(SchemaValidationError) as caught:
        est.transform(pl.DataFrame({"price": ["1"], "wrong": [1]}))
    message = str(caught.value)
    assert "Missing" in message and "Unexpected" in message
    assert "Expected: Float64" in message and "Received: String" in message


@pytest.mark.parametrize(
    "source, target, values",
    [
        (pl.Int8, pl.Int64, [1, None]),
        (pl.UInt8, pl.Int16, [255, None]),
        (pl.Int32, pl.Int64, [1, None]),
        (pl.Float32, pl.Float64, [1.5, None]),
        (pl.Int64, pl.Float64, [2**53, None]),
        (pl.Int64, pl.Float64, [2**54, None]),
        (pl.UInt64, pl.Float64, [2**63, None]),
    ],
)
def test_safe_widening(source, target, values):
    training = pl.DataFrame({"x": pl.Series([1], dtype=target)})
    incoming = pl.DataFrame({"x": pl.Series(values, dtype=source)})
    out = SchemaValidator(mode="semi-strict").fit(training).transform(incoming)
    assert out.schema["x"] == target
    assert out["x"].null_count() == 1


@pytest.mark.parametrize(
    "source, target, values",
    [
        (pl.Int64, pl.Float64, [2**53 + 1]),
        (pl.Int64, pl.Float64, [2**63 - 1]),
        (pl.UInt64, pl.Float64, [2**64 - 1]),
        (pl.Int64, pl.Int32, [1]),
        (pl.Int8, pl.UInt8, [1]),
        (pl.Float64, pl.Float32, [1.0]),
        (pl.String, pl.Int64, ["1"]),
        (pl.Null, pl.Int64, [None]),
    ],
)
def test_unsafe_semi_strict(source, target, values):
    est = SchemaValidator(mode="semi-strict").fit(
        pl.DataFrame({"x": pl.Series([1], dtype=target)})
    )
    with pytest.raises(SchemaValidationError, match="semi-strict"):
        est.transform(pl.DataFrame({"x": pl.Series(values, dtype=source)}))


def test_strict_never_casts(frame):
    with pytest.raises(SchemaValidationError, match="strict"):
        SchemaValidator().fit(frame).transform(
            frame.with_columns(pl.col("quantity").cast(pl.Int32))
        )


def test_permissive_and_failed_cast(frame):
    est = SchemaValidator(mode="permissive").fit(frame)
    incoming = frame.with_columns(pl.col("quantity").cast(pl.String))
    assert_frame_equal(est.transform(incoming), frame)
    bad = incoming.with_columns(pl.lit("invalid").alias("quantity"))
    with pytest.raises(SchemaValidationError, match="Cast failed") as caught:
        est.transform(bad)
    assert caught.value.__cause__ is not None
    assert bad["quantity"].null_count() == 0


def test_order_and_extras(frame):
    extra = frame.with_columns(pl.lit("ok").alias("extra"))
    with pytest.raises(SchemaValidationError, match="Unexpected"):
        SchemaValidator().fit(frame).transform(extra)
    est = SchemaValidator(allow_extra_columns=True).fit(frame)
    assert_frame_equal(est.transform(extra), extra)
    assert est.get_feature_names_out().tolist() == frame.columns
    dropping = SchemaValidator(allow_extra_columns=True, extra_columns="drop").fit(
        frame
    )
    assert_frame_equal(dropping.transform(extra), frame)
    reordered = extra.select("quantity", "extra", "price")
    with pytest.raises(SchemaValidationError, match="order"):
        est.transform(reordered)
    normalising = SchemaValidator(
        allow_extra_columns=True, enforce_column_order=False
    ).fit(frame)
    assert_frame_equal(normalising.transform(reordered), extra)
    with pytest.raises(SchemaValidationError, match="Missing"):
        normalising.transform(extra.drop("price"))


def test_nulls_and_no_mutation():
    frame = pl.DataFrame({"x": pl.Series([None, 1], dtype=pl.Int64)})
    original = frame.clone()
    assert_frame_equal(SchemaValidator().fit_transform(frame), original)
    assert_frame_equal(frame, original)
    null = pl.DataFrame({"x": [None]})
    est = SchemaValidator(mode="permissive").fit(frame)
    assert est.transform(null).schema == frame.schema


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad"},
        {"extra_columns": "bad"},
        {"mode": None},
        {"mode": True},
    ],
)
def test_invalid_configuration(frame, kwargs):
    with pytest.raises((ValueError, TypeError)):
        SchemaValidator(**kwargs).fit(frame)

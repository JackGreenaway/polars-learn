"""Check real cross-version artifacts in CI using two Polars environments."""

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from polars_learn import (
    ColumnSelector,
    FeatureEngineer,
    FeaturePassthrough,
    FeatureSpec,
    InconsistentVersionWarning,
    SchemaSpec,
    SchemaValidator,
    ValueValidator,
)


def main():
    mode, path = sys.argv[1], Path(sys.argv[2])
    frame = pl.DataFrame({"price": [1.0, 2.0, 3.0], "quantity": [2, 3, 4]})
    if mode == "write":

        def configured_callback():
            factor = 2.0
            return {
                "expression": pl.col("price").map_batches(
                    lambda s: s * factor, return_dtype=pl.Float64
                ),
                "alias": "custom",
            }

        objects = {
            "spec": FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
            "engineer": FeatureEngineer(
                {"expression": pl.col("price") * 2, "alias": "double"}
            ),
            "columns": ColumnSelector(
                ["quantity", "price"], rename={"quantity": "qty"}
            ),
            "passthrough": FeaturePassthrough(["price"]),
            "contract": SchemaSpec(dict(frame.schema)),
            "validator": SchemaValidator(),
            "values": ValueValidator({"positive_price": pl.col("price") > 0}),
            "callback": FeatureEngineer(configured_callback()),
            "pipeline": Pipeline(
                [
                    ("schema", SchemaValidator()),
                    (
                        "values",
                        ValueValidator({"positive_price": pl.col("price") > 0}),
                    ),
                    (
                        "features",
                        FeatureEngineer(
                            FeatureSpec(
                                pl.col("price") * pl.col("quantity"), alias="notional"
                            )
                        ),
                    ),
                    ("model", RandomForestRegressor(n_estimators=5, random_state=0)),
                ]
            ),
        }
        results = {}
        for name, obj in objects.items():
            if name == "pipeline":
                obj.fit(frame, np.array([1.0, 2.0, 3.0]))
                results[name] = obj.predict(frame).tolist()
            else:
                transformer = FeatureEngineer(obj) if name == "spec" else obj
                results[name] = transformer.fit_transform(frame).to_dict(
                    as_series=False
                )
        path.write_bytes(pickle.dumps(objects, protocol=pickle.HIGHEST_PROTOCOL))
        path.with_suffix(".json").write_text(json.dumps(results))
        print("Saved with Polars", pl.__version__)
    else:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", InconsistentVersionWarning)
            objects = pickle.loads(path.read_bytes())
        expected = json.loads(path.with_suffix(".json").read_text())
        for name, obj in objects.items():
            if name == "pipeline":
                np.testing.assert_array_equal(obj.predict(frame), expected[name])
            else:
                transformer = FeatureEngineer(obj).fit(frame) if name == "spec" else obj
                assert (
                    transformer.transform(frame).to_dict(as_series=False)
                    == expected[name]
                )
        version_warnings = [
            w.message
            for w in captured
            if isinstance(w.message, InconsistentVersionWarning)
        ]
        assert version_warnings
        assert all(w.dependency == "polars" for w in version_warnings)
        print(
            "Loaded with Polars",
            pl.__version__,
            "; all 9 objects match; version warnings:",
            len(version_warnings),
        )


if __name__ == "__main__":
    main()

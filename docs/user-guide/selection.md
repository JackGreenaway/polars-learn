# Selection and explicit passthrough

Use `FeaturePassthrough(columns)` when a pipeline layer should simply carry named columns forward. It returns just those columns in the specified order, with their Polars values, names, dtypes, nulls, and row order preserved. There is no renaming or casting option. Selected columns must exist; input order and dtypes may differ between calls. It supports the same verbosity, pandas conversion, fitted feature names, and output controls as the other transformers.

```python
import polars as pl
from sklearn.pipeline import Pipeline

from polars_learn import FeaturePassthrough

X = pl.DataFrame({"id": [7, 8], "flag": [True, False], "unused": [1, 2]})
pipeline = Pipeline([("unchanged", FeaturePassthrough(["id", "flag"]))])
result = pipeline.fit_transform(X.lazy())
assert isinstance(result, pl.LazyFrame)
assert result.collect().equals(X.select("id", "flag"))
```

A sequential `Pipeline` passes each step's output to the next step, so a passthrough step selecting two columns omits all the others. To keep unchanged columns **alongside** transformed columns, put the operations in parallel `ColumnTransformer` branches:

```python
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from polars_learn import FeaturePassthrough

X = pl.DataFrame({"value": [10.0, 20.0], "id": [7, 8], "flag": [True, False]})
preprocessing = ColumnTransformer(
    [
        ("scaled", StandardScaler(), ["value"]),
        ("unchanged", FeaturePassthrough(["id", "flag"]), ["id", "flag"]),
    ],
    verbose_feature_names_out=False,
).set_output(transform="polars")
result = preprocessing.fit_transform(X)
assert result["value"].to_list() == [-1.0, 1.0]
assert result.select("id", "flag").equals(X.select("id", "flag"))
assert result.columns == ["value", "id", "flag"]
```

The outer column lists tell `ColumnTransformer` which input to send to each branch; `FeaturePassthrough` declares its own selected-column contract. Use eager input at this boundary (`lazy_X.collect()` for a lazy query). Polars output keeps the separate column dtypes. `verbose_feature_names_out=False` prevents branch prefixes on output names; final names must then be unique. For branch behavior, see the [scikit-learn ColumnTransformer reference](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html). Use `ColumnSelector` when you also need optional renaming.

## Selecting and renaming

Use `ColumnSelector(["quantity", "price"], rename={"quantity": "qty"})` to establish output order and optionally rename selected columns. Use `FeaturePassthrough` to communicate that selected values require no transformation. Both drop unselected columns; neither checks dtype drift or casts values. Combine them with a schema validator when dtype enforcement is required.

ColumnTransformer examples use eager frames. Its outer output setting is a scikit-learn setting, while native transformers' `"polars-lazy"` setting belongs to this package. See the [selection API](../api.md#columnselector) and [passthrough API](../api.md#featurepassthrough).

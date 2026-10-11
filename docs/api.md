# API Reference

Import public classes and exceptions from `polars_learn`. This reference describes
the implementation in this checkout. See the [README](../README.md) for complete
workflows, examples, and development instructions.

## Shared Transformer Behavior

`ColumnSelector`, `FeaturePassthrough`, `SchemaSpec`, `SchemaValidator`, and
`FeatureEngineer` are
scikit-learn transformers. `FeatureSpec` describes an operation and is not an
estimator.

### Input and Output

Accepted input is a `polars.DataFrame` or `polars.LazyFrame`. A pandas DataFrame
is accepted only with `convert_pandas=True`. NumPy arrays, Series, lists, and
sparse arrays are unsupported. Original inputs are never mutated.

| Output configuration | Eager input, including converted pandas | Lazy input |
| --- | --- | --- |
| Initial setting or `set_output(transform="default")` | DataFrame | LazyFrame |
| `set_output(transform="polars")` | DataFrame | DataFrame, collected by the transformer |
| `set_output(transform="polars-lazy")` | LazyFrame | LazyFrame |

`set_output(transform=None)` keeps the current choice. `"pandas"` output is
unsupported. Global scikit-learn output settings do not change these native
transformers' returns. Explicit output choices survive cloning.

Default return type follows the input of each transform call, not the input
used during fitting. Lazy schema checks use metadata and do not evaluate rows;
source metadata access can still occur. Value-dependent failures occur at query
execution. Requesting lazy output for eager input can wrap an already evaluated
result. To defer row evaluation throughout, supply a LazyFrame.

### Common Constructor Options

These options are keyword-only on all five transformers:

| Option | Default | Meaning |
| --- | --- | --- |
| `convert_pandas` | `False` | Permit pandas DataFrames and convert them to Polars. Must be a bool. |
| `warn_on_conversion` | `True` | Emit `PandasConversionWarning` on conversion. Must be a bool; independent of verbosity. |
| `verbosity` | `0` | Print progress at 1 and additional metadata/operation details at 2. Must be an integer 0, 1, or 2; bool is rejected. |

Native Polars input needs neither pandas nor PyArrow. Pandas conversion may need
PyArrow depending on dtype; install `polars-learn[pandas]` for both dependencies.
Conversion follows Polars' pandas conversion semantics and discards the index.

### Common Methods

| Method | Return | Behavior |
| --- | --- | --- |
| `fit(X, y=None)` | `self` | Validate or learn the class-specific contract. Ignores `y`. Repeated fitting replaces learned state after validation succeeds. |
| `transform(X)` | DataFrame or LazyFrame | Apply a fitted transformer; output follows the table above. |
| `fit_transform(X, y=None, **fit_params)` | DataFrame or LazyFrame | Call `fit` then `transform`. Extra keyword parameters are unsupported and raise `TypeError`. |
| `get_feature_names_out(input_features=None)` | NumPy object array of strings | Return a copy of fitted output names. If supplied, input names must exactly match the fitted input names and order. |
| `get_params(deep=True)` | dict | Read constructor parameters using scikit-learn's estimator API. Output configuration is set separately with `set_output`. |
| `set_params(**params)` | `self` | Update constructor parameters. Does not validate immediately or clear fitted state; refit after contract changes. |
| `set_output(*, transform=None)` | `self` | Choose the output container for subsequent transforms. Does not fit, evaluate data, or change column names. |

`transform` and `get_feature_names_out` require fitting and raise
`sklearn.exceptions.NotFittedError` if fitted state is unavailable.
Invalid options changed through `set_params` are checked at the next relevant
fit/transform operation. A failed refit can leave a previous successful fitted
state in place; use a fresh instance if that state should be discarded.

All five transformers expose these fitted attributes:

| Attribute | Meaning |
| --- | --- |
| `feature_names_in_` | NumPy object array of all input names observed during fitting, in input order. |
| `n_features_in_` | Number of columns observed during fitting, including extras and unselected columns. |
| `output_features_` | NumPy object array of output names, with the class-specific extra-column behavior described below. |

Constructor parameters are retained for cloning. Cloning discards fitted
attributes. Configuration that changes names or schemas should be followed by
fitting again rather than assuming existing fitted attributes were reset.

## ColumnSelector

```text
ColumnSelector(
    columns,
    *,
    rename=None,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

| Parameter | Contract |
| --- | --- |
| `columns` | Nonempty sequence of unique, nonempty string column names. Determines output order. A bare string is invalid. |
| `rename` | Optional dict mapping selected source names to nonempty string output names. Keys must be selected columns, and final output names must be unique, including names left unchanged. |

Fitting checks that every selected column exists and records its output name.
The input can contain other columns, and its order need not match `columns`.
Transform selects the fitted `columns_`, applies the rename mapping, and verifies
that output names still match fitting. Input and output dtypes are not checked.
Unselected columns are omitted.

Additional fitted attribute: `columns_`, a tuple of selected source names.
`output_features_` lists the selected output names after renaming.

Wrong argument types raise `TypeError`; duplicate names, invalid rename mappings,
missing selected columns, or changed output names raise `ValueError`.
Source query errors can propagate from Polars.

```python
import polars as pl

from polars_learn import ColumnSelector

X = pl.DataFrame({"a": [1], "b": [2], "unused": [3]})
selector = ColumnSelector(["b", "a"], rename={"b": "value"})
result = selector.fit_transform(X.lazy())
assert result.collect().columns == ["value", "a"]
assert selector.get_feature_names_out().tolist() == ["value", "a"]
```

## FeaturePassthrough

```text
FeaturePassthrough(
    columns,
    *,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

`columns` is a nonempty sequence of unique, nonempty string names, in output
order. A bare string is invalid. Only the selected columns are returned, with
Polars values, names, dtypes, nulls, and row order preserved. Unselected input
columns are omitted. There is no `rename` parameter; use `ColumnSelector` for
renaming.

This class specializes `ColumnSelector` and shares its methods and fitted
attributes, including `columns_`. Fitting checks selected columns exist;
transform requires fitting. Input order and dtypes may change without rejection.
`feature_names_in_` and `n_features_in_` describe the whole training input,
including unselected columns; `output_features_` contains unchanged selected names.
The common input conversion, verbosity, cloning, and output policies apply.

Invalid selection types raise `TypeError`; duplicate names, missing columns,
and invalid verbosity levels raise `ValueError`. Transform before fitting raises
`NotFittedError`. `get_params` exposes only its constructor parameters, with no
rename option. Source/schema-resolution and collection errors are not wrapped.
Refit after changing `columns` through `set_params`.

A sequential pipeline step selects its output subset. Use an eager
`ColumnTransformer` branch when selected unchanged columns should be combined
with another branch's transformed columns. Configure the outer transformer
with `set_output(transform="polars")` to preserve separate column dtypes, and
`verbose_feature_names_out=False` to keep unprefixed names when they are unique.
See the [README branch example](../README.md#explicit-feature-passthrough).

```python
import polars as pl

from polars_learn import FeaturePassthrough

X = pl.DataFrame({"id": [1, None], "flag": [True, False], "unused": [9, 8]})
keep = FeaturePassthrough(["flag", "id"])
plan = keep.fit_transform(X.lazy())
assert plan.collect().equals(X.select("flag", "id"))
assert keep.get_feature_names_out().tolist() == ["flag", "id"]
```

## SchemaSpec

```text
SchemaSpec(
    schema,
    *,
    allow_extra_columns=False,
    enforce_column_order=True,
    extra_columns="preserve",
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

| Parameter | Contract |
| --- | --- |
| `schema` | Nonempty dict from nonempty string names to Polars dtypes, in required column order. Accepts dtype classes such as `pl.Int64` and instances such as `pl.Datetime("ms")`. |
| `allow_extra_columns` | Bool; whether input may include names absent from the explicit schema. |
| `enforce_column_order` | Bool; require the relative order of required input columns to match the schema's key order. |
| `extra_columns` | `"preserve"` or `"drop"`; treatment of extras if allowed. Does not itself allow extras. |

Fit and transform both require all schema columns and exact required dtypes;
there is no casting policy. Output always orders required columns by schema key
order. Preserved extras follow in their input order.

Fitting checks the declared contract, not an inferred replacement. Additional
fitted attribute: `schema_`, a copy of the required schema.
`output_features_` contains every fitted output column, including preserved extras.
Subsequent output names and order must match exactly. Extra dtypes are not checked.
Use `extra_columns="drop"` for a fixed output set containing only required columns.

Missing required columns, unexpected extras, disallowed order, wrong required
dtypes, and output-name mismatches raise `ValueError`. An invalid schema
container or empty schema raises `ValueError`; invalid schema keys/values raise
`TypeError`. Source/collection errors are not wrapped.

```python
import polars as pl

from polars_learn import SchemaSpec

X = pl.DataFrame({"extra": [9], "x": [1]})
contract = SchemaSpec({"x": pl.Int64}, allow_extra_columns=True, extra_columns="drop")
assert contract.fit_transform(X).columns == ["x"]
```

## SchemaValidator

```text
SchemaValidator(
    mode="strict",
    *,
    allow_extra_columns=False,
    enforce_column_order=True,
    extra_columns="preserve",
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

| Parameter | Contract |
| --- | --- |
| `mode` | `"strict"`, `"semi-strict"`, or `"permissive"`. Defaults to `"strict"`; `None` is invalid. |
| `allow_extra_columns` | Bool; allow input names not observed during fitting. All fitted names remain required. |
| `enforce_column_order` | Bool; require the relative order of fitted columns to match training. False normalizes output order. |
| `extra_columns` | `"preserve"` or `"drop"`; only affects extras that are allowed. |

Fit learns every input column's name and dtype, regardless of mode. Additional
fitted attribute: `schema_`, a dict in training column order. `output_features_`
contains those required names, excluding inference-only extras.

Transform validates structure and dtypes, collecting multiple schema mismatches
into one `SchemaValidationError`. Policies for differing dtypes:

| Mode | Accepted conversion |
| --- | --- |
| Strict | None; dtype must equal the fitted dtype. |
| Semi-strict | Integer widening based on entire dtype ranges; Float32 to Float64; integer to Float64 only when every evaluated non-null value round-trips exactly. |
| Permissive | Any successful Polars `cast(strict=True)` to the fitted dtype; successful casts can be lossy. |

Semi-strict rejects integer narrowing even when current values would fit,
Float64 to Float32, string to numeric, and Null dtype to a different dtype.
Compatible null values are preserved. Nullability is not learned. Permissive
mode can cast Null dtype columns, but invalid cast values raise rather than
becoming null silently.

With enforced order and preserved extras, output keeps input column order.
With order enforcement disabled, output uses required training order followed
by preserved extras in input order. Dropping extras returns required columns
in training order.

Lazy structural and dtype-policy failures raise during `transform`. Lazy
permissive casts execute when the plan is collected and can raise native Polars
errors. Lazy semi-strict integer-to-float value checks also run at execution and
raise `SchemaValidationError` on precision loss. An explicit eager output request
wraps Polars collection errors in `SchemaValidationError` with their causes.

```python
import polars as pl

from polars_learn import SchemaValidator

train = pl.DataFrame({"x": [1]})
incoming = pl.DataFrame({"extra": [7], "x": ["2"]})
validator = SchemaValidator(
    mode="permissive", allow_extra_columns=True, enforce_column_order=False
).fit(train.lazy())
result = validator.transform(incoming.lazy()).collect()
assert result.columns == ["x", "extra"]
assert result["x"].to_list() == [2]
assert validator.get_feature_names_out().tolist() == ["x"]
```

## FeatureSpec

```text
FeatureSpec(
    expression=None,
    *,
    expressions=None,
    alias=None,
    drop=None,
    name=None,
    verbosity=0,
)
```

| Parameter | Contract |
| --- | --- |
| `expression` | A single `pl.Expr`. Specify exactly one of this and `expressions`. |
| `expressions` | Nonempty sequence of `pl.Expr` objects evaluated together. A bare Expr or string is invalid. |
| `alias` | Optional nonempty string overriding the output name of a single expression with exactly one output. |
| `drop` | Optional sequence of nonempty string names to remove after evaluating the expressions. A bare string is invalid. |
| `name` | Optional nonempty string used to label feature execution errors and detailed progress. |
| `verbosity` | Integer 0, 1, or 2. Reports construction validation independently of an engineer's verbosity. |

Stored attributes are `expressions` (canonical tuple, including an explicit
alias), `drop` (tuple), `name`, and `verbosity`. There are no fitted attributes
or transformer methods. Pass specifications to `FeatureEngineer` for execution.

Native expression names and aliases are used unless `alias` overrides them.
Expressions with one output need a determinable name. Selectors expanding to
multiple outputs are resolved against input by the engineer. Existing columns
can be overwritten. Multiple expressions within this specification cannot use
new columns created by siblings; put dependent expressions in later specifications.

Bad expression or drop types raise `TypeError`. Conflicting/empty expression
arguments, invalid alias/name, ambiguous alias, and detectable duplicate output
names raise `ValueError`. Invalid verbosity types/levels also fail at construction.
`repr(spec)` shows its canonical expressions, drops, and name.

## FeatureEngineer

```text
FeatureEngineer(
    features,
    *,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)
```

`features` accepts a single `FeatureSpec`, a dict of `FeatureSpec` keyword
arguments, or an ordered sequence mixing these two forms. Single definitions are
normalized to one-element tuples for execution. Dicts are unpacked with
`FeatureSpec(**config)`. Generators and bare Polars expressions are unsupported.
An empty sequence is valid and produces an identity transformation. All
specifications are checked before processing input. The original parameter
object is retained unchanged for estimator cloning and `get_params`.
`set_params(features=...)` accepts the same forms; refit after changes.

Config keys are `expression`, `expressions`, `alias`, `drop`, `name`, and
`verbosity`, with the same types and validation as the FeatureSpec constructor.
Exactly one expression argument must be provided. Unknown keys fail with
`TypeError`; an empty dict or conflicting expression arguments fail with
`ValueError`. Config errors identify `features` or `features[index]` and preserve
the original exception as their cause. Real Polars expressions are required;
strings are not parsed or evaluated.

Configs are validated at construction and converted again once per fit or
transform call. This picks up parameter changes without mutating caller-owned
dicts. A config's own verbosity can emit specification construction summaries
on those conversions, independently of the engineer's verbosity. For optional
static annotations, `FeatureSpecConfig` is available from `polars_learn.types`;
it is a TypedDict describing an ordinary dict, with validation applied by the
engineer at runtime.

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

X = pl.DataFrame({"x": [1, 2]})
spec = FeatureSpec(pl.col("x") * 2, alias="doubled")
engineer = FeatureEngineer(spec)
assert engineer.get_params()["features"] is spec
assert engineer.fit_transform(X)["doubled"].to_list() == [2, 4]
```

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec
from polars_learn.types import FeatureSpecConfig

X = pl.DataFrame({"x": [1, 2]})
config: FeatureSpecConfig = {"expression": pl.col("x") * 2, "alias": "doubled"}
engineer = FeatureEngineer(
    [
        config,
        FeatureSpec(pl.col("doubled") + 1, alias="next", drop=["doubled"]),
    ]
)
assert engineer.fit_transform(X.lazy()).collect()["next"].to_list() == [3, 5]
```

Additional fitted attributes are `input_schema_` and `output_schema_`, dicts
containing input and resulting output names/dtypes in order. `output_features_`
lists all resulting columns, including retained input columns and new features.

Fit builds the ordered plan, checking each specification's expression
schema and drops. With eager input it executes the plan before recording state;
with lazy input it records schemas without evaluating rows. Transform requires
exact fitted input names and order, and an exact resulting output schema. Input
dtypes can differ only if that final schema still matches.

`fit_transform` currently calls `fit` then `transform`; eager feature expressions
are therefore evaluated twice, and pandas conversion can happen twice. Lazy
input remains unevaluated during those calls unless eager output is requested.
Feature expressions do not learn training statistics. Avoid nondeterministic
expressions when consistent training and inference are required.

Plan dependency, drop, and output-schema failures raise `FeatureEngineeringError`.
Eager execution failures are wrapped in that exception with their causes.
Execution errors from a returned lazy plan can be native Polars exceptions;
an explicit `"polars"` output request wraps them within `transform`.

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

X = pl.DataFrame({"x": [1, 2]})
engineer = FeatureEngineer(
    [
        FeatureSpec(pl.col("x") + 1, alias="a", name="increment"),
        FeatureSpec(pl.col("a") * 2, alias="b", drop=["a"], name="double"),
    ]
)
plan = engineer.fit_transform(X.lazy())
assert plan.collect().to_dict(as_series=False) == {"x": [1, 2], "b": [4, 6]}
```

## Exceptions and Warnings

Import each of these from `polars_learn`:

| Name | Base | Use |
| --- | --- | --- |
| `PolarsLearnError` | `Exception` | Base for the two package-specific processing errors. |
| `SchemaValidationError` | `PolarsLearnError` | Learned schema mismatches, eager cast failures, and lazy lossless widening failures. |
| `FeatureEngineeringError` | `PolarsLearnError` | Feature planning/schema failures and wrapped eager execution errors. |
| `PandasConversionWarning` | `UserWarning` | Enabled pandas input conversion when warnings are enabled. |

Configuration errors and fixed-contract/selector/passthrough validation use ordinary
`TypeError` or `ValueError`. Missing fitted state uses scikit-learn's
`NotFittedError`. Those exceptions, native Polars exceptions, and source errors
are not covered by catching `PolarsLearnError`. If handling lazy value errors,
place the handler around query execution as well as plan creation.

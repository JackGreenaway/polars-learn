# FeatureSpec

## About

A feature specification makes an operation explicit: the Polars expressions to evaluate together, their output names, any columns to drop, and an optional label for diagnostics. It is a lightweight description consumed by FeatureEngineer, rather than an estimator.

## Why use it?

Use it to organize a readable feature plan without creating a transformer class for each arithmetic operation. One required expression argument accepts either a single Expr or a nonempty sequence, normalized safely to a tuple.

## Example: input to output

### Input

| ask | bid |
| --- | --- |
| 11.0 | 9.0 |
| 22.0 | 18.0 |

### Code

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

quotes = pl.DataFrame({"ask": [11.0, 22.0], "bid": [9.0, 18.0]})
spec = FeatureSpec(
    [
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
    ],
    drop=["ask", "bid"],
    name="quote_features",
)
prepared = FeatureEngineer(spec).fit_transform(quotes)
assert prepared.to_dict(as_series=False) == {"spread": [2.0, 4.0], "mid": [10.0, 20.0]}
assert isinstance(spec.expression, tuple)
```

### Output

| spread | mid |
| --- | --- |
| 2.0 | 10.0 |
| 4.0 | 20.0 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Production considerations

Sibling expressions execute together; they cannot consume columns created by siblings. Put dependent expressions in later specifications. An explicit alias requires one expression with one output. Empty sequences, invalid types, and detectable duplicate output names fail during construction. Dict configs accept the same constructor keys; there is no expressions alias.

## Reference

```text
FeatureSpec(
    expression,
    *,
    alias=None,
    drop=None,
    name=None,
    verbosity=0,
)
```

| Parameter | Contract |
| --- | --- |
| `expression` | Required single `pl.Expr` or nonempty sequence of `pl.Expr` objects evaluated together. Normalized to a tuple; strings and generators are invalid. |
| `alias` | Optional nonempty string overriding the output name of a single expression with exactly one output. |
| `drop` | Optional sequence of nonempty string names to remove after evaluating the expressions. A bare string is invalid. |
| `name` | Optional nonempty string used to label feature execution errors and detailed progress. |
| `verbosity` | Integer 0, 1, or 2. Reports construction validation independently of an engineer's verbosity. |

Stored attributes are `expression` (canonical tuple, including an explicit
alias), `drop` (tuple), `name`, and `verbosity`. There are no fitted attributes
or transformer methods. Pass specifications to `FeatureEngineer` for execution.

Native expression names and aliases are used unless `alias` overrides them.
Expressions with one output need a determinable name. Selectors expanding to
multiple outputs are resolved against input by the engineer. Existing columns
can be overwritten. Multiple expressions within this specification cannot use
new columns created by siblings; put dependent expressions in later specifications.

Bad expression or drop types raise `TypeError`. Empty expression
sequences, invalid alias/name, ambiguous alias, and detectable duplicate output
names raise `ValueError`. Invalid verbosity types/levels also fail at construction.
`repr(spec)` shows its canonical expressions, drops, and name.

## Shared methods and configuration

FeatureSpec has no fit/transform methods. Its expression, drop, name, and verbosity attributes describe the operation executed by FeatureEngineer. See [FeatureEngineer](feature-engineer.md) for execution and fitted schemas.

See [persistence](../user-guide/persistence.md) for serialization and warning controls, and [diagnostics](../user-guide/diagnostics.md) for verbosity and error timing.

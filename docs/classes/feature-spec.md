# FeatureSpec

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><code>class polars_learn.FeatureSpec(expression, *, alias=None, drop=None, name=None, verbosity=0)</code></div>

Describe a native Polars feature operation.

### Parameters

<dl class="api-fields">
<dt>expression : polars.Expr or sequence of polars.Expr</dt><dd>Single expression or nonempty sequence evaluated together, normalized to a tuple. Native output names are used unless <code>alias</code> is set. Sibling expressions cannot depend on outputs created within the same specification.</dd>
<dt>alias : str or None, default=None</dt><dd>Override the output name of a single expression. Multiple expressions or an expression expanding to multiple outputs cannot use this option.</dd>
<dt>drop : sequence of str or None, default=None</dt><dd>Columns to remove after evaluating the expressions. Missing names fail during execution; later specifications cannot use dropped columns.</dd>
<dt>verbosity : {0, 1, 2}, default=0</dt><dd>Print validation summaries at 1 and output/drop details at 2.</dd>
<dt>name : str or None, default=None</dt><dd>Descriptive label included in feature execution errors.</dd>
</dl>

### Attributes

<dl class="api-fields">
<dt>expression : tuple of polars.Expr</dt><dd>Canonical expression representation, including any explicit alias.</dd>
<dt>drop : tuple of str</dt><dd>Columns to drop after evaluating this specification.</dd>
<dt>name : str or None</dt><dd>Optional debugging label.</dd>
<dt>verbosity : int</dt><dd>Configured diagnostic level, independent of the feature engineer.</dd>
</dl>

### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If expressions, drop columns, containers, or verbosity have invalid types.</dd>
<dt>ValueError</dt><dd>If the expression sequence is empty, an explicit alias is ambiguous, output names are duplicated, a label is invalid, or verbosity is outside the supported levels.</dd>
</dl>

### Notes

Existing columns may be overwritten deliberately. Expressions follow native
Polars naming, so an additional alias is not required when the expression
already has a determinable output name. Selector names and duplicates that
depend on the input schema are resolved by the feature engineer.
This class is an operation description, not a scikit-learn estimator. Its
verbosity reports construction independently of an engineer's verbosity.
Pickling preserves native expressions using Polars serialization, including
Python callbacks with serializable state. Version mismatches emit
InconsistentVersionWarning before state restoration. Compatible cross-version
expressions are attempted using JSON; incompatible state raises
SerializationError. Load trusted artifacts only and validate after upgrading.

### Methods

FeatureSpec is a description, not an estimator. It has no fit, transform, or prediction methods. Pass it to FeatureEngineer for execution.

### See also

[Shared estimator conventions](../api.md#shared-transformer-behavior) · [Production guidance](../production.md) · [Persistence](../user-guide/persistence.md)
<!-- END GENERATED REFERENCE -->

## About

A feature specification makes an operation explicit: the Polars expressions to evaluate together, their output names, any columns to drop, and an optional label for diagnostics. It is a lightweight description consumed by FeatureEngineer, rather than an estimator.

## Why use it?

Use it to organize a readable feature plan without creating a transformer class for each arithmetic operation. One required expression argument accepts either a single Expr or a nonempty sequence, normalized safely to a tuple.

## Examples

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

## Practical notes

Sibling expressions execute together; they cannot consume columns created by siblings. Put dependent expressions in later specifications. An explicit alias requires one expression with one output. Empty sequences, invalid types, and detectable duplicate output names fail during construction. Dict configs accept the same constructor keys; there is no expressions alias.

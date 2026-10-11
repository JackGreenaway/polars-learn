# FeatureSpec

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_spec.py#L267">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">FeatureSpec</span>(
    expression,
    *,
    alias=None,
    drop=None,
    name=None,
    verbosity=0,
)</code></pre></div>

Describe a native Polars feature operation.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">expression</code> <span class="api-field-type">polars.Expr or sequence of polars.Expr</span></dt><dd><p>Single expression or nonempty sequence evaluated together, normalized to a tuple. Native output names are used unless <code>alias</code> is set. Sibling expressions cannot depend on outputs created within the same specification.</p></dd>
<dt><code class="api-field-name">alias</code> <span class="api-field-type">str or None, default=None</span></dt><dd><p>Override the output name of a single expression. Multiple expressions or an expression expanding to multiple outputs cannot use this option.</p></dd>
<dt><code class="api-field-name">drop</code> <span class="api-field-type">sequence of str or None, default=None</span></dt><dd><p>Columns to remove after evaluating the expressions. Missing names fail during execution; later specifications cannot use dropped columns.</p></dd>
<dt><code class="api-field-name">verbosity</code> <span class="api-field-type">{0, 1, 2}, default=0</span></dt><dd><p>Print validation summaries at 1 and output/drop details at 2.</p></dd>
<dt><code class="api-field-name">name</code> <span class="api-field-type">str or None, default=None</span></dt><dd><p>Descriptive label included in feature execution errors.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">expression</code> <span class="api-field-type">tuple of polars.Expr</span></dt><dd><p>Canonical expression representation, including any explicit alias.</p></dd>
<dt><code class="api-field-name">drop</code> <span class="api-field-type">tuple of str</span></dt><dd><p>Columns to drop after evaluating this specification.</p></dd>
<dt><code class="api-field-name">name</code> <span class="api-field-type">str or None</span></dt><dd><p>Optional debugging label.</p></dd>
<dt><code class="api-field-name">verbosity</code> <span class="api-field-type">int</span></dt><dd><p>Configured diagnostic level, independent of the feature engineer.</p></dd>
</dl>

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If expressions, drop columns, containers, or verbosity have invalid types.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If the expression sequence is empty, an explicit alias is ambiguous, output names are duplicated, a label is invalid, or verbosity is outside the supported levels.</p></dd>
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

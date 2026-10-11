# FeatureSpec

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_spec.py#L318">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">FeatureSpec</span>(
    expression,
    *,
    alias=None,
    drop=None,
    name=None,
    verbosity=0,
)</code></pre></div>

<div class="api-class-body" markdown="1">

Describe a native Polars feature operation.

Use this class to describe a reusable feature calculation for
`FeatureEngineer`. Construction validates and stores the expressions;
it does not execute them, fit on data, or implement `transform`.
The engineer applies the expressions with Polars `with_columns`, then
removes the configured `drop` columns. Existing columns can be replaced.
Expressions within one specification read the same stage's input, so they
cannot read columns newly created by siblings. Compose expressions directly
from existing columns, or use later specifications for dependent features.
The engineer applies specifications in the order you supply them.

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

### See also

[FeatureEngineer](feature-engineer.md): Execute an ordered sequence of specifications.

### Notes

Existing columns may be overwritten deliberately. Expressions follow native
Polars naming, so an additional alias is not required when the expression
already has a determinable output name. Selector names and duplicates that
depend on the input schema are resolved by the feature engineer.
Expressions may contain map_elements or map_batches Python callbacks.
Declare return_dtype for predictable schema checks and use pure functions
with serializable state. Prefer native expressions for speed; callbacks
and their dependencies must remain usable in the deployment environment.
This class is an operation description, not a scikit-learn estimator. Its
verbosity reports construction independently of an engineer's verbosity.
Pickling preserves native expressions using Polars serialization, including
Python callbacks with serializable state. Version mismatches emit
InconsistentVersionWarning before state restoration. Compatible cross-version
expressions are attempted using JSON; incompatible state raises
SerializationError. Load trusted artifacts only and validate after upgrading.

### Examples

```pycon
>>> import polars as pl
>>> from polars_learn import FeatureSpec
>>> spec = FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional")
>>> spec.expression[0].meta.output_name()
'notional'
```

### Methods

FeatureSpec is a description, not an estimator. It has no fit, transform, or prediction methods. Pass it to FeatureEngineer for execution.

### See also

[Shared estimator conventions](../api.md#shared-transformer-behavior) · [Production guidance](../production.md) · [Persistence](../user-guide/persistence.md)

</div>
<!-- END GENERATED REFERENCE -->

## Related guide

[Concepts and worked workflows](../user-guide/feature-spec-details.md).

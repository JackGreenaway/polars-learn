# ValueValidator

`ValueValidator` applies named row-level Boolean expressions to catch nulls,
bounds, and allowed-value violations before data reaches model features. Add it
after a schema component when both structural and value contracts matter. See
the [value validation guide](../user-guide/value-validation.md) for composition
and lazy execution details.

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/value_validation.py#L16">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">ValueValidator</span>(
    checks,
    *,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)</code></pre></div>

<div class="api-class-body" markdown="1">

Reject rows that fail named Boolean Polars expressions.

Use this transformer after structural schema validation for row-level rules
such as non-null requirements, bounds, and allowed-value checks. Every check
must produce one Boolean value per row. False and null results are violations.
The input is returned unchanged when all checks pass.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">checks</code> <span class="api-field-type">dict of str to polars.Expr</span></dt><dd><p>Nonempty mapping from diagnostic labels to row-wise Boolean expressions. For example, <code>{&quot;positive_price&quot;: pl.col(&quot;price&quot;) &gt; 0}</code>.</p></dd>
<dt><code class="api-field-name">convert_pandas</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Allow pandas DataFrames and convert them to Polars.</p></dd>
<dt><code class="api-field-name">warn_on_conversion</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Emit <code>PandasConversionWarning</code> when converting pandas input.</p></dd>
<dt><code class="api-field-name">verbosity</code> <span class="api-field-type">{0, 1, 2}, default=0</span></dt><dd><p>Print progress at 1 and schema/check details at 2; 0 is silent.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray of str, dtype=object</span></dt><dd><p>Input column names and order recorded during fit.</p></dd>
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Number of input columns recorded during fit.</p></dd>
<dt><code class="api-field-name">output_features_</code> <span class="api-field-type">ndarray of str, dtype=object</span></dt><dd><p>Input column names, unchanged by validation.</p></dd>
</dl>

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If checks, expressions, or common options have invalid types.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If checks are empty, labels are invalid, or expressions do not resolve to Boolean values on the fitted schema.</p></dd>
<dt><code class="api-field-name">ValueValidationError</code></dt><dd><p>If an input schema differs from fitting, checks cannot execute, or any check returns false or null for one or more rows.</p></dd>
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If transform is called before a successful fit.</p></dd>
</dl>

### Notes

`fit` resolves the check expressions against schema metadata only; it never
evaluates row values. `transform` evaluates the checks once. Lazy inputs
are materialized during transform so rule violations fail before it returns
and default output can preserve the lazy container using an in-memory frame.
Value checks do not add columns, mutate input, or change row order.

### Examples

```pycon
>>> import polars as pl
>>> from polars_learn import ValueValidator
>>> frame = pl.DataFrame({"price": [2.0, 3.0], "quantity": [1, 2]})
>>> checks = {
...     "positive_price": pl.col("price") > 0,
...     "nonnegative_quantity": pl.col("quantity") >= 0,
... }
>>> validator = ValueValidator(checks)
>>> validator.fit_transform(frame).equals(frame)
True
```

### Methods

| Method | Purpose |
| --- | --- |
| [fit(X, y=None)](#fit) | Resolve checks against the input schema without evaluating rows. |
| [transform(X)](#transform) | Validate fitted input names, order, dtypes, and configured values. |
| [fit_transform(X, y=None, **fit_params)](#fit-transform) | Fit the schema contract, validate values, and return the input. |
| [get_feature_names_out(input_features=None)](#get-feature-names-out) | Return the fitted output column names. |
| [get_params(deep=True)](#get-params) | Get estimator constructor parameters. |
| [set_params(**params)](#set-params) | Set constructor parameters using scikit-learn conventions. |
| [set_output(*, transform=None)](#set-output) | Configure native Polars output or pandas compatibility output. |

### fit {#fit .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/value_validation.py#L171">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(X, y=None)</code></pre></div>

Resolve checks against the input schema without evaluating rows.

</div>

### transform {#transform .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/value_validation.py#L188">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">transform</span>(X)</code></pre></div>

Validate fitted input names, order, dtypes, and configured values.

</div>

### fit_transform {#fit-transform .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/value_validation.py#L176">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit_transform</span>(
    X,
    y=None,
    **fit_params,
)</code></pre></div>

Fit the schema contract, validate values, and return the input.

</div>

### get_feature_names_out {#get-feature-names-out .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/_utils.py#L170">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_feature_names_out</span>(input_features=None)</code></pre></div>

Return the fitted output column names.

Use these names to label model inputs or inspect a fitted pipeline.
Returns a copy of the names recorded during fitting; it does not
transform data, collect lazy rows, or refit the transformer. Supplied
input names are checked against all fitted input names in exact order.
For SchemaValidator, inference-only extra columns are not included;
this is the fitted contract rather than a live query's column list.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">input_features</code> <span class="api-field-type">array-like of str or None, default=None</span></dt><dd><p>If supplied, names must exactly match <code>feature_names_in_</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">feature_names_out</code> <span class="api-field-type">ndarray of shape (n_features_out_,), dtype=object</span></dt><dd><p>Copy of fitted output names. SchemaValidator excludes inference-only extras. SchemaSpec includes preserved training extras. A selector lists selected names, including any renames; a passthrough lists unchanged selected names. An engineer lists all output names.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If the transformer has not been fitted.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If supplied input names differ from fitted names or order.</p></dd>
</dl>

</div>

### get_params {#get-params .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_params</span>(deep=True)</code></pre></div>

Get estimator constructor parameters.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">deep</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Include nested estimator parameters when present.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">params</code> <span class="api-field-type">dict</span></dt><dd><p>Constructor parameters and, when requested, nested parameters. Output configuration is managed separately by set_output.</p></dd>
</dl>

</div>

### set_params {#set-params .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_params">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_params</span>(**params)</code></pre></div>

Set constructor parameters using scikit-learn conventions.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">**params</code> <span class="api-field-type">dict</span></dt><dd><p>Constructor parameter names and values. Nested estimator parameters use the step__parameter convention where supported.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">estimator instance</span></dt><dd><p>This estimator with updated configuration.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If a parameter name is unknown.</p></dd>
</dl>

#### Notes

Changing parameters does not immediately validate configuration or clear
fitted attributes. Refit after contract changes. Invalid parameter values
fail at the next relevant fit or transform operation.

</div>

### set_output {#set-output .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/_utils.py#L210">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_output</span>(*, transform=None)</code></pre></div>

Configure native Polars output or pandas compatibility output.

Call this when a downstream consumer needs a particular frame container.
It only stores a choice for subsequent transform/fit_transform calls;
it does not fit, transform, or collect data at configuration time.
Use "default" to follow each call's input kind, "polars" for eager
Polars, "polars-lazy" for a lazy plan, or "pandas" for an eager pandas
DataFrame. Converted pandas input is treated as eager Polars by default.
Pandas input permission remains separate: enable convert_pandas on any
transformer receiving pandas, even when pandas output is requested.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">transform</code> <span class="api-field-type">{&quot;default&quot;, &quot;polars&quot;, &quot;polars-lazy&quot;, &quot;pandas&quot;} or None, default=None</span></dt><dd><p>Default preserves the input kind. &quot;polars&quot; collects an eager DataFrame; &quot;polars-lazy&quot; returns a LazyFrame. <code>None</code> leaves the configuration unchanged. &quot;pandas&quot; collects if necessary and converts the result to a pandas DataFrame. Requires the pandas extra.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">estimator instance</span></dt><dd><p>This transformer.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If an unsupported output container is requested.</p></dd>
</dl>

#### Notes

Prefer native Polars output for speed and to avoid conversion overhead.
Pandas output is provided for compatibility and uses Polars to_pandas
defaults, including a fresh RangeIndex and dtype/null conversion.
Conversion raises ImportError if pandas or PyArrow is unavailable.
Input conversion remains separately controlled by convert_pandas.
Automatic scikit-learn output wrapping and global output settings are
bypassed to preserve native names and permitted extra columns.
Output choices survive cloning. None leaves the current choice intact;
"default" resets input-kind preservation. Fit always returns self and
does not collect lazy rows. Forcing lazy output from eager input may
wrap an already evaluated result; use lazy input to defer evaluation.

</div>

### See also

[Shared estimator conventions](../api.md#shared-transformer-behavior) · [Production guidance](../production.md) · [Persistence](../user-guide/persistence.md)

</div>
<!-- END GENERATED REFERENCE -->

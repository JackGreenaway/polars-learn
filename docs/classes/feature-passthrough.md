# FeaturePassthrough

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/passthrough.py#L8">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">FeaturePassthrough</span>(
    columns,
    *,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)</code></pre></div>

Pass explicitly selected columns through without changing their contents.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">columns</code> <span class="api-field-type">sequence of str</span></dt><dd><p>Nonempty ordered sequence of unique, nonempty column names to keep. Other input columns are omitted. A bare string is invalid.</p></dd>
<dt><code class="api-field-name">convert_pandas</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Permit pandas DataFrames and convert them to Polars. Some dtypes require PyArrow; install the pandas extra for both dependencies.</p></dd>
<dt><code class="api-field-name">warn_on_conversion</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Emit PandasConversionWarning on enabled pandas conversion, independently of verbosity.</p></dd>
<dt><code class="api-field-name">verbosity</code> <span class="api-field-type">{0, 1, 2}, default=0</span></dt><dd><p>Print progress at 1 and additional schema details at 2; 0 is silent.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray of str, dtype=object</span></dt><dd><p>All training input names in input order, including unselected columns.</p></dd>
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Number of all training input columns.</p></dd>
<dt><code class="api-field-name">columns_</code> <span class="api-field-type">tuple of str</span></dt><dd><p>Selected source names recorded by fit.</p></dd>
<dt><code class="api-field-name">output_features_</code> <span class="api-field-type">ndarray of str, dtype=object</span></dt><dd><p>Selected column names in the requested output order.</p></dd>
</dl>

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If columns or common options have invalid types.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If selected names are duplicated or verbosity is outside 0, 1, and 2. Fit and transform also reject missing selected columns.</p></dd>
</dl>

### Notes

This is a scikit-learn transformer with inherited fit, transform,
fit_transform, get_feature_names_out, get_params, set_params, and set_output.
Fit checks that selected columns exist; transform requires fitting.
Selected Polars values, names, dtypes, nulls, and row order are preserved.
Input order and dtypes may change between calls; no dtype validation occurs.
There is no renaming option. Refit after changing selected columns.

Default returns preserve input kind: DataFrame input returns DataFrame and
LazyFrame input returns an unevaluated LazyFrame. set_output can force
"polars" (eager) or "polars-lazy" output. Inputs are never mutated.
Source/schema-resolution and collection errors are not wrapped.

A sequential pipeline step returns only its selected columns. To retain
them alongside columns transformed in another branch, use ColumnTransformer
with eager input, a branch for each operation, and Polars output.

### Methods

| Method | Purpose |
| --- | --- |
| [fit(X, y=None)](#fit) | Validate selected columns and record feature names. |
| [transform(X)](#transform) | Select fitted columns unchanged in configured order. |
| [fit_transform(X, y=None, **fit_params)](#fit-transform) | Fit and transform the same input using the configured return kind. |
| [get_feature_names_out(input_features=None)](#get-feature-names-out) | Return the fitted output column names. |
| [get_params(deep=True)](#get-params) | Get estimator constructor parameters. |
| [set_params(**params)](#set-params) | Set constructor parameters using scikit-learn conventions. |
| [set_output(*, transform=None)](#set-output) | Configure native output without automatic container conversion. |

### fit {#fit .api-method-title}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_spec.py#L144">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(X, y=None)</code></pre></div>

Validate selected columns and record feature names.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Input containing all selected columns. Other columns are omitted from output. Input order and dtypes may differ. Pandas requires convert_pandas=True.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">object, default=None</span></dt><dd><p>Ignored; accepted for estimator compatibility.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">FeaturePassthrough</span></dt><dd><p>The fitted transformer.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>Unsupported input, column, or option types.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>Missing selected columns, duplicate names, or invalid configuration.</p></dd>
</dl>

#### Notes

Lazy metadata checks do not evaluate rows. Source and collection failures propagate from Polars. Dtypes are not validated.

### transform {#transform .api-method-title}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_spec.py#L217">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">transform</span>(X)</code></pre></div>

Select fitted columns unchanged in configured order.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Input containing all selected columns. Other columns are omitted from output. Input order and dtypes may differ. Pandas requires convert_pandas=True.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">X_selected</code> <span class="api-field-type">polars.DataFrame or polars.LazyFrame</span></dt><dd><p>Selected unchanged columns, preserving values, names, dtypes, nulls, and row order. Output follows set_output.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>Unsupported input, column, or option types.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>Missing selected columns, duplicate names, or invalid configuration.</p></dd>
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>Fit has not completed successfully.</p></dd>
</dl>

#### Notes

Lazy metadata checks do not evaluate rows. Source and collection failures propagate from Polars. Dtypes are not validated.

### fit_transform {#fit-transform .api-method-title}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_spec.py#L188">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit_transform</span>(
    X,
    y=None,
    **fit_params,
)</code></pre></div>

Fit and transform the same input using the configured return kind.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Input accepted by fit and transform. Pandas requires conversion enabled.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">object, default=None</span></dt><dd><p>Ignored; accepted for scikit-learn compatibility.</p></dd>
<dt><code class="api-field-name">**fit_params</code> <span class="api-field-type">dict</span></dt><dd><p>Extra fit parameters are unsupported and raise TypeError.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">X_transformed</code> <span class="api-field-type">polars.DataFrame or polars.LazyFrame</span></dt><dd><p>Transformed input. Default preserves input kind; set_output can force an eager DataFrame or LazyFrame. Converted pandas is eager.</p></dd>
</dl>

### get_feature_names_out {#get-feature-names-out .api-method-title}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/_utils.py#L162">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_feature_names_out</span>(input_features=None)</code></pre></div>

Return the fitted output column names.

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

### get_params {#get-params .api-method-title}

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

### set_params {#set-params .api-method-title}

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

### set_output {#set-output .api-method-title}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/_utils.py#L195">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_output</span>(*, transform=None)</code></pre></div>

Configure native output without automatic container conversion.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">transform</code> <span class="api-field-type">{&quot;default&quot;, &quot;polars&quot;, &quot;polars-lazy&quot;} or None, default=None</span></dt><dd><p>Default preserves the input kind. &quot;polars&quot; collects an eager DataFrame; &quot;polars-lazy&quot; returns a LazyFrame. <code>None</code> leaves the configuration unchanged. Pandas output is unsupported.</p></dd>
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

Automatic scikit-learn output wrapping and global output settings are
bypassed to preserve native names and permitted extra columns.
Output choices survive cloning. None leaves the current choice intact;
"default" resets input-kind preservation. Fit always returns self and
does not collect lazy rows. Forcing lazy output from eager input may
wrap an already evaluated result; use lazy input to defer evaluation.

### See also

[Shared estimator conventions](../api.md#shared-transformer-behavior) · [Production guidance](../production.md) · [Persistence](../user-guide/persistence.md)
<!-- END GENERATED REFERENCE -->

## About

An explicit unchanged-feature step selects named columns while preserving their Polars values, names, dtypes, nulls, and row order. It communicates the selection contract more clearly than an anonymous passthrough string.

## Why use it?

Use it for features that need no transformation, particularly a ColumnTransformer branch alongside transformed features. In a sequential pipeline it returns only the named columns, so the next step receives that subset.

## Examples

### Input

| id (Int64) | flag (Boolean) | unused (Int64) |
| --- | --- | --- |
| 7 | true | 9 |
| null | false | 8 |

### Code

```python
import polars as pl

from polars_learn import FeaturePassthrough

raw = pl.DataFrame({"id": [7, None], "flag": [True, False], "unused": [9, 8]})
keep = FeaturePassthrough(["flag", "id"])
prepared = keep.fit_transform(raw)
assert prepared.equals(raw.select("flag", "id"))
assert prepared.to_dict(as_series=False) == {"flag": [True, False], "id": [7, None]}
assert dict(prepared.schema) == {"flag": pl.Boolean, "id": pl.Int64}
```

### Output

| flag (Boolean) | id (Int64) |
| --- | --- |
| true | 7 |
| false | null |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

There is no rename or cast option. Input order and dtypes may change without rejection; add schema validation when those changes must fail. See [parallel branch examples](../user-guide/selection.md) for combining unchanged and transformed features.

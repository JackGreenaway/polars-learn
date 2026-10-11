# FeaturePassthrough

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><code>class polars_learn.FeaturePassthrough(columns, *, convert_pandas=False, warn_on_conversion=True, verbosity=0)</code></div>

Pass explicitly selected columns through without changing their contents.

### Parameters

<dl class="api-fields">
<dt>columns : sequence of str</dt><dd>Nonempty ordered sequence of unique, nonempty column names to keep. Other input columns are omitted. A bare string is invalid.</dd>
<dt>convert_pandas : bool, default=False</dt><dd>Permit pandas DataFrames and convert them to Polars. Some dtypes require PyArrow; install the pandas extra for both dependencies.</dd>
<dt>warn_on_conversion : bool, default=True</dt><dd>Emit PandasConversionWarning on enabled pandas conversion, independently of verbosity.</dd>
<dt>verbosity : {0, 1, 2}, default=0</dt><dd>Print progress at 1 and additional schema details at 2; 0 is silent.</dd>
</dl>

### Attributes

<dl class="api-fields">
<dt>feature_names_in_ : ndarray of str, dtype=object</dt><dd>All training input names in input order, including unselected columns.</dd>
<dt>n_features_in_ : int</dt><dd>Number of all training input columns.</dd>
<dt>columns_ : tuple of str</dt><dd>Selected source names recorded by fit.</dd>
<dt>output_features_ : ndarray of str, dtype=object</dt><dd>Selected column names in the requested output order.</dd>
</dl>

### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If columns or common options have invalid types.</dd>
<dt>ValueError</dt><dd>If selected names are duplicated or verbosity is outside 0, 1, and 2. Fit and transform also reject missing selected columns.</dd>
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

### fit {#fit}

<div class="api-signature"><code>fit(X, y=None)</code></div>

Validate selected columns and record feature names.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input containing all selected columns. Other columns are omitted from output. Input order and dtypes may differ. Pandas requires convert_pandas=True.</dd>
<dt>y : object, default=None</dt><dd>Ignored; accepted for estimator compatibility.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>self : FeaturePassthrough</dt><dd>The fitted transformer.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>Unsupported input, column, or option types.</dd>
<dt>ValueError</dt><dd>Missing selected columns, duplicate names, or invalid configuration.</dd>
</dl>

#### Notes

Lazy metadata checks do not evaluate rows. Source and collection failures propagate from Polars. Dtypes are not validated.

### transform {#transform}

<div class="api-signature"><code>transform(X)</code></div>

Select fitted columns unchanged in configured order.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input containing all selected columns. Other columns are omitted from output. Input order and dtypes may differ. Pandas requires convert_pandas=True.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>X_selected : polars.DataFrame or polars.LazyFrame</dt><dd>Selected unchanged columns, preserving values, names, dtypes, nulls, and row order. Output follows set_output.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>Unsupported input, column, or option types.</dd>
<dt>ValueError</dt><dd>Missing selected columns, duplicate names, or invalid configuration.</dd>
<dt>sklearn.exceptions.NotFittedError</dt><dd>Fit has not completed successfully.</dd>
</dl>

#### Notes

Lazy metadata checks do not evaluate rows. Source and collection failures propagate from Polars. Dtypes are not validated.

### fit_transform {#fit-transform}

<div class="api-signature"><code>fit_transform(X, y=None, **fit_params)</code></div>

Fit and transform the same input using the configured return kind.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input accepted by fit and transform. Pandas requires conversion enabled.</dd>
<dt>y : object, default=None</dt><dd>Ignored; accepted for scikit-learn compatibility.</dd>
<dt>**fit_params : dict</dt><dd>Extra fit parameters are unsupported and raise TypeError.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>X_transformed : polars.DataFrame or polars.LazyFrame</dt><dd>Transformed input. Default preserves input kind; set_output can force an eager DataFrame or LazyFrame. Converted pandas is eager.</dd>
</dl>

### get_feature_names_out {#get-feature-names-out}

<div class="api-signature"><code>get_feature_names_out(input_features=None)</code></div>

Return the fitted output column names.

#### Parameters

<dl class="api-fields">
<dt>input_features : array-like of str or None, default=None</dt><dd>If supplied, names must exactly match <code>feature_names_in_</code>.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>feature_names_out : ndarray of shape (n_features_out_,), dtype=object</dt><dd>Copy of fitted output names. SchemaValidator excludes inference-only extras. SchemaSpec includes preserved training extras. A selector lists selected names, including any renames; a passthrough lists unchanged selected names. An engineer lists all output names.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>sklearn.exceptions.NotFittedError</dt><dd>If the transformer has not been fitted.</dd>
<dt>ValueError</dt><dd>If supplied input names differ from fitted names or order.</dd>
</dl>

### get_params {#get-params}

<div class="api-signature"><code>get_params(deep=True)</code></div>

Get estimator constructor parameters.

#### Parameters

<dl class="api-fields">
<dt>deep : bool, default=True</dt><dd>Include nested estimator parameters when present.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>params : dict</dt><dd>Constructor parameters and, when requested, nested parameters. Output configuration is managed separately by set_output.</dd>
</dl>

### set_params {#set-params}

<div class="api-signature"><code>set_params(**params)</code></div>

Set constructor parameters using scikit-learn conventions.

#### Parameters

<dl class="api-fields">
<dt>**params : dict</dt><dd>Constructor parameter names and values. Nested estimator parameters use the step__parameter convention where supported.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>self : estimator instance</dt><dd>This estimator with updated configuration.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>ValueError</dt><dd>If a parameter name is unknown.</dd>
</dl>

#### Notes

Changing parameters does not immediately validate configuration or clear
fitted attributes. Refit after contract changes. Invalid parameter values
fail at the next relevant fit or transform operation.

### set_output {#set-output}

<div class="api-signature"><code>set_output(*, transform=None)</code></div>

Configure native output without automatic container conversion.

#### Parameters

<dl class="api-fields">
<dt>transform : {&quot;default&quot;, &quot;polars&quot;, &quot;polars-lazy&quot;} or None, default=None</dt><dd>Default preserves the input kind. &quot;polars&quot; collects an eager DataFrame; &quot;polars-lazy&quot; returns a LazyFrame. <code>None</code> leaves the configuration unchanged. Pandas output is unsupported.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>self : estimator instance</dt><dd>This transformer.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>ValueError</dt><dd>If an unsupported output container is requested.</dd>
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

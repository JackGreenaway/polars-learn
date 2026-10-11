# ColumnSelector

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><code>class polars_learn.ColumnSelector(columns, *, rename=None, convert_pandas=False, warn_on_conversion=True, verbosity=0)</code></div>

Select and optionally rename columns in a Polars DataFrame or LazyFrame.

### Parameters

<dl class="api-fields">
<dt>columns : sequence of str</dt><dd>Nonempty ordered sequence of unique, nonempty input column names to keep.</dd>
<dt>rename : dict of str to str, default=None</dt><dd>Optional mapping from selected source names to nonempty output names. Final output names, including unchanged names, must remain unique.</dd>
<dt>convert_pandas : bool, default=False</dt><dd>Allow pandas DataFrames and convert them to Polars. Requires pandas; some dtypes also require PyArrow.</dd>
<dt>verbosity : {0, 1, 2}, default=0</dt><dd>Print progress at 1 and schema/operation details at 2; 0 is silent.</dd>
<dt>warn_on_conversion : bool, default=True</dt><dd>Emit <code>PandasConversionWarning</code> when converting pandas input.</dd>
</dl>

### Attributes

<dl class="api-fields">
<dt>feature_names_in_ : ndarray of str, dtype=object</dt><dd>All training input column names in input order, including unselected names.</dd>
<dt>n_features_in_ : int</dt><dd>Number of all training input columns.</dd>
<dt>columns_ : tuple of str</dt><dd>Selected source names recorded by fit.</dd>
<dt>output_features_ : ndarray of str, dtype=object</dt><dd>Selected output names after renaming, in selection order.</dd>
</dl>

### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If columns, rename, or common options have invalid types.</dd>
<dt>ValueError</dt><dd>If selected or final renamed names are duplicated, rename keys are unselected, or verbosity is outside 0, 1, and 2.</dd>
</dl>

### Notes

DataFrames return DataFrames and LazyFrames return unevaluated LazyFrames
by default. Use set_output to force an eager or lazy output container.

### Methods

| Method | Purpose |
| --- | --- |
| [fit(X, y=None)](#fit) | Check selected columns and record input and output names. |
| [transform(X)](#transform) | Select fitted columns, rename them, and verify output names. |
| [fit_transform(X, y=None, **fit_params)](#fit-transform) | Fit and transform the same input using the configured return kind. |
| [get_feature_names_out(input_features=None)](#get-feature-names-out) | Return the fitted output column names. |
| [get_params(deep=True)](#get-params) | Get estimator constructor parameters. |
| [set_params(**params)](#set-params) | Set constructor parameters using scikit-learn conventions. |
| [set_output(*, transform=None)](#set-output) | Configure native output without automatic container conversion. |

### fit {#fit}

<div class="api-signature"><code>fit(X, y=None)</code></div>

Check selected columns and record input and output names.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input containing every selected column. Other columns and a different input order are allowed. Pandas requires conversion enabled.</dd>
<dt>y : object, default=None</dt><dd>Ignored; accepted for scikit-learn compatibility.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>self : ColumnSelector</dt><dd>Fitted selector. Dtypes are not learned or validated.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If input, options, or column/rename parameter types are unsupported.</dd>
<dt>ValueError</dt><dd>If configuration is invalid or selected columns are missing.</dd>
</dl>

#### Notes

Lazy input is checked through schema metadata without collecting rows.
Output configuration does not change the return from fit.

### transform {#transform}

<div class="api-signature"><code>transform(X)</code></div>

Select fitted columns, rename them, and verify output names.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input containing the selected fitted columns. Unselected columns are ignored; input order and dtypes may differ from fitting. Pandas requires conversion enabled.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>X_selected : polars.DataFrame or polars.LazyFrame</dt><dd>Selected columns in configured order, including renames. Default preserves input kind; set_output can force eager or lazy output.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>sklearn.exceptions.NotFittedError</dt><dd>If fit has not completed successfully.</dd>
<dt>TypeError</dt><dd>If input or parameter types are unsupported.</dd>
<dt>ValueError</dt><dd>If selected columns are missing, configuration is invalid, or renamed output names differ from fitting.</dd>
</dl>

#### Notes

Lazy metadata errors occur now; source/value errors can occur when a
returned plan is collected. Explicit eager output collects within this
method. Source and collection errors are not wrapped.

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

A named selection step establishes the exact feature subset and order, with optional renaming. It makes the model-facing layout visible in the pipeline instead of leaving selection to an external helper.

## Why use it?

Use it to remove unused raw or intermediate columns, reorder features deliberately, or adapt names for downstream consumers. Use FeaturePassthrough when selection should communicate that names and values remain unchanged.

## Examples

### Input

| price | quantity | request_id |
| --- | --- | --- |
| 10.0 | 2 | 7 |
| 20.0 | 3 | 8 |

### Code

```python
import polars as pl

from polars_learn import ColumnSelector

raw = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3], "request_id": [7, 8]})
selector = ColumnSelector(["quantity", "price"], rename={"quantity": "qty"})
prepared = selector.fit_transform(raw)
assert prepared.to_dict(as_series=False) == {"qty": [2, 3], "price": [10.0, 20.0]}
assert selector.get_feature_names_out().tolist() == ["qty", "price"]
```

### Output

| qty | price |
| --- | --- |
| 2 | 10.0 |
| 3 | 20.0 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

Selection does not validate dtype drift or cast values. Pair it with a schema step when dtype enforcement matters. Selected columns must exist, and output names must be unique. Unselected columns are omitted.

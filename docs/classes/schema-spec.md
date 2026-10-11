# SchemaSpec

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><code>class polars_learn.SchemaSpec(schema, *, allow_extra_columns=False, enforce_column_order=True, extra_columns=&#x27;preserve&#x27;, convert_pandas=False, warn_on_conversion=True, verbosity=0)</code></div>

Describe a fixed contract for Polars DataFrames and LazyFrames.

### Parameters

<dl class="api-fields">
<dt>schema : dict of str to polars.DataType</dt><dd>Required input column names and dtypes in order.</dd>
<dt>allow_extra_columns : bool, default=False</dt><dd>Whether additional, non-required columns are accepted.</dd>
<dt>enforce_column_order : bool, default=True</dt><dd>Require the relative order of required columns to match schema key order.</dd>
<dt>extra_columns : {&quot;preserve&quot;, &quot;drop&quot;}, default=&quot;preserve&quot;</dt><dd>If extra columns are permitted, whether to keep or drop them in output.</dd>
<dt>convert_pandas : bool, default=False</dt><dd>Allow pandas DataFrames and convert them to Polars. Requires pandas; some dtypes also require PyArrow.</dd>
<dt>verbosity : {0, 1, 2}, default=0</dt><dd>Print progress at 1 and schema/operation details at 2; 0 is silent.</dd>
<dt>warn_on_conversion : bool, default=True</dt><dd>Emit <code>PandasConversionWarning</code> when converting pandas input.</dd>
</dl>

### Attributes

<dl class="api-fields">
<dt>schema_ : dict of str to polars.DataType</dt><dd>Copy of the required explicit schema recorded by fit.</dd>
<dt>feature_names_in_ : ndarray of str, dtype=object</dt><dd>All training input column names in input order, including extras.</dd>
<dt>n_features_in_ : int</dt><dd>Number of all training input columns.</dd>
<dt>output_features_ : ndarray of str, dtype=object</dt><dd>Fitted output names, including preserved training extras.</dd>
</dl>

### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If schema keys/values or common options have invalid types.</dd>
<dt>ValueError</dt><dd>If schema is empty or not a dict, or an option has an invalid value.</dd>
</dl>

### Notes

DataFrames return DataFrames and LazyFrames return unevaluated LazyFrames
by default. Use set_output to force an eager or lazy output container.
Required dtypes must match exactly; there is no casting. Required output
columns follow schema key order, then preserved extras follow input order.
Transform requires fitting and exact fitted output names/order, including
preserved extras. Extra dtypes are not validated.

### Methods

| Method | Purpose |
| --- | --- |
| [fit(X, y=None)](#fit) | Validate the explicit contract and record fitted output names. |
| [transform(X)](#transform) | Check exact required dtypes and return the fitted output columns. |
| [fit_transform(X, y=None, **fit_params)](#fit-transform) | Fit and transform the same input using the configured return kind. |
| [get_feature_names_out(input_features=None)](#get-feature-names-out) | Return the fitted output column names. |
| [get_params(deep=True)](#get-params) | Get estimator constructor parameters. |
| [set_params(**params)](#set-params) | Set constructor parameters using scikit-learn conventions. |
| [set_output(*, transform=None)](#set-output) | Configure native output without automatic container conversion. |

### fit {#fit}

<div class="api-signature"><code>fit(X, y=None)</code></div>

Validate the explicit contract and record fitted output names.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input with all required columns and exact required dtypes. Extra columns and ordering follow the constructor&#x27;s policies. Pandas requires conversion enabled.</dd>
<dt>y : object, default=None</dt><dd>Ignored; accepted for scikit-learn compatibility.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>self : SchemaSpec</dt><dd>Fitted contract. The provided schema is checked, not inferred anew.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If input, option types, or schema keys/values are unsupported.</dd>
<dt>ValueError</dt><dd>If the schema or input contract is invalid.</dd>
</dl>

#### Notes

Lazy input is checked through schema metadata without collecting rows.
Fitted output names include preserved training extras. No casts occur.

### transform {#transform}

<div class="api-signature"><code>transform(X)</code></div>

Check exact required dtypes and return the fitted output columns.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input satisfying the explicit required schema and configured extra-column/order policies. Pandas requires conversion enabled.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>X_validated : polars.DataFrame or polars.LazyFrame</dt><dd>Required columns in schema order, then preserved extras in input order. Default preserves input kind; set_output can force the kind.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>sklearn.exceptions.NotFittedError</dt><dd>If fit has not completed successfully.</dd>
<dt>TypeError</dt><dd>If input or parameter types are unsupported.</dd>
<dt>ValueError</dt><dd>If the contract fails, configuration is invalid, or resulting output names/order differ from fitting, including preserved extra names.</dd>
</dl>

#### Notes

No casting occurs; extra dtypes are not checked. Lazy validation uses
metadata. Explicit eager output collects the result; source/collection
errors are not wrapped.

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

A declared schema is an executable input contract. SchemaSpec checks that required names, dtypes, and order match your application definition, instead of inferring the definition from whichever training frame happens to arrive.

## Why use it?

Use it when the upstream interface is known in advance, such as a documented scoring request schema. Allow extra transport metadata while dropping it from model input, or reject extras entirely.

## Examples

### Input

| request_id (Int64) | price (Float64) | quantity (Int64) |
| --- | --- | --- |
| 7 | 10.0 | 2 |
| 8 | 20.0 | 3 |

### Code

```python
import polars as pl

from polars_learn import SchemaSpec

raw = pl.DataFrame({"request_id": [7, 8], "price": [10.0, 20.0], "quantity": [2, 3]})
contract = SchemaSpec(
    {"price": pl.Float64, "quantity": pl.Int64},
    allow_extra_columns=True,
    extra_columns="drop",
)
prepared = contract.fit_transform(raw)
assert prepared.equals(raw.select("price", "quantity"))
assert contract.schema_ == {"price": pl.Float64, "quantity": pl.Int64}
assert contract.get_feature_names_out().tolist() == ["price", "quantity"]
```

### Output

| price (Float64) | quantity (Int64) |
| --- | --- |
| 10.0 | 2 |
| 20.0 | 3 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

This class does not cast: wrong required dtypes are errors. It validates names and types, not nullability, value ranges, or business rules. Use SchemaValidator for a learned schema and explicit cast policy.

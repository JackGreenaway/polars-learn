# SchemaValidator

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><code>class polars_learn.SchemaValidator(mode=&#x27;strict&#x27;, *, allow_extra_columns=False, enforce_column_order=True, extra_columns=&#x27;preserve&#x27;, convert_pandas=False, warn_on_conversion=True, verbosity=0)</code></div>

Learn a schema and validate subsequent Polars DataFrames or LazyFrames.

### Parameters

<dl class="api-fields">
<dt>mode : {&quot;strict&quot;, &quot;semi-strict&quot;, &quot;permissive&quot;}, default=&quot;strict&quot;</dt><dd>Validation policy. Strict mode requires identical dtypes. Semi-strict allows only the documented lossless widening policy. Permissive mode attempts Polars casts with strict=True. Successful casts can be lossy; failure occurs immediately for eager input or at lazy execution.</dd>
<dt>allow_extra_columns : bool, default=False</dt><dd>Allow columns not present during fitting. Missing required columns always raise, regardless of the validation mode.</dd>
<dt>enforce_column_order : bool, default=True</dt><dd>Require the relative order of required columns to match training. If false, output uses fitted column order followed by preserved extras.</dd>
<dt>extra_columns : {&quot;preserve&quot;, &quot;drop&quot;}, default=&quot;preserve&quot;</dt><dd>Treatment of permitted extras. This option does not itself permit extra columns; <code>allow_extra_columns</code> must also be true.</dd>
<dt>convert_pandas : bool, default=False</dt><dd>Allow pandas DataFrames and convert them to Polars. Requires pandas; some dtypes also require PyArrow.</dd>
<dt>verbosity : {0, 1, 2}, default=0</dt><dd>Print progress at 1 and schema/operation details at 2; 0 is silent.</dd>
<dt>warn_on_conversion : bool, default=True</dt><dd>Emit <code>PandasConversionWarning</code> when converting pandas input.</dd>
</dl>

### Attributes

<dl class="api-fields">
<dt>schema_ : dict of str to polars.DataType</dt><dd>Learned required columns and dtypes, in training order.</dd>
<dt>n_features_in_ : int</dt><dd>Number of training columns.</dd>
<dt>feature_names_in_ : ndarray of shape (n_features_in_,), dtype=object</dt><dd>Required input names in training order.</dd>
<dt>output_features_ : ndarray of shape (n_features_in_,), dtype=object</dt><dd>Fitted required names. Inference-only extras are not included.</dd>
</dl>

### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If common options have invalid types.</dd>
<dt>ValueError</dt><dd>If mode, extra_columns, or verbosity is invalid.</dd>
</dl>

### Notes

Semi-strict allows integer widening when the source dtype's entire range
fits inside the target dtype, Float32 to Float64, and integer to Float64
only when every non-null value round-trips exactly. It rejects string to
numeric conversion, narrowing, and Null dtype to another dtype. Null values
in compatible columns are preserved; nullability constraints are not learned.
Fit always learns a fresh schema; transform always validates against it.
Input data and schemas owned by other transformers are never mutated.
Lazy inputs are validated using metadata without collecting rows. Cast
failures and value-dependent integer-to-float checks run at collection.
Explicit eager output wraps Polars cast errors in SchemaValidationError;
collecting a returned LazyFrame can raise native Polars exceptions.
Default output preserves input kind. set_output can force "polars" (eager)
or "polars-lazy" output. With enforced order and preserved extras, output
keeps input order; disabling order enforcement puts required columns first.

### Methods

| Method | Purpose |
| --- | --- |
| [fit(X, y=None)](#fit) | Learn column names, order, and dtypes without changing the input. |
| [transform(X)](#transform) | Validate the learned schema and normalise permitted differences. |
| [fit_transform(X, y=None, **fit_params)](#fit-transform) | Fit to training data and return its transformed representation. |
| [get_feature_names_out(input_features=None)](#get-feature-names-out) | Return the fitted output column names. |
| [get_params(deep=True)](#get-params) | Get estimator constructor parameters. |
| [set_params(**params)](#set-params) | Set constructor parameters using scikit-learn conventions. |
| [set_output(*, transform=None)](#set-output) | Configure native output without automatic container conversion. |

### fit {#fit}

<div class="api-signature"><code>fit(X, y=None)</code></div>

Learn column names, order, and dtypes without changing the input.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Training data whose schema becomes the required schema. All input columns become required. Pandas requires <code>convert_pandas=True</code>. Lazy input uses metadata without evaluating rows.</dd>
<dt>y : object, default=None</dt><dd>Ignored. Present for scikit-learn estimator compatibility.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>self : SchemaValidator</dt><dd>Fitted validator with a newly learned schema.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If the input type is unsupported or pandas conversion is disabled.</dd>
<dt>ValueError</dt><dd>If validation configuration is invalid.</dd>
</dl>

### transform {#transform}

<div class="api-signature"><code>transform(X)</code></div>

Validate the learned schema and normalise permitted differences.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input containing all fitted required columns. Additional columns must be explicitly permitted. Pandas requires conversion enabled.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>X_validated : polars.DataFrame or polars.LazyFrame</dt><dd>Validated input with any configured casts, ordering, and extra column policy applied. Default preserves input kind; set_output can force the return container. The original input is never mutated.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>sklearn.exceptions.NotFittedError</dt><dd>If <code>fit</code> has not completed successfully.</dd>
<dt>TypeError</dt><dd>If input or options are unsupported, or pandas conversion is disabled.</dd>
<dt>ValueError</dt><dd>If validation configuration is invalid.</dd>
<dt>SchemaValidationError</dt><dd>If required columns are missing, extras or order are disallowed, dtypes are incompatible, or a cast fails. Multiple mismatches are reported together, with eager cast exceptions preserved as the cause.</dd>
</dl>

#### Notes

Lazy schema and policy failures occur during this method. Permissive
cast failures occur at lazy execution and can be native Polars errors.
Lossless integer-to-float checks run at execution and raise
SchemaValidationError on failure. Explicit eager output wraps Polars
collection errors in SchemaValidationError with their causes.

### fit_transform {#fit-transform}

<div class="api-signature"><code>fit_transform(X, y=None, **fit_params)</code></div>

Fit to training data and return its transformed representation.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Training data. Pandas input requires <code>convert_pandas=True</code>. Lazy input is checked through metadata without evaluating rows.</dd>
<dt>y : object, default=None</dt><dd>Ignored. Present for scikit-learn estimator compatibility.</dd>
<dt>**fit_params : dict</dt><dd>Accepted for the scikit-learn method signature. Additional fit parameters are unsupported and raise <code>TypeError</code> when supplied.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>X_transformed : polars.DataFrame or polars.LazyFrame</dt><dd>Result of fitting and transforming the same input. Default preserves input kind; set_output can force an eager or lazy container.</dd>
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

Training data establishes the schema that new observations must satisfy. SchemaValidator learns names and dtypes during fit, then checks incoming structure and applies the selected dtype policy during transform.

## Why use it?

Use it to detect prediction-time schema drift, keep required feature order stable, and make conversions a deliberate policy. Strict rejects dtype changes; semi-strict permits defined widening; permissive attempts strict Polars casts.

## Examples

### Input

Training: `price: Float64`, `quantity: Int64`.

| quantity (String) | price (String) |
| --- | --- |
| "2" | "15.0" |
| "4" | "25.0" |

### Code

```python
import polars as pl

from polars_learn import SchemaValidator

train = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
incoming = pl.DataFrame({"quantity": ["2", "4"], "price": ["15.0", "25.0"]})
validator = SchemaValidator(mode="permissive", enforce_column_order=False).fit(train)
prepared = validator.transform(incoming)
assert prepared.to_dict(as_series=False) == {"price": [15.0, 25.0], "quantity": [2, 4]}
assert dict(prepared.schema) == validator.schema_
```

### Output

| price (Float64) | quantity (Int64) |
| --- | --- |
| 15.0 | 2 |
| 25.0 | 4 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

Permissive casts may lose information even when successful; choose the policy for your data contract. Missing required columns still fail. Lazy value checks execute at collection, and nullability is not learned. Consult the reference below for the exact widening and precision rules.

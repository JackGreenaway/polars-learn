# SchemaValidator

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L320">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">SchemaValidator</span>(
    mode=&#x27;strict&#x27;,
    *,
    allow_extra_columns=False,
    enforce_column_order=True,
    extra_columns=&#x27;preserve&#x27;,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)</code></pre></div>

<div class="api-class-body" markdown="1">

Learn a schema and validate subsequent Polars DataFrames or LazyFrames.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">mode</code> <span class="api-field-type">{&quot;strict&quot;, &quot;semi-strict&quot;, &quot;permissive&quot;}, default=&quot;strict&quot;</span></dt><dd><p>Validation policy. Strict mode requires identical dtypes. Semi-strict allows only the documented lossless widening policy. Permissive mode attempts Polars casts with strict=True. Successful casts can be lossy; failure occurs immediately for eager input or at lazy execution.</p></dd>
<dt><code class="api-field-name">allow_extra_columns</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Allow columns not present during fitting. Missing required columns always raise, regardless of the validation mode.</p></dd>
<dt><code class="api-field-name">enforce_column_order</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Require the relative order of required columns to match training. If false, output uses fitted column order followed by preserved extras.</p></dd>
<dt><code class="api-field-name">extra_columns</code> <span class="api-field-type">{&quot;preserve&quot;, &quot;drop&quot;}, default=&quot;preserve&quot;</span></dt><dd><p>Treatment of permitted extras. This option does not itself permit extra columns; <code>allow_extra_columns</code> must also be true.</p></dd>
<dt><code class="api-field-name">convert_pandas</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Allow pandas DataFrames and convert them to Polars. Requires pandas; some dtypes also require PyArrow.</p></dd>
<dt><code class="api-field-name">verbosity</code> <span class="api-field-type">{0, 1, 2}, default=0</span></dt><dd><p>Print progress at 1 and schema/operation details at 2; 0 is silent.</p></dd>
<dt><code class="api-field-name">warn_on_conversion</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Emit <code>PandasConversionWarning</code> when converting pandas input.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">schema_</code> <span class="api-field-type">dict of str to polars.DataType</span></dt><dd><p>Learned required columns and dtypes, in training order.</p></dd>
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Number of training columns.</p></dd>
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray of shape (n_features_in_,), dtype=object</span></dt><dd><p>Required input names in training order.</p></dd>
<dt><code class="api-field-name">output_features_</code> <span class="api-field-type">ndarray of shape (n_features_in_,), dtype=object</span></dt><dd><p>Fitted required names. Inference-only extras are not included.</p></dd>
</dl>

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If common options have invalid types.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If mode, extra_columns, or verbosity is invalid.</p></dd>
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
or "polars-lazy" output, or pandas output for compatibility.
With enforced order and preserved extras, output
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
| [set_output(*, transform=None)](#set-output) | Configure native Polars output or pandas compatibility output. |

### fit {#fit .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L437">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(X, y=None)</code></pre></div>

Learn column names, order, and dtypes without changing the input.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Training data whose schema becomes the required schema. All input columns become required. Pandas requires <code>convert_pandas=True</code>. Lazy input uses metadata without evaluating rows.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">object, default=None</span></dt><dd><p>Ignored. Present for scikit-learn estimator compatibility.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">SchemaValidator</span></dt><dd><p>Fitted validator with a newly learned schema.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If the input type is unsupported or pandas conversion is disabled.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If validation configuration is invalid.</p></dd>
</dl>

</div>

### transform {#transform .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L512">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">transform</span>(X)</code></pre></div>

Validate the learned schema and normalise permitted differences.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Input containing all fitted required columns. Additional columns must be explicitly permitted. Pandas requires conversion enabled.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">X_validated</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Validated input with any configured casts, ordering, and extra column policy applied. Default preserves input kind; set_output can force the return container. The original input is never mutated.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If <code>fit</code> has not completed successfully.</p></dd>
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If input or options are unsupported, or pandas conversion is disabled.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If validation configuration is invalid.</p></dd>
<dt><code class="api-field-name">SchemaValidationError</code></dt><dd><p>If required columns are missing, extras or order are disallowed, dtypes are incompatible, or a cast fails. Multiple mismatches are reported together, with eager cast exceptions preserved as the cause.</p></dd>
</dl>

#### Notes

Lazy schema and policy failures occur during this method. Permissive
cast failures occur at lazy execution and can be native Polars errors.
Lossless integer-to-float checks run at execution and raise
SchemaValidationError on failure. Explicit eager output wraps Polars
collection errors in SchemaValidationError with their causes.

</div>

### fit_transform {#fit-transform .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L471">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit_transform</span>(
    X,
    y=None,
    **fit_params,
)</code></pre></div>

Fit to training data and return its transformed representation.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Training data. Pandas input requires <code>convert_pandas=True</code>. Lazy input is checked through metadata without evaluating rows.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">object, default=None</span></dt><dd><p>Ignored. Present for scikit-learn estimator compatibility.</p></dd>
<dt><code class="api-field-name">**fit_params</code> <span class="api-field-type">dict</span></dt><dd><p>Accepted for the scikit-learn method signature. Additional fit parameters are unsupported and raise <code>TypeError</code> when supplied.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">X_transformed</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Result of fitting and transforming the same input. Default preserves input kind; set_output can force eager, lazy, or pandas output.</p></dd>
</dl>

</div>

### get_feature_names_out {#get-feature-names-out .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/_utils.py#L170">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_feature_names_out</span>(input_features=None)</code></pre></div>

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

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/_utils.py#L203">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_output</span>(*, transform=None)</code></pre></div>

Configure native Polars output or pandas compatibility output.

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

Permissive casts may lose information even when successful; choose the policy for your data contract. Missing required columns still fail. Lazy value checks execute at collection, and nullability is not learned. Consult the [class reference](#reference) and [casting policy details](../user-guide/schemas.md#learn-the-training-schema) for the exact widening and precision rules.

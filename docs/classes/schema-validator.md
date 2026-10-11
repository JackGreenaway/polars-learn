# SchemaValidator

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L371">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">SchemaValidator</span>(
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

Use this transformer when training data defines your model's input contract.
`fit` records every training column's name, order, and Polars dtype;
`transform` checks new data against that learned schema and handles
permitted casts, extra columns, and ordering according to your options.
Strict mode rejects dtype changes, semi-strict allows defined lossless
widening, and permissive attempts Polars casts that may lose information.
Missing required columns always fail. This validates structure and dtype
compatibility, not business meaning, nullability, or feature distributions.
Choose `SchemaSpec` when the required schema is already known.

Polars is the execution engine. Native DataFrames and LazyFrames are
accepted; pandas input requires `convert_pandas=True` and is converted
behind the scenes. Output is native Polars by default, including an eager
Polars frame for converted pandas input. Use
`set_output(transform="pandas")` for compatibility output; install
`polars-learn[pandas]` for pandas and PyArrow. Lazy output defers row
execution, while eager Polars or pandas output collects during transform.
Prefer native Polars input/output for speed and to avoid conversion overhead.

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

### See also

[FeatureEngineer](feature-engineer.md): Engineer features after validating raw inputs.

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

### Examples

```pycon
>>> import polars as pl
>>> from polars_learn import SchemaValidator
>>> train = pl.DataFrame({"x": pl.Series([1], dtype=pl.Int64)})
>>> test = pl.DataFrame({"x": pl.Series([2], dtype=pl.Int32)})
>>> validator = SchemaValidator(mode="semi-strict").fit(train)
>>> validator.transform(test).schema["x"]
Int64
```

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

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L507">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(X, y=None)</code></pre></div>

Learn column names, order, and dtypes without changing the input.

Call on training data to establish the schema used by later transforms.
Reads column metadata and records every input column as required, in its
training order. It does not cast values, learn nullability, or evaluate lazy
rows. A successful refit replaces the previous learned contract; `y` is
ignored because validation does not learn from the target.

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

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L601">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">transform</span>(X)</code></pre></div>

Validate the learned schema and normalise permitted differences.

Call after fitting to validate new observations before feature engineering
or prediction. Checks all required columns against `schema_`, applies the
chosen dtype policy, and handles permitted extras and ordering. The fitted
schema remains unchanged. Structural errors fail during this call; lazy
value-dependent cast checks run when the resulting query is collected.
The original input is never modified.

Output follows `set_output`: native Polars by default, eager Polars for
`"polars"`, a LazyFrame for `"polars-lazy"`, or a pandas DataFrame for
`"pandas"`. Eager Polars and pandas output collect lazy rows. Pandas output
uses Polars conversion defaults and may change dtype/null representation;
missing pandas or PyArrow raises `ImportError` at conversion. Pandas input
requires `convert_pandas=True` independently of the output choice.

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

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L547">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit_transform</span>(
    X,
    y=None,
    **fit_params,
)</code></pre></div>

Fit to training data and return its transformed representation.

Convenience method for preparing training data: calls `fit(X, y)` and
then `transform(X)` on the same input. This updates fitted state and
returns the prepared frame. The target is ignored and extra fit parameters
are unsupported. See `fit` for the class-specific learned contract and
`transform` for validation and output behavior.

Output follows `set_output`: native Polars by default, eager Polars for
`"polars"`, a LazyFrame for `"polars-lazy"`, or a pandas DataFrame for
`"pandas"`. Eager Polars and pandas output collect lazy rows. Pandas output
uses Polars conversion defaults and may change dtype/null representation;
missing pandas or PyArrow raises `ImportError` at conversion. Pandas input
requires `convert_pandas=True` independently of the output choice.

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

#### See also

`fit`: Learn the fitted schema and feature names.
`transform`: Apply the fitted transformer and its validation policies.

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

## Related guide

[Concepts and worked workflows](../user-guide/schema-validator-details.md).

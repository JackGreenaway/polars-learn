# SchemaSpec

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L16">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">SchemaSpec</span>(
    schema,
    *,
    allow_extra_columns=False,
    enforce_column_order=True,
    extra_columns=&#x27;preserve&#x27;,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)</code></pre></div>

<div class="api-class-body" markdown="1">

Describe a fixed contract for Polars DataFrames and LazyFrames.

Use this transformer when your application already defines the required
column names and dtypes. Both `fit` and `transform` check that explicit
contract; required dtypes must match exactly and are never cast.
Output places required columns in schema-key order and handles additional
columns according to the extra-column options. `fit` records the output
layout, including preserved extras, which later transformations must match.
Choose `SchemaValidator` instead to learn the contract from training data
or permit configured dtype conversions. Neither class checks business
rules such as allowed value ranges or required non-null values.

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
<dt><code class="api-field-name">schema</code> <span class="api-field-type">dict of str to polars.DataType</span></dt><dd><p>Required input column names and dtypes in order.</p></dd>
<dt><code class="api-field-name">allow_extra_columns</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Whether additional, non-required columns are accepted.</p></dd>
<dt><code class="api-field-name">enforce_column_order</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Require the relative order of required columns to match schema key order.</p></dd>
<dt><code class="api-field-name">extra_columns</code> <span class="api-field-type">{&quot;preserve&quot;, &quot;drop&quot;}, default=&quot;preserve&quot;</span></dt><dd><p>If extra columns are permitted, whether to keep or drop them in output.</p></dd>
<dt><code class="api-field-name">convert_pandas</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Allow pandas DataFrames and convert them to Polars. Requires pandas; some dtypes also require PyArrow.</p></dd>
<dt><code class="api-field-name">verbosity</code> <span class="api-field-type">{0, 1, 2}, default=0</span></dt><dd><p>Print progress at 1 and schema/operation details at 2; 0 is silent.</p></dd>
<dt><code class="api-field-name">warn_on_conversion</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Emit <code>PandasConversionWarning</code> when converting pandas input.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">schema_</code> <span class="api-field-type">dict of str to polars.DataType</span></dt><dd><p>Copy of the required explicit schema recorded by fit.</p></dd>
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray of str, dtype=object</span></dt><dd><p>All training input column names in input order, including extras.</p></dd>
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Number of all training input columns.</p></dd>
<dt><code class="api-field-name">output_features_</code> <span class="api-field-type">ndarray of str, dtype=object</span></dt><dd><p>Fitted output names, including preserved training extras.</p></dd>
</dl>

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If schema keys/values or common options have invalid types.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If schema is empty or not a dict, or an option has an invalid value.</p></dd>
</dl>

### See also

[SchemaValidator](schema-validator.md): Learn required dtypes and allow inference-only extra names.

### Notes

DataFrames return DataFrames and LazyFrames return unevaluated LazyFrames
by default. Use set_output to force eager, lazy, or pandas output.
Required dtypes must match exactly; there is no casting. Required output
columns follow schema key order, then preserved extras follow input order.
Transform requires fitting and exact fitted output names/order, including
preserved extras. Extra dtypes are not validated.

### Examples

```pycon
>>> import polars as pl
>>> from polars_learn import SchemaSpec
>>> X = pl.DataFrame({"extra": [9], "x": [1]})
>>> contract = SchemaSpec({"x": pl.Int64}, allow_extra_columns=True, extra_columns="drop")
>>> contract.fit_transform(X.lazy()).collect().columns
['x']
```

### Methods

| Method | Purpose |
| --- | --- |
| [fit(X, y=None)](#fit) | Validate the explicit contract and record fitted output names. |
| [transform(X)](#transform) | Check exact required dtypes and return the fitted output columns. |
| [fit_transform(X, y=None, **fit_params)](#fit-transform) | Fit and transform the same input using the configured return kind. |
| [get_feature_names_out(input_features=None)](#get-feature-names-out) | Return the fitted output column names. |
| [get_params(deep=True)](#get-params) | Get estimator constructor parameters. |
| [set_params(**params)](#set-params) | Set constructor parameters using scikit-learn conventions. |
| [set_output(*, transform=None)](#set-output) | Configure native Polars output or pandas compatibility output. |

### fit {#fit .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L148">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(X, y=None)</code></pre></div>

Validate the explicit contract and record fitted output names.

Call before transforming to verify the declared contract and record the
output layout. Checks required names, exact dtypes, ordering, and allowed
extras, then stores the explicit schema and output names. The schema comes
from the constructor, not from inferred training dtypes. Preserved training
extras become part of the output-name contract. No values are cast.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Input with all required columns and exact required dtypes. Extra columns and ordering follow the constructor&#x27;s policies. Pandas requires conversion enabled.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">object, default=None</span></dt><dd><p>Ignored; accepted for scikit-learn compatibility.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">SchemaSpec</span></dt><dd><p>Fitted contract. The provided schema is checked, not inferred anew.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If input, option types, or schema keys/values are unsupported.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If the schema or input contract is invalid.</p></dd>
</dl>

#### Notes

Lazy input is checked through schema metadata without collecting rows.
Fitted output names include preserved training extras. No casts occur.

</div>

### transform {#transform .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L271">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">transform</span>(X)</code></pre></div>

Check exact required dtypes and return the fitted output columns.

Call after fitting to enforce the application's fixed contract on new data.
Checks required names, exact dtypes, configured order, and extra-column
policies, then selects required columns in schema-key order. The output
names and order must match fitting, including preserved extras. No casting
occurs and the original input is never modified.

Output follows `set_output`: native Polars by default, eager Polars for
`"polars"`, a LazyFrame for `"polars-lazy"`, or a pandas DataFrame for
`"pandas"`. Eager Polars and pandas output collect lazy rows. Pandas output
uses Polars conversion defaults and may change dtype/null representation;
missing pandas or PyArrow raises `ImportError` at conversion. Pandas input
requires `convert_pandas=True` independently of the output choice.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Input satisfying the explicit required schema and configured extra-column/order policies. Pandas requires conversion enabled.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">X_validated</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Required columns in schema order, then preserved extras in input order. Default preserves input kind; set_output can force eager, lazy, or pandas output.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If fit has not completed successfully.</p></dd>
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If input or parameter types are unsupported.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If the contract fails, configuration is invalid, or resulting output names/order differ from fitting, including preserved extra names.</p></dd>
</dl>

#### Notes

No casting occurs; extra dtypes are not checked. Lazy validation uses
metadata. Explicit eager output collects the result; source/collection
errors are not wrapped.

</div>

### fit_transform {#fit-transform .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/schema_validation.py#L193">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit_transform</span>(
    X,
    y=None,
    **fit_params,
)</code></pre></div>

Fit and transform the same input using the configured return kind.

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
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Input accepted by fit and transform. Pandas requires conversion enabled.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">object, default=None</span></dt><dd><p>Ignored; accepted for scikit-learn compatibility.</p></dd>
<dt><code class="api-field-name">**fit_params</code> <span class="api-field-type">dict</span></dt><dd><p>Extra fit parameters are unsupported and raise TypeError.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">X_transformed</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Transformed input. Default preserves input kind; set_output can force an eager DataFrame or LazyFrame. Converted pandas is eager.</p></dd>
</dl>

#### See also

`fit`: Validate input and record fitted state.
`transform`: Apply the fitted transformation and output configuration.

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

[Concepts and worked workflows](../user-guide/schema-spec-details.md).

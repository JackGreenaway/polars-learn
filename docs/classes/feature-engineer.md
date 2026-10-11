# FeatureEngineer

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_engineering.py#L21">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">polars_learn.</span><span class="api-name">FeatureEngineer</span>(
    features,
    *,
    convert_pandas=False,
    warn_on_conversion=True,
    verbosity=0,
)</code></pre></div>

<div class="api-class-body" markdown="1">

Apply ordered feature specifications to Polars DataFrames or LazyFrames.

Use this transformer to keep Polars feature logic inside a model pipeline,
so training and prediction apply the same preparation to raw observations.
It applies each `FeatureSpec` in entry order using Polars `with_columns`,
then applies that specification's drops before starting the next. Later
specifications can therefore use retained outputs from earlier stages.
`fit` records input names/order and the resulting output schema;
`transform` replays the definitions and checks that schema. It does not
learn expression-specific training statistics: add a fitted statistical
transformer for quantities that must remain fixed after training.

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
<dt><code class="api-field-name">features</code> <span class="api-field-type">FeatureSpec, dict, or sequence of FeatureSpec/dict</span></dt><dd><p>A specification or a dict of FeatureSpec keyword arguments, optionally in an ordered sequence mixing both forms. Dicts are unpacked into FeatureSpec and validated before processing input. Single definitions are treated as one-element sequences. Specifications are applied in the supplied sequence order. Each specification adds its expressions, then drops its configured columns, before the next specification is applied. Later specifications may reference earlier outputs that remain present. An empty sequence produces an identity transformation; an empty dict is invalid. The original parameter is retained unchanged for estimator cloning.</p></dd>
<dt><code class="api-field-name">convert_pandas</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Allow pandas DataFrames and convert them to Polars. Requires pandas; some dtypes also require PyArrow. No conversion occurs for Polars input.</p></dd>
<dt><code class="api-field-name">verbosity</code> <span class="api-field-type">{0, 1, 2}, default=0</span></dt><dd><p>Print progress at 1 and schema/operation details at 2; 0 is silent.</p></dd>
<dt><code class="api-field-name">warn_on_conversion</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Emit <code>PandasConversionWarning</code> when converting pandas input.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Number of columns observed during fitting.</p></dd>
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray of shape (n_features_in_,), dtype=object</span></dt><dd><p>Input column names in training order.</p></dd>
<dt><code class="api-field-name">input_schema_</code> <span class="api-field-type">dict of str to polars.DataType</span></dt><dd><p>Input dtypes observed during fitting, independent of other estimators.</p></dd>
<dt><code class="api-field-name">output_schema_</code> <span class="api-field-type">dict of str to polars.DataType</span></dt><dd><p>Expected transformed names, order, and dtypes.</p></dd>
<dt><code class="api-field-name">output_features_</code> <span class="api-field-type">ndarray of shape (n_features_out_,), dtype=object</span></dt><dd><p>Output column names, including retained inputs and engineered features.</p></dd>
</dl>

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If feature definitions or common options have invalid types, a config dict has unknown keys, or a config has invalid FeatureSpec value types.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If verbosity is outside 0, 1, and 2 or a config dict fails FeatureSpec validation, such as an empty expression sequence or an ambiguous alias.</p></dd>
</dl>

### Notes

Specifications preserve the supplied sequence order in fit and transform.
Expressions within one specification are siblings: they read the columns
available before that specification, not new sibling outputs. Use later
specifications to reference newly created columns, or compose expressions
directly from existing input columns in one specification.
The public API accepts DataFrames and LazyFrames. Lazy inputs are checked
using schema metadata and return unevaluated plans by default. Eager input
executes the training transformation; lazy fit validates its schema only.
Value-dependent errors in lazy plans surface when collected.
Transform requires fitted input names and order and the fitted output schema.
Default output preserves input kind. set_output(transform="polars")
collects a DataFrame; "polars-lazy" returns a LazyFrame, which may wrap an
evaluated result for eager input. "pandas" converts an eager result for
compatibility. Prefer native Polars output to avoid conversion overhead.
Fit always returns self. fit_transform
calls fit then transform and evaluates eager feature expressions twice.
Pickling retains input definitions, fitted schemas, and output settings.
Expressions use Polars' native persistence, with cloudpickle for embedded
Python callbacks. Version mismatches warn before state restoration, then
compatible loading is attempted. Incompatible state raises SerializationError.
Load trusted artifacts only and validate features after upgrading.

### Methods

| Method | Purpose |
| --- | --- |
| [fit(X, y=None)](#fit) | Validate training expressions and learn input and output schemas. |
| [transform(X)](#transform) | Generate features and verify the fitted output schema. |
| [fit_transform(X, y=None, **fit_params)](#fit-transform) | Fit to training data and return its transformed representation. |
| [get_feature_names_out(input_features=None)](#get-feature-names-out) | Return the fitted output column names. |
| [get_params(deep=True)](#get-params) | Get estimator constructor parameters. |
| [set_params(**params)](#set-params) | Set constructor parameters using scikit-learn conventions. |
| [set_output(*, transform=None)](#set-output) | Configure native Polars output or pandas compatibility output. |

### fit {#fit .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_engineering.py#L226">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(X, y=None)</code></pre></div>

Validate training expressions and learn input and output schemas.

Call on raw training data to establish the expected input and feature layout.
Builds the ordered feature plan and records input names/order and resulting
output names/dtypes. Eager input executes the plan; lazy input checks schema
metadata without evaluating rows. Expressions are not converted into learned
statistics. Returns the fitted transformer, not the engineered frame.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Training data. Pandas input requires <code>convert_pandas=True</code>. Lazy training validates plan schemas without evaluating row values.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">object, default=None</span></dt><dd><p>Ignored. Present for scikit-learn estimator compatibility.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">FeatureEngineer</span></dt><dd><p>Fitted transformer.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If input, feature definitions, or options have invalid types, or a config dict contains unknown keys.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If verbosity or FeatureSpec config values are invalid.</p></dd>
<dt><code class="api-field-name">FeatureEngineeringError</code></dt><dd><p>If expressions or drop operations fail. The underlying exception is preserved as the cause.</p></dd>
</dl>

</div>

### transform {#transform .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_engineering.py#L318">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">transform</span>(X)</code></pre></div>

Generate features and verify the fitted output schema.

Call after fitting to prepare new raw observations with the same feature
logic. Requires the fitted input names in their original order, applies
specifications in entry order, and checks output names, order, and dtypes
against `output_schema_`. Each stage adds its expressions and then drops
its configured columns; subsequent stages can use retained earlier outputs.
The original input is never modified.

Output follows `set_output`: native Polars by default, eager Polars for
`"polars"`, a LazyFrame for `"polars-lazy"`, or a pandas DataFrame for
`"pandas"`. Eager Polars and pandas output collect lazy rows. Pandas output
uses Polars conversion defaults and may change dtype/null representation;
missing pandas or PyArrow raises `ImportError` at conversion. Pandas input
requires `convert_pandas=True` independently of the output choice.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Input with the fitted column names and order. Pandas input requires <code>convert_pandas=True</code>. Input dtypes may differ only if the final output schema still matches the fitted expectations.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">X_transformed</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Retained and engineered columns in native Polars order. Default preserves input kind; set_output can force eager, lazy, or pandas output.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If <code>fit</code> has not completed successfully.</p></dd>
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If input, features, or options are unsupported, config keys are unknown, or pandas conversion is disabled.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If verbosity or FeatureSpec config values are invalid.</p></dd>
<dt><code class="api-field-name">FeatureEngineeringError</code></dt><dd><p>If input columns differ, feature execution fails, or the generated output schema differs from the fitted schema.</p></dd>
</dl>

#### Notes

Schema/dependency errors fail before a lazy plan is returned. Value
errors in that plan occur when collected and can be native Polars
errors. Explicit eager output wraps execution errors in
FeatureEngineeringError, preserving their causes.

</div>

### fit_transform {#fit-transform .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/polars-learn/blob/main/src/polars_learn/feature_engineering.py#L270">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit_transform</span>(
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
Eager feature expressions run twice because fit and transform each execute
the feature plan. Lazy input remains a plan unless eager output is requested.

Output follows `set_output`: native Polars by default, eager Polars for
`"polars"`, a LazyFrame for `"polars-lazy"`, or a pandas DataFrame for
`"pandas"`. Eager Polars and pandas output collect lazy rows. Pandas output
uses Polars conversion defaults and may change dtype/null representation;
missing pandas or PyArrow raises `ImportError` at conversion. Pandas input
requires `convert_pandas=True` independently of the output choice.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">polars.DataFrame, polars.LazyFrame or pandas.DataFrame</span></dt><dd><p>Training data. Pandas input requires <code>convert_pandas=True</code>. Lazy training validates plan schemas without evaluating row values.</p></dd>
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

## About

Feature preparation becomes part of the model pipeline. Define derived columns once, fit the raw-input and output-schema contract, and let prediction replay the same definitions before the estimator receives its features.

## Why use it?

Use it when training and serving must share ordered, deterministic feature logic. It is especially useful for dependent features: calculate an intermediate column in one specification, consume it in the next, and drop it before the model boundary.

## Examples

### Input

| price | quantity |
| --- | --- |
| 10.0 | 2 |
| 20.0 | 3 |

### Code

```python
import polars as pl

from polars_learn import FeatureEngineer, FeatureSpec

raw = pl.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
engineer = FeatureEngineer(
    [
        FeatureSpec(pl.col("price") * pl.col("quantity"), alias="notional"),
        FeatureSpec(
            pl.col("notional").log1p(), alias="log_notional", drop=["quantity"]
        ),
    ]
)
prepared = engineer.fit_transform(raw)
assert prepared.columns == ["price", "notional", "log_notional"]
assert prepared["notional"].to_list() == [20.0, 60.0]
assert prepared["log_notional"].equals(pl.Series("log_notional", [20.0, 60.0]).log1p())
assert raw.columns == ["price", "quantity"]
```

### Output

| price | notional | log_notional |
| --- | --- | --- |
| 10.0 | 20.0 | 3.044522437723423 |
| 20.0 | 60.0 | 4.110873864173311 |

The assertions in the example verify the output against the implementation. Python examples are executed during documentation validation.

## Practical notes

Put the engineer before the predictive estimator and pass raw data to both pipeline.fit and pipeline.predict. It records schemas rather than learned feature statistics; use fitted statistical preprocessing for training-derived means, scales, or vocabularies. Eager fit_transform evaluates expressions twice. See the [complete prediction pipeline](../getting-started.md#your-first-raw-data-prediction-pipeline).

# FeatureEngineer

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><code>class polars_learn.FeatureEngineer(features, *, convert_pandas=False, warn_on_conversion=True, verbosity=0)</code></div>

Apply ordered feature specifications to Polars DataFrames or LazyFrames.

### Parameters

<dl class="api-fields">
<dt>features : FeatureSpec, dict, or sequence of FeatureSpec/dict</dt><dd>A specification or a dict of FeatureSpec keyword arguments, optionally in an ordered sequence mixing both forms. Dicts are unpacked into FeatureSpec and validated before processing input. Single definitions are treated as one-element sequences. Later specifications may reference earlier outputs. An empty sequence produces an identity transformation; an empty dict is invalid. The original parameter is retained unchanged for estimator cloning.</dd>
<dt>convert_pandas : bool, default=False</dt><dd>Allow pandas DataFrames and convert them to Polars. Requires pandas; some dtypes also require PyArrow. No conversion occurs for Polars input.</dd>
<dt>verbosity : {0, 1, 2}, default=0</dt><dd>Print progress at 1 and schema/operation details at 2; 0 is silent.</dd>
<dt>warn_on_conversion : bool, default=True</dt><dd>Emit <code>PandasConversionWarning</code> when converting pandas input.</dd>
</dl>

### Attributes

<dl class="api-fields">
<dt>n_features_in_ : int</dt><dd>Number of columns observed during fitting.</dd>
<dt>feature_names_in_ : ndarray of shape (n_features_in_,), dtype=object</dt><dd>Input column names in training order.</dd>
<dt>input_schema_ : dict of str to polars.DataType</dt><dd>Input dtypes observed during fitting, independent of other estimators.</dd>
<dt>output_schema_ : dict of str to polars.DataType</dt><dd>Expected transformed names, order, and dtypes.</dd>
<dt>output_features_ : ndarray of shape (n_features_out_,), dtype=object</dt><dd>Output column names, including retained inputs and engineered features.</dd>
</dl>

### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If feature definitions or common options have invalid types, a config dict has unknown keys, or a config has invalid FeatureSpec value types.</dd>
<dt>ValueError</dt><dd>If verbosity is outside 0, 1, and 2 or a config dict fails FeatureSpec validation, such as an empty expression sequence or an ambiguous alias.</dd>
</dl>

### Notes

The public API accepts DataFrames and LazyFrames. Lazy inputs are checked
using schema metadata and return unevaluated plans by default. Eager input
executes the training transformation; lazy fit validates its schema only.
Value-dependent errors in lazy plans surface when collected.
Transform requires fitted input names and order and the fitted output schema.
Default output preserves input kind. set_output(transform="polars")
collects a DataFrame; "polars-lazy" returns a LazyFrame, which may wrap an
evaluated result for eager input. Fit always returns self. fit_transform
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
| [set_output(*, transform=None)](#set-output) | Configure native output without automatic container conversion. |

### fit {#fit}

<div class="api-signature"><code>fit(X, y=None)</code></div>

Validate training expressions and learn input and output schemas.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Training data. Pandas input requires <code>convert_pandas=True</code>. Lazy training validates plan schemas without evaluating row values.</dd>
<dt>y : object, default=None</dt><dd>Ignored. Present for scikit-learn estimator compatibility.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>self : FeatureEngineer</dt><dd>Fitted transformer.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>TypeError</dt><dd>If input, feature definitions, or options have invalid types, or a config dict contains unknown keys.</dd>
<dt>ValueError</dt><dd>If verbosity or FeatureSpec config values are invalid.</dd>
<dt>FeatureEngineeringError</dt><dd>If expressions or drop operations fail. The underlying exception is preserved as the cause.</dd>
</dl>

### transform {#transform}

<div class="api-signature"><code>transform(X)</code></div>

Generate features and verify the fitted output schema.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Input with the fitted column names and order. Pandas input requires <code>convert_pandas=True</code>. Input dtypes may differ only if the final output schema still matches the fitted expectations.</dd>
</dl>

#### Returns

<dl class="api-fields">
<dt>X_transformed : polars.DataFrame or polars.LazyFrame</dt><dd>Retained and engineered columns in native Polars order. Default preserves input kind; set_output can force an eager or lazy container.</dd>
</dl>

#### Raises

<dl class="api-fields">
<dt>sklearn.exceptions.NotFittedError</dt><dd>If <code>fit</code> has not completed successfully.</dd>
<dt>TypeError</dt><dd>If input, features, or options are unsupported, config keys are unknown, or pandas conversion is disabled.</dd>
<dt>ValueError</dt><dd>If verbosity or FeatureSpec config values are invalid.</dd>
<dt>FeatureEngineeringError</dt><dd>If input columns differ, feature execution fails, or the generated output schema differs from the fitted schema.</dd>
</dl>

#### Notes

Schema/dependency errors fail before a lazy plan is returned. Value
errors in that plan occur when collected and can be native Polars
errors. Explicit eager output wraps execution errors in
FeatureEngineeringError, preserving their causes.

### fit_transform {#fit-transform}

<div class="api-signature"><code>fit_transform(X, y=None, **fit_params)</code></div>

Fit to training data and return its transformed representation.

#### Parameters

<dl class="api-fields">
<dt>X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame</dt><dd>Training data. Pandas input requires <code>convert_pandas=True</code>. Lazy training validates plan schemas without evaluating row values.</dd>
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

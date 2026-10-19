# Diagnostics and errors

All seven public classes accept an integer `verbosity`:

| Level         | Output to standard output                                                          |
| ------------- | ---------------------------------------------------------------------------------- |
| `0` (default) | No progress diagnostics.                                                           |
| `1`           | Input dimensions and fit/transform progress; specification construction summaries. |
| `2`           | Level 1 plus schemas and feature operation details.                                |

```python
import polars as pl

from polars_learn import SchemaValidator

X = pl.DataFrame({"x": [1, 2]})
validator = SchemaValidator(verbosity=1)
validated = validator.fit_transform(X)
# [SchemaValidator] Validating input
# [SchemaValidator] Input: 2 rows, 1 columns
# [SchemaValidator] Fit complete; output columns: ['x']
# [SchemaValidator] Validating input
# [SchemaValidator] Input: 2 rows, 1 columns
# [SchemaValidator] Transform complete: 2 rows, 1 columns
```

`FeatureSpec` reports its construction separately. An engineer's verbosity does not change its specifications' verbosity. A dict config's `verbosity` applies when that config is converted into a specification, during construction and each fit/transform call. Frame summaries show metadata rather than printing data rows; feature details can include expression representations. Levels must be `0`, `1`, or `2`; booleans and strings are rejected. Verbosity does not suppress exceptions or pandas conversion warnings.

## Failure timing

| Situation                                                                               | Exception or warning                | Timing                                      |
| --------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------- |
| Invalid argument type or unsupported input                                              | `TypeError`                         | Construction or the next relevant operation |
| Invalid settings, conflicting policies, or duplicate configured names                   | `ValueError`                        | Construction or revalidation                |
| Transform or feature-name access before fitting                                         | `sklearn.exceptions.NotFittedError` | Method call                                 |
| `SchemaSpec` contract or fitted output-name mismatch                                    | `ValueError`                        | Fit/transform schema check                  |
| `ColumnSelector`/`FeaturePassthrough` missing columns or fitted output-name mismatch    | `ValueError`                        | Fit/transform schema check                  |
| `SchemaValidator` missing columns, disallowed extras/order/types, or eager cast failure | `SchemaValidationError`             | Transform                                   |
| `ValueValidator` check false/null or check evaluation failure                       | `ValueValidationError`               | Transform/fit_transform, including lazy input |
| Feature dependency, drop, or schema failure                                         | `FeatureEngineeringError`           | Fit/transform schema check                   |
| Eager feature execution failure                                                      | `FeatureEngineeringError`           | Transform/fit_transform                      |
| Value-dependent feature failure or permissive cast failure in a returned lazy plan      | Native Polars exception             | `collect()` or another execution operation  |
| Lossy semi-strict integer to Float64 cast in a lazy plan                                | `SchemaValidationError` (may be wrapped in Polars `ComputeError` on older Polars versions) | Lazy execution |
| Enabled pandas conversion with warnings enabled                                         | `PandasConversionWarning`           | Conversion, even at verbosity 0             |

`PolarsLearnError` is the base of `SchemaValidationError`, `ValueValidationError`, and `FeatureEngineeringError`. Plain `TypeError`/`ValueError`, `NotFittedError`, and native Polars errors do not inherit from it. Input query and source failures can also propagate as native Polars or source exceptions during schema resolution or collection. Wrapped errors preserve their original exception as `__cause__`.

For structural/feature transformers, request `"polars"` output to force value evaluation within `transform`. `ValueValidator.fit` checks expression schemas only; its row checks run in `transform`, including for lazy input. `FeatureEngineer.fit` resolves the feature schema only; feature expressions run in `transform`. `FeatureEngineer` wraps resulting execution errors in `FeatureEngineeringError`; `SchemaValidator` wraps Polars collection errors in `SchemaValidationError`. `SchemaSpec`, `ColumnSelector`, and `FeaturePassthrough` do not wrap source/collection errors.

## Warning controls

Verbosity controls progress output and does not control warnings. Version warnings use standard Python filters; see [persistence](persistence.md). Pandas conversion warnings can be disabled with `warn_on_conversion=False`. These controls do not suppress exceptions. Feature details at verbosity 2 can include expression representations, so choose a diagnostic level appropriate for your application's logs.

For problem-specific remedies, see [troubleshooting](../troubleshooting.md). All public exceptions are listed in the [API reference](../api.md#exceptions-and-warnings).

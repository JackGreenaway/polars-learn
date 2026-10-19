# Validate row values

Use `ValueValidator` for row-level rules that a schema cannot express: required
values, numeric limits, and allowed categories. Each named check is a Polars
expression that must resolve to Boolean values. `False` and null results both
fail validation. The transformer returns the original columns and rows unchanged
when every check passes.

```python
import polars as pl

from polars_learn import SchemaValidator, ValueValidator

training = pl.DataFrame(
    {
        "price": [10.0, 20.0],
        "quantity": [2, 3],
        "service": ["standard", "priority"],
    }
)
pipeline = [
    SchemaValidator(),
    ValueValidator(
        {
            "positive_price": pl.col("price") > 0,
            "quantity_present": pl.col("quantity").is_not_null(),
            "known_service": pl.col("service").is_in(["standard", "priority"]),
        }
    ),
]
checked = training
for step in pipeline:
    checked = step.fit_transform(checked)
assert checked.equals(training)
```

For a service boundary, put these checks before feature engineering and the
estimator. Use explicit names because they appear in `ValueValidationError` and
can help identify which condition failed. Keep rule expressions deterministic
and side-effect free.

## Lazy execution

`ValueValidator.fit` resolves each check against schema metadata only. It does
not inspect row values or mutate the input. `transform` evaluates the checks
once. For a `LazyFrame`, checking values requires executing the query, so the
validator materializes the source once during transform. Default output preserves a lazy container as
an in-memory LazyFrame; collecting it later does not reread the source. Use
`set_output(transform="polars")` to return the already checked eager frame.
Validation therefore uses memory proportional to the materialized batch.

Value rules are different from fitted statistics. A range check verifies a
condition on each incoming row; it does not learn a mean, impute values, or
replace training-learned preprocessing. Put learned operations inside the
scikit-learn pipeline so cross-validation fits them separately in each fold.

See [schema contracts](schemas.md) for structural validation,
[diagnostics](diagnostics.md) for exception types, and the
[ValueValidator API](../classes/value-validator.md) for the full contract.

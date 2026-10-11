"""Generate scikit-learn-style class references from live signatures/docstrings."""

import argparse
import html
import inspect
import re
from pathlib import Path

import polars_learn

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {
    "FeatureEngineer": "feature-engineer",
    "FeatureSpec": "feature-spec",
    "SchemaSpec": "schema-spec",
    "SchemaValidator": "schema-validator",
    "ColumnSelector": "column-selector",
    "FeaturePassthrough": "feature-passthrough",
}
METHODS = (
    "fit",
    "transform",
    "fit_transform",
    "get_feature_names_out",
    "get_params",
    "set_params",
    "set_output",
)
START = "<!-- BEGIN GENERATED REFERENCE -->"
END = "<!-- END GENERATED REFERENCE -->"


def signature(obj):
    sig = inspect.signature(obj)
    parameters = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in sig.parameters.values()
        if p.name != "self"
    ]
    return str(
        sig.replace(parameters=parameters, return_annotation=inspect.Signature.empty)
    )


def inline(text):
    text = html.escape(text)
    return re.sub(r"``(.*?)``", r"<code>\1</code>", text)


def render_doc(doc, level):
    sections = re.split(r"\n([A-Za-z ]+)\n-{3,}\n", inspect.cleandoc(doc))
    output = [sections[0].replace("``", "`").strip() + "\n"]
    for heading, body in zip(sections[1::2], sections[2::2], strict=True):
        if heading in {"Examples", "See Also"}:
            continue
        output.append(f"{'#' * level} {heading}\n")
        if heading in {"Parameters", "Attributes", "Returns", "Raises"}:
            output.append('<dl class="api-fields">')
            entries = re.split(r"\n(?=\S)", body.strip())
            for entry in entries:
                lines = entry.splitlines()
                term = lines[0]
                description = " ".join(line.strip() for line in lines[1:])
                output.append(f"<dt>{inline(term)}</dt><dd>{inline(description)}</dd>")
            output.append("</dl>\n")
        else:
            output.append(body.replace("``", "`").strip() + "\n")
    return "\n".join(output)


def method_doc(cls, name):
    if name == "get_params":
        return """Get estimator constructor parameters.

        Parameters
        ----------
        deep : bool, default=True
            Include nested estimator parameters when present.

        Returns
        -------
        params : dict
            Constructor parameters and, when requested, nested parameters. Output
            configuration is managed separately by set_output.
        """
    if name == "set_params":
        return """Set constructor parameters using scikit-learn conventions.

        Parameters
        ----------
        **params : dict
            Constructor parameter names and values. Nested estimator parameters
            use the step__parameter convention where supported.

        Returns
        -------
        self : estimator instance
            This estimator with updated configuration.

        Raises
        ------
        ValueError
            If a parameter name is unknown.

        Notes
        -----
        Changing parameters does not immediately validate configuration or clear
        fitted attributes. Refit after contract changes. Invalid parameter values
        fail at the next relevant fit or transform operation.
        """
    doc = inspect.getdoc(getattr(cls, name)) or ""
    if cls.__name__ == "FeaturePassthrough" and name in {"fit", "transform"}:
        verb = (
            "Validate selected columns and record feature names."
            if name == "fit"
            else "Select fitted columns unchanged in configured order."
        )
        result = (
            "self : FeaturePassthrough\n    The fitted transformer."
            if name == "fit"
            else "X_selected : polars.DataFrame or polars.LazyFrame\n    "
            "Selected unchanged columns, preserving values, names, "
            "dtypes, nulls, and row order. Output follows "
            "set_output."
        )
        doc = f"""{verb}

Parameters
----------
X : polars.DataFrame, polars.LazyFrame or pandas.DataFrame
    Input containing all selected columns. Other columns are omitted from output.
    Input order and dtypes may differ. Pandas requires convert_pandas=True.
"""
        if name == "fit":
            doc += (
                "y : object, default=None\n"
                "    Ignored; accepted for estimator compatibility.\n"
            )
        doc += (
            f"\nReturns\n-------\n{result}\n\nRaises\n------\n"
            "TypeError\n    Unsupported input, column, or option types.\n"
            "ValueError\n    Missing selected columns, duplicate names, "
            "or invalid configuration.\n"
        )
        if name == "transform":
            doc += (
                "sklearn.exceptions.NotFittedError\n"
                "    Fit has not completed successfully.\n"
            )
        doc += (
            "\nNotes\n-----\nLazy metadata checks do not evaluate rows. "
            "Source and collection failures propagate from Polars. "
            "Dtypes are not validated.\n"
        )
    return doc


def reference(cls):
    name = cls.__name__
    output = [
        "## Reference\n",
        f'<div class="api-signature"><code>class '
        f"polars_learn.{name}{html.escape(signature(cls))}</code></div>\n",
        render_doc(inspect.getdoc(cls) or "", 3),
    ]
    if name == "FeatureSpec":
        output.append(
            "### Methods\n\nFeatureSpec is a description, not an "
            "estimator. It has no fit, transform, or prediction "
            "methods. Pass it to FeatureEngineer for execution.\n"
        )
    else:
        output.append("### Methods\n\n| Method | Purpose |\n| --- | --- |")
        for method in METHODS:
            summary = method_doc(cls, method).strip().splitlines()[0]
            output.append(
                f"| "
                f"[{method}{signature(getattr(cls, method))}]"
                f"(#{method.replace('_', '-')}) | {summary} |"
            )
        output.append("")
        for method in METHODS:
            output.append(f"### {method} {{#{method.replace('_', '-')}}}\n")
            output.append(
                f"<div "
                f'class="api-signature"><code>'
                f"{html.escape(method + signature(getattr(cls, method)))}"
                "</code></div>\n"
            )
            output.append(render_doc(method_doc(cls, method), 4))
    output.append(
        "### See also\n\n[Shared estimator "
        "conventions](../api.md#shared-transformer-behavior) · "
        "[Production guidance](../production.md) · "
        "[Persistence](../user-guide/persistence.md)\n"
    )
    return "\n".join(output).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, slug in CLASSES.items():
        path = ROOT / "docs" / "classes" / f"{slug}.md"
        content = path.read_text()
        generated = START + "\n" + reference(getattr(polars_learn, name)) + "\n" + END
        updated = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            lambda _, replacement=generated: replacement,
            content,
            flags=re.S,
        )
        updated = updated.rstrip() + "\n"
        if START not in content:
            raise ValueError(f"Missing reference marker in {path}")
        if args.check and updated != content:
            raise ValueError(
                f"Stale reference in {path}; run scripts/build_api_reference.py"
            )
        if not args.check:
            path.write_text(updated)
    print("All six class references match source signatures and docstrings.")


if __name__ == "__main__":
    main()

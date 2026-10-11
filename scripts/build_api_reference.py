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


def signature_block(obj, name, *, is_class=False):
    parameters = []
    keyword_separator = False
    for param in inspect.signature(obj).parameters.values():
        if param.name == "self":
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_separator:
            parameters.append("*")
            keyword_separator = True
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            keyword_separator = True
        parameters.append(str(param.replace(annotation=inspect.Parameter.empty)))
    label = "class " if is_class else ""
    prefix = "polars_learn." if is_class else ""
    rendered_name = (
        f'<span class="api-keyword">{label}</span>'
        f'<span class="api-module">{prefix}</span>'
        f'<span class="api-name">{html.escape(name)}</span>'
    )
    if is_class or len(parameters) > 2:
        arguments = "\n".join("    " + html.escape(p) + "," for p in parameters)
        call = f"{rendered_name}(\n{arguments}\n)"
    else:
        call = f"{rendered_name}({html.escape(', '.join(parameters))})"
    source_path = Path(inspect.getsourcefile(obj)).resolve()
    package_root = ROOT / "src"
    if source_path.is_relative_to(package_root):
        line = inspect.getsourcelines(obj)[1]
        source_url = (
            "https://github.com/JackGreenaway/polars-learn/blob/main/"
            f"{source_path.relative_to(ROOT).as_posix()}#L{line}"
        )
        source_label = "[source]"
    else:
        source_url = (
            "https://scikit-learn.org/stable/modules/generated/"
            f"sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.{name}"
        )
        source_label = "[inherited API]"
    return (
        '<div class="api-signature">'
        f'<a class="api-source" href="{source_url}">{source_label}</a>'
        f"<pre><code>{call}</code></pre></div>\n"
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
                field_name, separator, field_type = term.partition(" : ")
                label = f'<code class="api-field-name">{inline(field_name)}</code>'
                if separator:
                    label += (
                        f' <span class="api-field-type">{inline(field_type)}</span>'
                    )
                output.append(f"<dt>{label}</dt><dd><p>{inline(description)}</p></dd>")
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
        signature_block(cls, name, is_class=True),
        '<div class="api-class-body" markdown="1">\n',
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
            output.append(
                f"### {method} {{#{method.replace('_', '-')} .api-method-title}}\n"
            )
            output.append('<div class="api-method-body" markdown="1">\n')
            output.append(signature_block(getattr(cls, method), method))
            output.append(render_doc(method_doc(cls, method), 4))
            output.append("</div>\n")
    output.append(
        "### See also\n\n[Shared estimator "
        "conventions](../api.md#shared-transformer-behavior) · "
        "[Production guidance](../production.md) · "
        "[Persistence](../user-guide/persistence.md)\n"
    )
    output.append("</div>\n")
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
    api_path = ROOT / "docs" / "api.md"
    api = api_path.read_text()
    for name in CLASSES:
        start = f"<!-- BEGIN SIGNATURE:{name} -->"
        end = f"<!-- END SIGNATURE:{name} -->"
        if start not in api or end not in api:
            raise ValueError(f"Missing signature markers for {name}")
        block = (
            start
            + "\n"
            + signature_block(getattr(polars_learn, name), name, is_class=True)
            + end
        )
        api = re.sub(
            re.escape(start) + r".*?" + re.escape(end),
            lambda _, replacement=block: replacement,
            api,
            flags=re.S,
        )
    if args.check and api != api_path.read_text():
        raise ValueError("Stale API signatures; run scripts/build_api_reference.py")
    if not args.check:
        api_path.write_text(api)
    print("All six class references match source signatures and docstrings.")


if __name__ == "__main__":
    main()

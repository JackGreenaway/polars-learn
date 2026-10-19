# Development and documentation publishing

## Local development

```bash
uv sync --locked --group docs
uv run --locked pytest --doctest-modules src tests
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
uv build --out-dir /tmp/polars-learn-dist
```

Pandas and PyArrow are development dependencies, so interoperability tests run rather than skip. Runtime pandas support remains optional. The docs dependency group is separate from runtime dependencies and includes MkDocs; it is locked in `uv.lock`.

## Preview and build the documentation

All documentation sources live in `docs/`; `mkdocs.yml` defines the navigation, search, appearance, and strict link validation. The generated `site/` directory is ignored by Git.

```bash
uv sync --locked --group docs
uv run --locked --group docs mkdocs serve
```

Open the local URL printed by MkDocs. To build a deployable static site and validate every standalone Python example:

```bash
uv run --locked --group docs python scripts/build_api_reference.py --check
uv run --locked --group docs python scripts/check_docs.py
uv run --locked --group docs mkdocs build --strict
uv run --locked --group docs python scripts/check_site.py
```

Ruff is the formatter for Python source and documentation snippets. After changing a snippet, run `uv run --locked python scripts/check_docs.py --format`, then review the changes. The checker executes each Python block independently, so examples must include their own imports and sample data. Shell commands are instructions rather than executed tests.

## Writing documentation

Start with the reader's question and the behavior verified by the implementation.
Choose the page type before writing:

| Content | Where it belongs | What to include |
| --- | --- | --- |
| Package overview and installation | `docs/index.md`, `docs/getting-started.md` | Audience, requirements, capabilities, limitations, and a first runnable workflow. Keep the README brief. |
| Concepts and decisions | `docs/user-guide/` | Why and when to use a component, configuration choices, assumptions, tradeoffs, theory, and links to its reference. |
| Complete worked example | `docs/examples/` | A relatable question, local data, configuration, executable code, results, interpretation, and next steps. Add a gallery entry. |
| API contract | NumPy-style docstrings in `src/polars_learn/` | Concise summary, parameters and defaults, attributes, returns, errors, cross-references, and focused doctests. Generate the reference from these docstrings. |
| Contributor workflow | This development guide | Setup, authoring, validation, builds, and hosting requirements. |

### Style and page structure

Use one descriptive level-one heading per page, then level-two headings for
major sections. Write short paragraphs in plain language, use active verbs,
and explain a term before relying on it. Use backticks for identifiers and
parameter values. Tables work well for configuration comparisons; use lists
for steps or genuinely parallel items. Separate broader rationale from API
contracts, and preserve useful existing explanations when moving content.

State limitations alongside the behavior they qualify. Distinguish structural
schema validation from business rules, and deterministic expressions from
preprocessing that learns training statistics. Define equation symbols and
units in surrounding prose; use readable Unicode notation or native MathML
with an accessible label, following the [feature guide](user-guide/features.md#mathematical-model-and-leakage).

### Runnable examples and expected output

Every `python` fence runs independently. Include all imports, sample data, and
configuration; do not rely on variables from an earlier block. Prefer small,
local fixtures and fixed random seeds. Explain row selection and keep targets
outside feature frames. In predictive examples, split raw observations before
fitting and put learned preprocessing inside the pipeline or validation fold.
Use assertions to verify meaningful results, rather than only checking that
execution succeeds.

To verify displayed stdout, put the literal marker `<!-- verified-output -->`
immediately after the closing Python fence, followed by a `text` fence containing
the exact printed output, including its final newline. See the
[held-out order example](examples/held-out-orders.md). Round numerical output
explicitly when full precision would obscure the result. Explain what the result
means and what the example's data can support.

For source docstrings, use NumPy-style section headings with dashed underlines.
Include a short `Examples` section using `>>>` prompts and expected outputs;
these appear as `pycon` fences in the generated reference. Each example should
demonstrate one operation with its own imports and data. Follow the existing
public class docstrings, documenting methods' returns and errors where relevant.
Do not edit text inside generated reference markers or hand-maintain signatures.
Update the source docstring and run the generator instead.

### Links, assets, and navigation

Use relative Markdown links to source pages, including `.md`, and descriptive
link labels. Use heading fragments for cross-references and update incoming
links if a heading changes. Store assets under `docs/`, reference them relatively,
and give images meaningful alternative text. Avoid root-relative asset URLs:
the site is hosted below `/polars-learn/`. Use the existing theme and code-block
styles, checking readability in both light and dark modes and on narrow screens.

Add new pages to `nav` in `mkdocs.yml` and link them from the relevant guide or
gallery index. Keep both navigation order and page titles meaningful to readers.

### Authoring and review workflow

After editing source docstrings or Markdown, run from the repository root:

```bash
uv run --locked ruff format src
uv run --locked python scripts/build_api_reference.py
uv run --locked python scripts/check_docs.py --format
uv run --locked python scripts/build_api_reference.py --check
uv run --locked pytest --doctest-modules src -q
uv run --locked python scripts/check_docs.py
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked --group docs mkdocs build --strict
uv run --locked --group docs python scripts/check_site.py
```

Review generated and formatted changes before committing. Preview with
`uv run --locked --group docs mkdocs serve`, follow the changed page's links,
and inspect code, equations, navigation, and assets at desktop and mobile widths.
The strict build and rendered-site checker validate local references, but visual
review is still needed for layout and browser behavior. If behavior or dependency
requirements change, also run the repository checks in [Local development](#local-development).

## Publish with GitHub Pages

1. In the repository, open **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**.
2. Commit the documentation sources, `mkdocs.yml`, lockfile, and `.github/workflows/docs.yml`, then push to `main`.
3. The **Documentation** workflow checks examples, builds with strict validation, uploads the generated site, and deploys through the `github-pages` environment. Review any environment protection rules that apply to your repository.
4. Open the deployment URL shown by the workflow. For this repository the configured URL is `https://jackgreenaway.github.io/polars-learn/`.

Pull requests build and validate the docs without deploying. Manual runs deploy only when run from `main`. If your default branch, repository owner, or repository name changes, update the branch checks in the workflow and `site_url`, `repo_url`, and `edit_uri` in `mkdocs.yml`. Pages must be available for the repository's account and visibility settings. This setup follows [GitHub's custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

The Markdown sources are not a branch-published static site: choose **GitHub Actions**, which publishes the built HTML artifact. You can also host the generated `site/` directory on another static host.

## Package releases

The package release workflows are separate from documentation deployment. The existing **Create GitHub Release** workflow builds validated distributions, and **Upload Python Package** publishes them to PyPI. Keep `pyproject.toml` and `polars_learn.__version__` aligned and update the lockfile when dependencies change. See the [repository release instructions](legacy-workflows.md#releases).

## Keeping docs accurate

Update the user guide, API contract, and relevant examples together with interface changes. Keep public signatures, defaults, fitted attributes, output policies, and error timing synchronized with the implementation. Production claims should describe behavior that is verified, rather than promise universal estimator compatibility, portable arbitrary callbacks, or identical results across dependency upgrades.

The documentation describes the code deployed from the main branch. It is a single current site, not a separate hosted manual for each package release; check the package version and API when using an older installed distribution.

Class references are generated from the live public signatures and NumPy-style docstrings by `scripts/build_api_reference.py`. After changing source documentation or signatures, run `uv run --locked python scripts/build_api_reference.py` and commit the updated Markdown. CI checks that generated references match the implementation. Concepts and worked workflows live in the user guide. Reference examples are rendered from source doctests; the checker verifies them again in their displayed form. Every exported object must be covered by the generator. Python blocks followed by `<!-- verified-output -->` and a text fence have their printed output compared exactly.

## Documentation design

This manual follows the separation of explanation, runnable gallery examples,
and source-generated contracts used by the current scikit-learn
[user guide](https://scikit-learn.org/stable/user_guide.html),
[examples gallery](https://scikit-learn.org/stable/auto_examples/index.html),
[StandardScaler class reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
and [train_test_split function reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
There are no public standalone functions in this package; the reference covers
its seven components and seven exported exceptions/warnings. Private helpers are excluded.

The existing MkDocs theme supplies search, syntax highlighting, responsive
navigation, a keyboard skip link, and a light/dark toggle. Equations use native
MathML and readable Unicode notation without a network-dependent math renderer.
The rendered-site checker resolves relative resources and repository-prefixed
URLs, and verifies HTML fragments and asset existence. It does not test remote
websites or execute browser JavaScript. No hosting credentials are required to
preview or build locally. Publishing still requires the Pages setup above.

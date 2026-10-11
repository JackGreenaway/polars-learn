# Development and documentation publishing

## Local development

```bash
uv sync --locked --group docs
uv run --locked pytest --doctest-modules src tests
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
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
```

Ruff is the formatter for Python source and documentation snippets. After changing a snippet, run `uv run --locked python scripts/check_docs.py --format`, then review the changes. The checker executes each Python block independently, so examples must include their own imports and sample data. Shell commands are instructions rather than executed tests.

## Publish with GitHub Pages

1. In the repository, open **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**.
2. Commit the documentation sources, `mkdocs.yml`, lockfile, and `.github/workflows/docs.yml`, then push to `main`.
3. The **Documentation** workflow checks examples, builds with strict validation, uploads the generated site, and deploys through the `github-pages` environment. Review any environment protection rules that apply to your repository.
4. Open the deployment URL shown by the workflow. For this repository the configured URL is `https://jackgreenaway.github.io/polars-learn/`.

Pull requests build and validate the docs without deploying. Manual runs deploy only when run from `main`. If your default branch, repository owner, or repository name changes, update the branch checks in the workflow and `site_url`, `repo_url`, and `edit_uri` in `mkdocs.yml`. Pages must be available for the repository's account and visibility settings. This setup follows [GitHub's custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

The Markdown sources are not a branch-published static site: choose **GitHub Actions**, which publishes the built HTML artifact. You can also host the generated `site/` directory on another static host.

## Package releases

The package release workflows are separate from documentation deployment. The existing **Create GitHub Release** workflow builds validated distributions, and **Upload Python Package** publishes them to PyPI. Keep `pyproject.toml` and `polars_learn.__version__` aligned and update the lockfile when dependencies change. See the [repository release instructions](https://github.com/JackGreenaway/polars-learn#releases).

## Keeping docs accurate

Update the user guide, API contract, and relevant examples together with interface changes. Keep public signatures, defaults, fitted attributes, output policies, and error timing synchronized with the implementation. Production claims should describe behavior that is verified, rather than promise universal estimator compatibility, portable arbitrary callbacks, or identical results across dependency upgrades.

The documentation describes the code deployed from the main branch. It is a single current site, not a separate hosted manual for each package release; check the package version and API when using an older installed distribution.

Class references are generated from the live public signatures and NumPy-style docstrings by `scripts/build_api_reference.py`. After changing source documentation or signatures, run `uv run --locked python scripts/build_api_reference.py` and commit the updated Markdown. CI checks that generated references match the implementation. Class introductions and worked examples remain manually authored outside the generated markers.

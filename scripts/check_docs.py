"""Lint, format-check, and execute standalone Python documentation examples."""

import argparse
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format", action="store_true", help="Format snippets with Ruff"
    )
    args = parser.parse_args()
    count = 0
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        contents = path.read_text()

        def check(match: re.Match[str], example_path: Path = path) -> str:
            nonlocal count
            code = match[1]
            formatted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "format",
                    "--stdin-filename",
                    "example.py",
                    "-",
                ],
                input=code,
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
            ).stdout
            if not args.format and code != formatted:
                raise ValueError(
                    f"Unformatted example in {example_path}; run with --format"
                )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--stdin-filename",
                    "example.py",
                    "-",
                ],
                input=formatted,
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exec(
                    compile(formatted, str(example_path), "exec"),
                    {"__name__": "__main__"},
                )
            count += 1
            return f"```python\n{formatted}```"

        contents = re.sub(r"```python\n(.*?)```", check, contents, flags=re.S)
        if args.format:
            path.write_text(contents)
    print(f"Verified {count} standalone Python examples with Ruff and execution.")


if __name__ == "__main__":
    main()

"""Lint, format-check, and execute standalone Python documentation examples."""

import argparse
import contextlib
import doctest
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
        in_fence = False
        for line_number, line in enumerate(contents.splitlines(), 1):
            if line.startswith("```"):
                if not in_fence and line.strip() not in {
                    "```python",
                    "```pycon",
                    "```bash",
                    "```text",
                }:
                    raise ValueError(
                        f"Missing or unsupported language: {path}:{line_number}"
                    )
                in_fence = not in_fence
        if in_fence:
            raise ValueError(f"Unclosed code block in {path}")

        def check(
            match: re.Match[str], example_path: Path = path, source: str = contents
        ) -> str:
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
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                exec(
                    compile(formatted, str(example_path), "exec"),
                    {"__name__": "__main__"},
                )
            following = source[match.end() :]
            expected = re.match(
                r"\s*<!-- verified-output -->\n```text\n(.*?)```", following, re.S
            )
            if expected and capture.getvalue() != expected[1]:
                raise ValueError(
                    f"Displayed output differs from execution: {example_path}"
                )
            count += 1
            return f"```python\n{formatted}```"

        for transcript in re.findall(r"```pycon\n(.*?)```", contents, re.S):
            test = doctest.DocTestParser().get_doctest(
                transcript, {}, str(path), str(path), 0
            )
            result = doctest.DocTestRunner().run(test)
            if result.failed:
                raise ValueError(f"Failed reference doctest: {path}")
        contents = re.sub(r"```python\n(.*?)```", check, contents, flags=re.S)
        if args.format:
            path.write_text(contents)
    print(f"Verified {count} standalone Python examples with Ruff and execution.")


if __name__ == "__main__":
    main()

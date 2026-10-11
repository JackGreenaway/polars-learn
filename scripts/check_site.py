"""Validate local links, fragments, and assets in built HTML at the Pages path."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1] / "site"
PREFIX = "/polars-learn/"


class Page(HTMLParser):
    def __init__(self, path: Path):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        for name in ("href", "src"):
            if name in attributes:
                self.links.append(attributes[name])


def main():
    pages = {p.resolve(): Page(p) for p in ROOT.rglob("*.html")}
    checked = 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            route = unquote(url.path)
            if route.startswith("/"):
                if not route.startswith(PREFIX):
                    raise ValueError(f"Link escapes repository path: {path}: {link}")
                target = ROOT / route.removeprefix(PREFIX)
            else:
                target = path.parent / route if route else path
            if target.is_dir():
                target /= "index.html"
            target = target.resolve()
            if not target.is_relative_to(ROOT.resolve()) or not target.exists():
                raise ValueError(f"Missing local resource: {path}: {link}")
            if url.fragment and target in pages:
                if unquote(url.fragment) not in pages[target].ids:
                    raise ValueError(f"Missing anchor: {path}: {link}")
            checked += 1
    print(f"Verified {checked} rendered local links/assets across {len(pages)} pages.")


if __name__ == "__main__":
    main()

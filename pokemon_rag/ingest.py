"""Download and extract sections from the first three walkthrough pages."""

import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
BASE_URL = (
    "https://bulbapedia.bulbagarden.net/wiki/"
    "Walkthrough:Pok%C3%A9mon_Red_and_Blue"
)
URLS = {part: f"{BASE_URL}/Part_{part}" for part in range(1, 4)}


def download_page(part, url):
    """Read a cached page, or download it once and cache the response."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"part_{part}.html"
    if path.exists():
        print(f"Using cached Part {part}")
        return path.read_text(encoding="utf-8")

    print(f"Downloading Part {part}")
    response = requests.get(
        url,
        headers={"User-Agent": "PokemonRedExperiments-RAG/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    html = response.text
    path.write_text(html, encoding="utf-8")
    return html


def extract_sections(html, part, source):
    """Extract prose and list items with their heading hierarchy."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one(".mw-parser-output")
    if content is None:
        raise RuntimeError("Article content not found")
    for element in content.select(
        "table, .mw-editsection, .toc, .bv-toc-bar, .navbox, script, style"
    ):
        element.decompose()

    sections = []
    headings = {}
    paragraphs = []

    def save_section():
        if paragraphs:
            sections.append({
                "part": part,
                "location": headings.get(2, "Introduction"),
                "section": " > ".join(headings.values()) or "Introduction",
                "text": "\n".join(paragraphs),
                "source": source,
            })

    for element in content.find_all(["h2", "h3", "h4", "p", "li"]):
        if element.name in {"h2", "h3", "h4"}:
            save_section()
            paragraphs = []
            level = int(element.name[1])
            headings = {key: value for key, value in headings.items() if key < level}
            heading = element.get_text(" ", strip=True)
            if heading:
                headings[level] = heading
        elif element.name == "p":
            # List-item paragraphs are included with their containing item below.
            if element.find_parent("li") is None:
                text = element.get_text(" ", strip=True)
                if text:
                    paragraphs.append(text)
        else:
            # A nested list is processed separately, without duplicating its text.
            text = " ".join(
                text.strip()
                for text in element.descendants
                if isinstance(text, NavigableString)
                and text.strip()
                and text.find_parent("li") is element
            )
            if text:
                paragraphs.append(text)

    save_section()
    return sections


def main():
    sections = []
    for part, url in URLS.items():
        html = download_page(part, url)
        extracted = extract_sections(html, part, url)
        if not extracted:
            raise RuntimeError(f"No sections extracted from Part {part}")
        sections.extend(extracted)

    output = DATA_DIR / "sections.json"
    output.write_text(
        json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Extracted {len(sections)} sections")
    for section in sections:
        print(section["part"], section["section"], len(section["text"]))


if __name__ == "__main__":
    main()

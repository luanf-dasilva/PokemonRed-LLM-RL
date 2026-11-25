"""Build a separate, source-attributed corpus from configured wiki pages."""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
import yaml

from pokemon_rag.build_index import DATA_DIR
from pokemon_rag.ingest import extract_sections

CATALOG = Path(__file__).with_name("sources.yaml")
STRUCTURED = Path(__file__).with_name("structured_knowledge.json")
USER_AGENT = "PokemonRedExperiments-RAG/1.0"


def catalog(include_later=False):
    config = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    sources = list(config["sources"])
    walkthrough = config["bulbapedia_walkthrough"]
    last = walkthrough["last_part"] if include_later else walkthrough["initial_last_part"]
    for part in range(walkthrough["first_part"], last + 1):
        sources.append({"id": f"bulbapedia-walkthrough-{part}", "site": "bulbapedia",
            "category": "walkthroughs", "title": f"Walkthrough part {part}", "part": part,
            "url": f"https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Red_and_Blue/Part_{part}"})
    return sources


class Downloader:
    def __init__(self, offline=False):
        self.offline = offline
        self.robots = {}
        self.last_request = {}

    def fetch(self, source):
        path = DATA_DIR / "raw" / source["site"] / (source["id"] + ".html")
        if path.exists():
            print(f"Using cached {source['id']}", flush=True)
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise FileNotFoundError(f"Missing manually saved HTML: {path}")
        url = source["url"]
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            robots_url = origin + "/robots.txt"
            response = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=30)
            response.raise_for_status()
            parser = RobotFileParser(robots_url)
            parser.parse(response.text.splitlines())
            self.robots[origin] = parser
            self.last_request[origin] = time.monotonic()
        parser = self.robots[origin]
        if not parser.can_fetch(USER_AGENT, url):
            raise PermissionError(f"robots.txt disallows {url}")
        delay = max(1, parser.crawl_delay(USER_AGENT) or parser.crawl_delay("*") or 0)
        time.sleep(max(0, delay - (time.monotonic() - self.last_request[origin])))
        print(f"Downloading {source['id']}", flush=True)
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        self.last_request[origin] = time.monotonic()
        # Access errors stop the run. Do not retry using another identity or route.
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        return response.text


def annotate_sections(source, sections):
    result = []
    for section in sections:
        title = section["section"]
        # General Bulbapedia articles include other games and media. Exclude the
        # most clearly non-game sections, while retaining honest mixed scope.
        is_walkthrough = source["category"] == "walkthroughs"
        if not is_walkthrough:
            lower = title.casefold()
            if any(marker in lower for marker in (
                "in the anime", "in the manga", "in the tcg", "in animation", "in the tfg",
                "spin-off games", "side series games", "other appearances", "other media", "in the novels",
                "in live-action", "in the guidebooks", "merchandise", "trivia", "references", "related articles",
                "external links", "see also", "names >", "in other languages",
            )):
                continue
            generations = re.findall(r"\bgenerations?\s+([ivx]+)\b", title, re.IGNORECASE)
            if generations and not any(generation.upper() == "I" for generation in generations):
                continue
            if any(version in lower for version in (
                "diamond, pearl", "gold, silver", "heartgold", "black 2", "pokémon yellow"
            )) and not any(version in lower for version in ("red, blue", "red and blue")):
                continue
        game = ("Pokémon Red, Blue, and Yellow" if source["site"] == "strategywiki"
                else "Pokémon Red and Blue") if is_walkthrough else None
        scope = ("generation I walkthrough; may include Yellow" if source["site"] == "strategywiki"
                 else "generation I walkthrough") if is_walkthrough else "mixed games and generations"
        result.append({**section, "section": f"{source['title']} > {title}",
            "location": section["location"] if is_walkthrough else (
                source["title"] if source["category"] == "locations" else ""),
            "source_id": source["id"], "site": source["site"], "category": source["category"],
            "game": game,
            "generation": 1 if is_walkthrough else None,
            "scope": scope,
            "entities": [] if is_walkthrough else [source["title"]],
        })
    return result


def structured_sections():
    output = []
    for record in json.loads(STRUCTURED.read_text(encoding="utf-8")):
        facts = {k: v for k, v in record.items() if k not in {"source", "source_section", "source_note", "additional_sources"}}
        title = record.get("event") or f"{record['location']} {record['category']}"
        output.append({**record, "part": None, "section": title,
            "text": json.dumps(facts, ensure_ascii=False, sort_keys=True), "site": "curated",
            "game": "Pokémon Red and Blue", "generation": 1,
            "scope": "manually authored source-grounded knowledge"})
    return output


def ingest(sites, output, include_later=False, offline=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    sources = [source for source in catalog(include_later) if source["site"] in sites]
    if not sources:
        raise ValueError("No sources selected")
    base = json.loads((DATA_DIR / "sections.json").read_text(encoding="utf-8"))
    grouped = {"walkthroughs": [
        {**section, "site": "bulbapedia", "category": "walkthroughs", "game": "Pokémon Red and Blue",
         "generation": 1, "scope": "generation I walkthrough", "entities": []}
        for section in base], "locations": [], "mechanics": [], "progression": structured_sections()}
    downloader = Downloader(offline=offline)
    for source in sources:
        html = downloader.fetch(source)
        extracted = extract_sections(html, source.get("part"), source["url"])
        sections = annotate_sections(source, extracted)
        if not sections:
            raise RuntimeError(f"No usable sections extracted from {source['id']}")
        grouped[source["category"]].extend(sections)
        print(f"Extracted {source['id']}: {len(sections)} sections", flush=True)
    output.mkdir(parents=True)
    processed = output / "processed"
    processed.mkdir()
    for name, sections in grouped.items():
        (processed / f"{name}.json").write_text(json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8")
    all_sections = [section for sections in grouped.values() for section in sections]
    (output / "sections.json").write_text(json.dumps(all_sections, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(all_sections)} sections to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=("bulbapedia", "strategywiki"), action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for the expanded corpus")
    parser.add_argument("--include-later", action="store_true", help="Include walkthrough parts 7–17")
    parser.add_argument("--offline", action="store_true", help="Only use manually saved or cached HTML")
    args = parser.parse_args()
    ingest(set(args.site), args.output, args.include_later, args.offline)


if __name__ == "__main__":
    main()

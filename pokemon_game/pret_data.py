"""Parse a pinned subset of pret/pokered into lookups and semantic records."""

import argparse
import hashlib
import json
import re
from pathlib import Path

REVISION = "1982c0c7c5fee5ce28c6e3406cca1f4dc6b2b9c7"
SOURCE_DIR = Path("pokemon_rag/data/pret-pokered")
PINNED_MANIFEST = Path("research/pokered/source_manifest.json")
SOURCE_URL = f"https://github.com/pret/pokered/blob/{REVISION}/"


def number(expression):
    """Parse the numeric +/- expressions used by the selected const macros."""
    expression = expression.strip()
    if not re.fullmatch(r"(?:\$[0-9a-fA-F]+|\d+)(?:\s*[-+]\s*\d+)?", expression):
        raise ValueError(f"Unsupported constant expression: {expression}")
    pieces = re.split(r"\s*([-+])\s*", expression)
    value = int(pieces[0][1:], 16) if pieces[0].startswith("$") else int(pieces[0])
    if len(pieces) == 3:
        value += int(pieces[2]) * (1 if pieces[1] == "+" else -1)
    return value


def parse_constants(text, kind):
    value, active, in_macro = 0, False, False
    records = {}
    for line in text.splitlines():
        code, _, comment = line.partition(";")
        code = code.strip()
        if re.match(r"MACRO\s", code):
            in_macro = True
            continue
        if code == "ENDM":
            in_macro = False
            continue
        if in_macro:
            continue
        if re.match(r"const_def(?:\s|$)", code):
            parts = code.split(maxsplit=1)
            value = number(parts[1]) if len(parts) > 1 else 0
            active = True
            continue
        if not active:
            continue
        match = re.fullmatch(r"const_(next|skip)(?:\s+(.+))?", code)
        if match:
            arg = number(match.group(2)) if match.group(2) else 1
            value = arg if match.group(1) == "next" else value + arg
            continue
        if kind == "maps":
            match = re.fullmatch(r"map_const\s+([A-Z0-9_]+)\s*,\s*(\d+)\s*,\s*(\d+)", code)
        elif kind == "items":
            match = re.fullmatch(r"(const|add_hm|add_tm)\s+([A-Z0-9_]+)", code)
        else:
            match = re.fullmatch(r"const\s+([A-Z0-9_]+)", code)
        if not match:
            continue
        if kind == "items":
            operation, name = match.groups()
            name = ("HM_" if operation == "add_hm" else "TM_" if operation == "add_tm" else "") + name
        else:
            name = match.group(1)
        expected = re.search(r"\$([0-9A-Fa-f]+)\b", comment)
        if expected and int(expected.group(1), 16) != value:
            raise ValueError(f"Comment ID mismatch for {name}: {value} vs {expected.group(1)}")
        if name in records:
            raise ValueError(f"Duplicate constant: {name}")
        record = {"id": value}
        if kind == "maps":
            record.update({"width_blocks": int(match.group(2)), "height_blocks": int(match.group(3))})
        records[name] = record
        value += 1
    if not records:
        raise ValueError(f"No {kind} constants parsed")
    return records


def parse_map_graph(source_dir, maps, selected_files=None):
    """Extract directed header connections and explicit warps; dynamic warps remain unresolved."""
    graph, unresolved = [], []
    headers = source_dir / "data/maps/headers"
    objects = source_dir / "data/maps/objects"
    for header in sorted(headers.glob("*.asm")):
        if selected_files is not None and str(header.relative_to(source_dir)) not in selected_files:
            continue
        text = header.read_text(encoding="utf-8")
        match = re.search(r"^\s*map_header\s+\w+\s*,\s*([A-Z0-9_]+)", text, re.MULTILINE)
        if not match or match.group(1) not in maps:
            raise ValueError(f"Cannot identify source map in {header}")
        source = match.group(1)
        for direction, target in re.findall(r"^\s*connection\s+(\w+)\s*,\s*\w+\s*,\s*([A-Z0-9_]+)", text, re.MULTILINE):
            if target not in maps:
                raise ValueError(f"Unknown target map {target}")
            graph.append({"from": source, "to": target, "kind": "connection", "direction": direction,
                          "source": SOURCE_URL + str(header.relative_to(source_dir))})
        objects_file = objects / header.name
        if not objects_file.exists():
            continue
        for x, y, target, destination in re.findall(
                r"^\s*warp_event\s+(\d+)\s*,\s*(\d+)\s*,\s*([A-Z0-9_]+)\s*,\s*([^;\s]+)",
                objects_file.read_text(encoding="utf-8"), re.MULTILINE):
            edge = {"from": source, "to": target, "kind": "warp", "x": int(x), "y": int(y),
                    "destination_warp": destination, "source": SOURCE_URL + str(objects_file.relative_to(source_dir))}
            if target == "LAST_MAP":
                unresolved.append(edge)
            elif target in maps:
                graph.append(edge)
            else:
                raise ValueError(f"Unknown warp target {target}")
    if not list(headers.glob("*.asm")):
        raise ValueError("No map headers found")
    return graph, unresolved


def build(source_dir=SOURCE_DIR, pinned_manifest=PINNED_MANIFEST):
    source_dir = Path(source_dir)
    reference = json.loads(Path(pinned_manifest).read_text(encoding="utf-8"))
    if reference.get("revision") != REVISION:
        raise ValueError("Source manifest has an unexpected revision")
    for relative, expected in reference["files"].items():
        actual = hashlib.sha256((source_dir / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source file hash mismatch: {relative}")
    paths = {"maps": "constants/map_constants.asm", "events": "constants/event_constants.asm",
             "items": "constants/item_constants.asm", "species": "constants/pokemon_constants.asm"}
    lookups = {kind: parse_constants((source_dir / path).read_text(encoding="utf-8"), kind)
               for kind, path in paths.items()}
    graph, unresolved = parse_map_graph(source_dir, lookups["maps"], set(reference["files"]))
    files = [source_dir / relative for relative in reference["files"]]
    manifest = {"repository": "https://github.com/pret/pokered", "revision": REVISION,
                "files": {str(path.relative_to(source_dir)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                "graph_note": "Partial directed topology only; ignores collisions, event locks and unresolved LAST_MAP warps."}
    semantic = []
    for kind in ("maps", "events"):
        for name, record in lookups[kind].items():
            semantic.append({"category": "map_identifier" if kind == "maps" else "event_identifier",
                "section": name, "text": f"Pokémon Red {kind[:-1]} identifier {name} has numeric value {record['id']}.",
                "source": SOURCE_URL + paths[kind], "game": "Pokémon Red", "generation": 1})
    output = source_dir / "processed"
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("lookups", lookups), ("map_graph", {"edges": graph, "unresolved": unresolved}),
                        ("semantic_records", semantic), ("manifest", manifest)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"Parsed {len(lookups['maps'])} maps, {len(lookups['events'])} events, {len(lookups['items'])} items, {len(lookups['species'])} species")
    print(f"Graph: {len(graph)} directed connections/warps; {len(unresolved)} dynamic warps unresolved")
    return lookups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--manifest", type=Path, default=PINNED_MANIFEST)
    args = parser.parse_args()
    build(args.source_dir, args.manifest)

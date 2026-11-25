"""Read a small, explicit Pokémon Red state snapshot from byte-addressable RAM."""

import json
from pathlib import Path

LOOKUPS = Path("pokemon_rag/data/pret-pokered/processed/lookups.json")
MAP_ID, Y_COORD, X_COORD = 0xD35E, 0xD361, 0xD362
PARTY_COUNT, PARTY_SPECIES = 0xD163, 0xD164
PARTY_HP, PARTY_LEVEL, PARTY_MAX_HP, PARTY_STRIDE = 0xD16C, 0xD18C, 0xD18D, 44
BAG_COUNT, BAG_ITEMS, BAG_CAPACITY = 0xD31D, 0xD31E, 20
BADGES, EVENT_FLAGS = 0xD356, 0xD747
EVENTS = ("EVENT_GOT_STARTER", "EVENT_GOT_OAKS_PARCEL", "EVENT_OAK_GOT_PARCEL",
          "EVENT_GOT_POKEDEX", "EVENT_BEAT_BROCK")


def load_lookups(path=LOOKUPS):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_state(memory, lookups):
    """Read from a PyBoy-compatible memory object or a 64 KiB snapshot.

    Event flags use the pret bit indices; reads do not mutate emulator state.
    Call only when a coherent game snapshot is available, not during partial
    writes. This covers early-game fields, not a complete emulator encoder.
    """
    def byte(address):
        value = memory[address]
        if not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError(f"Invalid byte at {address:#06x}")
        return value

    def word(address):
        return (byte(address) << 8) | byte(address + 1)

    def names_by_id(kind):
        return {record["id"]: name for name, record in lookups[kind].items()}

    party_count = byte(PARTY_COUNT)
    bag_count = byte(BAG_COUNT)
    if party_count > 6 or bag_count > BAG_CAPACITY:
        raise ValueError("Invalid party or bag count; snapshot may be uninitialized")
    map_id = byte(MAP_ID)
    map_name = names_by_id("maps").get(map_id, f"UNKNOWN_MAP_{map_id:02X}")
    species = names_by_id("species")
    items = names_by_id("items")
    party = []
    for i in range(party_count):
        species_id = byte(PARTY_SPECIES + i)
        hp, max_hp = word(PARTY_HP + i * PARTY_STRIDE), word(PARTY_MAX_HP + i * PARTY_STRIDE)
        if max_hp == 0 or hp > max_hp:
            raise ValueError("Invalid party HP; snapshot may be inconsistent")
        party.append({"species_id": species_id, "species": species.get(species_id, f"UNKNOWN_{species_id:02X}"),
                      "level": byte(PARTY_LEVEL + i * PARTY_STRIDE), "hp": hp, "max_hp": max_hp})
    inventory = []
    for i in range(bag_count):
        item_id = byte(BAG_ITEMS + 2 * i)
        inventory.append({"id": item_id, "item": items.get(item_id, f"UNKNOWN_{item_id:02X}"),
                          "quantity": byte(BAG_ITEMS + 2 * i + 1)})
    completed = []
    for event in EVENTS:
        bit = lookups["events"][event]["id"]
        if byte(EVENT_FLAGS + bit // 8) & (1 << (bit % 8)):
            completed.append(event)
    total_hp, total_max_hp = sum(p["hp"] for p in party), sum(p["max_hp"] for p in party)
    return {"map_id": map_id, "map_name": map_name, "location": map_name.replace("_", " ").title(),
            "x": byte(X_COORD), "y": byte(Y_COORD), "party_size": party_count,
            "party_levels": [p["level"] for p in party], "party": party,
            "hp_fraction": total_hp / total_max_hp if total_max_hp else None,
            "inventory": inventory, "completed_events": completed,
            "has_starter": "EVENT_GOT_STARTER" in completed,
            "has_oaks_parcel": any(item["item"] == "OAKS_PARCEL" and item["quantity"] > 0 for item in inventory),
            "delivered_oaks_parcel": "EVENT_OAK_GOT_PARCEL" in completed,
            "badges": byte(BADGES).bit_count()}

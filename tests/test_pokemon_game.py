import tempfile
import unittest
from pathlib import Path

from pokemon_game.pret_data import number, parse_constants, parse_map_graph
from pokemon_game.state_encoder import encode_state
from pokemon_game.value_function import LocationValue


LOOKUPS = {
    "maps": {"PALLET_TOWN": {"id": 0}, "ROUTE_1": {"id": 12}, "VIRIDIAN_CITY": {"id": 1}},
    "items": {"OAKS_PARCEL": {"id": 70}, "POKE_BALL": {"id": 4}},
    "species": {"SQUIRTLE": {"id": 177}},
    "events": {name: {"id": value} for name, value in (
        ("EVENT_GOT_STARTER", 34), ("EVENT_GOT_OAKS_PARCEL", 57), ("EVENT_OAK_GOT_PARCEL", 56),
        ("EVENT_GOT_POKEDEX", 37), ("EVENT_BEAT_BROCK", 119))},
}


class PretParserTests(unittest.TestCase):
    def test_constant_offsets_macros_and_comments(self):
        text = """MACRO unused
 const INVALID
ENDM
 const_def
 const EVENT_A
 const_skip 2
 const_next $F0 - 2
 const EVENT_B
"""
        records = parse_constants(text, "events")
        self.assertEqual(records, {"EVENT_A": {"id": 0}, "EVENT_B": {"id": 238}})
        self.assertEqual(number("$28 + 2"), 42)
        with self.assertRaises(ValueError):
            number("__import__('os')")
        with self.assertRaises(ValueError):
            parse_constants("const_def\nconst WRONG ; $05", "items")

    def test_directed_connections_and_unresolved_dynamic_warps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            headers, objects = root / "data/maps/headers", root / "data/maps/objects"
            headers.mkdir(parents=True)
            objects.mkdir(parents=True)
            (headers / "PalletTown.asm").write_text("map_header PalletTown, PALLET_TOWN, OVERWORLD\nconnection north, Route1, ROUTE_1, 0\n")
            (objects / "PalletTown.asm").write_text("warp_event 1, 2, VIRIDIAN_CITY, 3\nwarp_event 4, 5, LAST_MAP, 1\n")
            edges, unresolved = parse_map_graph(root, LOOKUPS["maps"])
            self.assertEqual([(e["from"], e["to"]) for e in edges],
                             [("PALLET_TOWN", "ROUTE_1"), ("PALLET_TOWN", "VIRIDIAN_CITY")])
            self.assertEqual(unresolved[0]["to"], "LAST_MAP")


class StateTests(unittest.TestCase):
    def test_decodes_inventory_history_and_party_independently(self):
        memory = bytearray(65536)
        memory[0xD35E], memory[0xD362], memory[0xD361] = 1, 10, 15
        memory[0xD163], memory[0xD164] = 1, 177
        memory[0xD16C:0xD16E] = bytes([0, 16])
        memory[0xD18D:0xD18F] = bytes([0, 20])
        memory[0xD18C] = 7
        memory[0xD31D], memory[0xD31E], memory[0xD31F] = 1, 70, 1
        for bit in (34, 57):
            memory[0xD747 + bit // 8] |= 1 << (bit % 8)
        state = encode_state(memory, LOOKUPS)
        self.assertEqual((state["map_name"], state["x"], state["y"]), ("VIRIDIAN_CITY", 10, 15))
        self.assertEqual(state["party_levels"], [7])
        self.assertEqual(state["hp_fraction"], 0.8)
        self.assertTrue(state["has_oaks_parcel"])
        self.assertFalse(state["delivered_oaks_parcel"])
        memory[0xD31D] = 0
        memory[0xD747 + 56 // 8] |= 1 << (56 % 8)
        state = encode_state(memory, LOOKUPS)
        self.assertFalse(state["has_oaks_parcel"])
        self.assertTrue(state["delivered_oaks_parcel"])
        self.assertIn("EVENT_GOT_OAKS_PARCEL", state["completed_events"])

    def test_rejects_uninitialized_or_inconsistent_state(self):
        memory = bytearray(65536)
        memory[0xD163] = 7
        with self.assertRaises(ValueError):
            encode_state(memory, LOOKUPS)
        memory[0xD163] = 1
        with self.assertRaises(ValueError):
            encode_state(memory, LOOKUPS)


class ValueTests(unittest.TestCase):
    def test_map_distance_and_reached_goal(self):
        value = LocationValue(LOOKUPS, [{"from": "PALLET_TOWN", "to": "ROUTE_1"},
                                        {"from": "ROUTE_1", "to": "VIRIDIAN_CITY"}])
        goal = {"goal_type": "REACH_LOCATION", "target": "Viridian City"}
        self.assertAlmostEqual(value.evaluate({"map_name": "PALLET_TOWN"}, goal), 1/3)
        self.assertAlmostEqual(value.evaluate({"map_name": "ROUTE_1"}, goal), 1/2)
        self.assertEqual(value.evaluate({"map_name": "VIRIDIAN_CITY"}, goal), 1)
        with self.assertRaisesRegex(ValueError, "No route"):
            value.evaluate({"map_name": "VIRIDIAN_CITY"}, {"goal_type": "REACH_LOCATION", "target": "Pallet Town"})

    def test_same_map_coordinates_and_unsupported_goals(self):
        value = LocationValue(LOOKUPS, [])
        goal = {"goal_type": "REACH_LOCATION", "target": "Pallet Town", "target_x": 3, "target_y": 5}
        self.assertAlmostEqual(value.evaluate({"map_name": "PALLET_TOWN", "x": 1, "y": 4}, goal), 1/4)
        with self.assertRaises(NotImplementedError):
            value.evaluate({"map_name": "PALLET_TOWN"}, {"goal_type": "HEAL", "target": "party"})


if __name__ == "__main__":
    unittest.main()

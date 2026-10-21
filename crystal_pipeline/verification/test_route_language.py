"""Check route sentence content, missing conditions, and retrieval prompt integration."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.route_language import route_to_sentence
from modules.routes import compact_references


class RouteLanguageTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "formula": "LiFePO4", "synthesis_method": "Solid-state reaction",
            "precursors": ["Li2CO3", "FeC2O4·2H2O", "NH4H2PO4"],
            "conditions": {"temperature": "650 °C", "time": "12 h", "atmosphere": "Ar",
                           "other": "Heat at 5.0 °C/min. Cool slowly. Wash with water."},
            "equipment": "Tube furnace",
        }

    def test_complete_sentence_preserves_process_and_decimal(self):
        sentence = route_to_sentence(self.record)
        self.assertIn("using precursors listed as Li2CO3; FeC2O4·2H2O; NH4H2PO4", sentence)
        self.assertIn("temperature recorded as 650 °C, duration as 12 h, atmosphere as Ar", sentence)
        self.assertIn("Heat at 5.0 °C/min; Cool slowly; Wash with water", sentence)
        self.assertTrue(sentence.endswith("equipment listed as Tube furnace."))
        self.assertNotIn(". Cool", sentence)

    def test_missing_conditions_are_explicit(self):
        self.record.update(precursors=[], equipment=None)
        self.record["conditions"].update(time="null", atmosphere=None)
        sentence = route_to_sentence(self.record)
        self.assertIn("precursors listed as not specified", sentence)
        self.assertIn("duration as not specified, atmosphere as not specified", sentence)
        self.assertTrue(sentence.endswith("equipment listed as not specified."))

    def test_alternative_conditions_remain_in_order(self):
        self.record["conditions"].update(temperature=["220 °C", "RT"], time=["2 days", "3 days"], atmosphere=["Air", None])
        sentence = route_to_sentence(self.record)
        self.assertIn("temperature recorded as 220 °C; RT, duration as 2 days; 3 days, atmosphere as Air; not specified", sentence)

    def test_sentence_reaches_compact_llm_reference(self):
        reference = {"reference_id": "DB00000", "route_sentence": route_to_sentence(self.record), "record": self.record}
        compact = compact_references([reference])[0]
        self.assertEqual(compact["route_sentence"], reference["route_sentence"])
        self.assertEqual(compact["record"]["precursors"], self.record["precursors"])


if __name__ == "__main__":
    unittest.main()

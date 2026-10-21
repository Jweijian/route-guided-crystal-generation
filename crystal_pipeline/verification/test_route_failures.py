"""Regression checks for cuboctahedral coordination and batch failure isolation."""
import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import routes
from modules.chemistry import check_chemical_consistency
from modules.common import save_json
from modules.configuration import load_config


ROOT = Path(__file__).resolve().parents[1]


class RouteFailureTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")["routes"]
        self.schemas = {k: json.loads(Path(v).read_text(encoding="utf-8")) for k, v in self.config["schema_files"].items()}
        self.prompts = {k: Path(v).read_text(encoding="utf-8") for k, v in self.config["prompt_files"].items()}
        self.document = json.loads((ROOT / "verification/fixtures/batch/routes/F001/routes.json").read_text(encoding="utf-8"))

    def test_cuboctahedral_is_twelve_coordinate(self):
        document = self.document
        h = copy.deepcopy(document["routes"][0]["structure_hypotheses"][0])
        h["coordination"] = [{"center": "Ti", "neighbor": "O", "number": 12, "geometry": "cuboctahedral"}]
        for geometry in ("cuboctahedral", "立方八面体"):
            h["coordination"][0].update(geometry=geometry, number=12)
            check_chemical_consistency(h, document["target"], document["references"])
            h["coordination"][0]["number"] = 6
            with self.assertRaisesRegex(ValueError, "requires 12 neighbors"):
                check_chemical_consistency(h, document["target"], document["references"])
        h["coordination"][0].update(geometry="octahedral", number=6)
        check_chemical_consistency(h, document["target"], document["references"])
        h["coordination"][0]["number"] = 12
        with self.assertRaisesRegex(ValueError, "requires 6 neighbors"):
            check_chemical_consistency(h, document["target"], document["references"])

    def test_failed_route_retains_completed_route_and_can_resume(self):
        config = copy.deepcopy(self.config)
        config["max_routes"] = 2
        routes.configure(config, self.schemas, self.prompts)
        document = copy.deepcopy(self.document)
        document.update(settings_signature=routes.SETTINGS_SIGNATURE, max_routes=2)
        pending = copy.deepcopy(document["routes"][0])
        pending["route_id"] = "R02"
        expected = {"structure_hypotheses": pending.pop("structure_hypotheses"), "assessment": pending.pop("assessment")}
        document["routes"].append(pending)
        llm = SimpleNamespace(model=document["model"], thinking=document["thinking"], reasoning_effort=document["reasoning_effort"],
                              api_key="test", base_url="https://example.invalid", response_dir="unused")
        worker = SimpleNamespace(call=Mock(side_effect=ValueError("no valid output after 4 attempts")), session=Mock())
        with TemporaryDirectory() as directory:
            save_json(Path(directory) / "routes.json", document)
            with patch.object(routes, "JsonLLM", return_value=worker):
                result = routes.infer_routes(document["target"], document["references"], llm, directory)
            self.assertEqual(result["status"], "partial_failure")
            self.assertEqual(len(result["routes"][0]["structure_hypotheses"]), 1)
            self.assertIn("hypothesis_error", result["routes"][1])
            self.assertEqual(len(result["failures"]), 1)
            worker.call.side_effect = None
            worker.call.return_value = expected
            worker.call.reset_mock()
            with patch.object(routes, "JsonLLM", return_value=worker):
                resumed = routes.infer_routes(document["target"], document["references"], llm, directory)
            worker.call.assert_called_once()
            self.assertEqual(resumed["status"], "hypotheses_ready")
            self.assertEqual(resumed["failures"], [])
            self.assertNotIn("hypothesis_error", resumed["routes"][1])

    def test_formula_failure_does_not_stop_later_formulas(self):
        target = self.document["target"]
        retriever = Mock()
        retriever.retrieve.return_value = self.document["references"]
        llm = Mock()
        def infer(target, references, worker, folder):
            if Path(folder).name == "F001":
                raise ValueError("mock generation failure")
            return copy.deepcopy(self.document)
        with TemporaryDirectory() as directory, patch.object(routes, "Retriever", return_value=retriever), \
                patch.object(routes, "JsonLLM", return_value=llm), \
                patch.object(routes, "infer_routes", side_effect=infer):
            results = routes.generate_routes([target, target], self.config, self.schemas, self.prompts, directory)
            self.assertEqual([d["status"] for d in results], ["failed", "hypotheses_ready"])
            self.assertTrue((Path(directory) / "F001/failure.json").is_file())
            self.assertEqual(results[1]["formula_id"], "F002")
            self.assertEqual(len(json.loads((Path(directory) / "all_results.json").read_text(encoding="utf-8"))), 2)

    def test_streaming_finishes_route_cifs_before_next_description_request(self):
        config = copy.deepcopy(self.config)
        config["max_routes"] = 2
        routes.configure(config, self.schemas, self.prompts)
        document = copy.deepcopy(self.document)
        document.update(settings_signature=routes.SETTINGS_SIGNATURE, max_routes=2)
        first = document["routes"][0]
        expected = {"structure_hypotheses": first.pop("structure_hypotheses"), "assessment": first.pop("assessment")}
        second = copy.deepcopy(first)
        second["route_id"] = "R02"
        document["routes"].append(second)
        events = []
        worker = SimpleNamespace(session=Mock())
        def request(*args):
            events.append("description")
            return copy.deepcopy(expected)
        worker.call = request
        llm = SimpleNamespace(model=document["model"], thinking=document["thinking"], reasoning_effort=document["reasoning_effort"],
                              api_key="test", base_url="https://example.invalid", response_dir="unused")
        with TemporaryDirectory() as directory, patch.object(routes, "JsonLLM", return_value=worker):
            save_json(Path(directory) / "routes.json", document)
            def consume(result, route, hypothesis):
                saved = json.loads((Path(directory) / "routes.json").read_text(encoding="utf-8"))
                self.assertIn("structure_hypotheses", next(r for r in saved["routes"] if r["route_id"] == route["route_id"]))
                events.append("cif:" + route["route_id"])
            routes.infer_routes(document["target"], document["references"], llm, directory, on_hypothesis=consume)
        self.assertEqual(events, ["description", "cif:R01", "description", "cif:R02"])


if __name__ == "__main__":
    unittest.main()

"""Exercise the packaged two-stage workflow without weights, databases or API calls."""
import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline
from modules import routes
from modules.common import save_json
from modules.configuration import load_config
from modules.deepseek_api import complete_chat, redact_config

ROOT = Path(__file__).resolve().parents[1]


class StageTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")
        self.config["api"]["api_key"] = "offline-test-key"
        self.config["routes"]["api_key"] = "offline-test-key"
        self.schemas = {k: json.loads(Path(v).read_text(encoding="utf-8"))
                        for k, v in self.config["routes"]["schema_files"].items()}
        self.prompts = {k: Path(v).read_text(encoding="utf-8")
                        for k, v in self.config["routes"]["prompt_files"].items()}
        self.document = json.loads((ROOT / "verification/fixtures/batch/routes/F001/routes.json").read_text(encoding="utf-8"))
        self.document["formula_id"] = "F001"
        routes.configure(self.config["routes"], self.schemas, self.prompts)
        self.document["settings_signature"] = routes.SETTINGS_SIGNATURE
        loader = patch.object(pipeline, "load_config", side_effect=lambda _: copy.deepcopy(self.config))
        loader.start()
        self.addCleanup(loader.stop)

    def batch(self, folder):
        save_json(folder / "formulas.json", {"formulas": [self.document["target"]]})
        save_json(folder / "config_snapshot.json", redact_config(self.config))
        save_json(folder / "prompts_snapshot.json", {**self.prompts, "crystal": ""})
        save_json(folder / "routes/F001/routes.json", self.document)

    def test_description_stage_does_not_load_model(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "batch"
            self.batch(folder)
            with patch("modules.routes.generate_routes", return_value=[self.document]), \
                    patch("modules.crystal_generation.generate_crystals") as generate, \
                    patch("modules.crystalformer_crystal.CrystalFormerCrystalGenerator") as model:
                pipeline.run(ROOT / "config.json", resume_run=folder, stage="descriptions")
            generate.assert_not_called()
            model.assert_not_called()
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "descriptions_ready")

    def test_saved_descriptions_reused_without_any_api_or_retrieval(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "batch"
            self.batch(folder)
            original = (folder / "routes/F001/routes.json").read_bytes()
            record = {"formula_id": "F001", "route_id": "R01", "hypothesis_id": "H01",
                      "actual_formula": "TiOF2", "actual_space_group": 123,
                      "formula_matches": True, "space_group_matches": True, "status": "generated_unverified"}
            with patch.object(pipeline, "validate_api", side_effect=AssertionError("Unexpected API validation")), \
                    patch("requests.Session.post", side_effect=AssertionError("Unexpected API")), \
                    patch("modules.routes.generate_routes", side_effect=AssertionError("Unexpected descriptions")), \
                    patch("modules.routes.Retriever", side_effect=AssertionError("Unexpected retrieval")), \
                    patch("modules.crystal_generation.generate_crystals", return_value=[record]) as generate:
                pipeline.run(ROOT / "config.json", resume_run=folder, stage="crystals")
            self.assertEqual(generate.call_args.args[1]["backend"], "crystalformer")
            self.assertEqual(generate.call_args.args[3], folder / "crystals_crystalformer")
            self.assertEqual((folder / "routes/F001/routes.json").read_bytes(), original)
            self.assertIn("crystals_crystalformer/F001/R01/H01/candidate.cif", (folder / "results.md").read_text(encoding="utf-8"))

    def test_missing_saved_batch_is_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, "requires --resume-run"):
            pipeline.run(ROOT / "config.json", stage="crystals")

    def test_deepseek_request_and_snapshot_redaction(self):
        session = Mock()
        complete_chat(session, "offline-test-key", "deepseek-v4-pro", "https://api.deepseek.com", [], 32768, 600, "low")
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["max_tokens"], 32768)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertNotIn("offline-test-key", json.dumps(redact_config(self.config)))


if __name__ == "__main__":
    unittest.main()

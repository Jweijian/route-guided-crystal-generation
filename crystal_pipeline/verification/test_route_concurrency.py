"""Verify real formula overlap, ordered checkpoints and serial crystal callbacks."""
import copy
import json
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import routes
from modules.common import save_json
from modules.configuration import load_config


ROOT = Path(__file__).resolve().parents[1]


class FormulaConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")["routes"]
        self.config.update(formula_workers=4, max_routes=1)
        self.schemas = {k: json.loads(Path(v).read_text(encoding="utf-8")) for k, v in self.config["schema_files"].items()}
        self.prompts = {k: Path(v).read_text(encoding="utf-8") for k, v in self.config["prompt_files"].items()}
        self.formulas = [{"formula": element + "O", "oxidation_states": {element: 2, "O": -2}}
                         for element in ("Mg", "Ca", "Sr", "Ba")]
        self.sessions = []
        retriever = Mock()
        retriever.retrieve.return_value = []
        loader = patch.object(routes, "Retriever", return_value=retriever)
        loader.start()
        self.addCleanup(loader.stop)

        def create_llm(*args, **kwargs):
            session = Mock()
            self.sessions.append(session)
            return SimpleNamespace(session=session)

        llm = patch.object(routes, "JsonLLM", side_effect=create_llm)
        llm.start()
        self.addCleanup(llm.stop)

    def test_four_formulas_overlap_and_checkpoints_keep_input_order(self):
        barrier = threading.Barrier(4, timeout=10)
        last_saved = threading.Event()
        threads = []
        checkpoints = []

        def infer(target, references, llm, folder):
            threads.append(threading.get_ident())
            barrier.wait()
            if Path(folder).name == "F001":
                self.assertTrue(last_saved.wait(10), "F004 must be saved before F001 finishes")
            return {"target": target, "routes": [], "status": "hypotheses_ready", "elapsed_seconds": 0}

        def checkpoint(path, value):
            save_json(path, value)
            if Path(path).name == "all_results.json":
                checkpoints.append(copy.deepcopy(value))
                if any(item["formula_id"] == "F004" for item in value):
                    last_saved.set()

        with TemporaryDirectory() as directory, patch.object(routes, "infer_routes", side_effect=infer), \
                patch.object(routes, "save_json", side_effect=checkpoint):
            results = routes.generate_routes(self.formulas, self.config, self.schemas, self.prompts, directory)
            saved = json.loads((Path(directory) / "all_results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, results)
        self.assertEqual(len(set(threads)), 4)
        self.assertNotIn("F001", [item["formula_id"] for item in checkpoints[0]])
        self.assertEqual([item["formula_id"] for item in results], ["F001", "F002", "F003", "F004"])
        self.assertEqual([item["target"] for item in results], self.formulas)
        for session in self.sessions:
            session.close.assert_called_once()

    def test_parallel_failure_keeps_other_formulas_and_sessions(self):
        barrier = threading.Barrier(4, timeout=10)

        def infer(target, references, llm, folder):
            barrier.wait()
            if Path(folder).name == "F002":
                raise ValueError("mock formula failure")
            return {"target": target, "routes": [], "status": "hypotheses_ready", "elapsed_seconds": 0}

        with TemporaryDirectory() as directory, patch.object(routes, "infer_routes", side_effect=infer):
            results = routes.generate_routes(self.formulas, self.config, self.schemas, self.prompts, directory)
            self.assertTrue((Path(directory) / "F002/failure.json").is_file())
            self.assertEqual([item["status"] for item in results],
                             ["hypotheses_ready", "failed", "hypotheses_ready", "hypotheses_ready"])
            self.assertEqual(len(json.loads((Path(directory) / "all_results.json").read_text(encoding="utf-8"))), 4)
        for session in self.sessions:
            session.close.assert_called_once()

    def test_shared_crystal_callback_remains_serial(self):
        main_thread = threading.get_ident()
        completed = []

        def infer(target, references, llm, folder, on_hypothesis):
            self.assertEqual(threading.get_ident(), main_thread)
            document = {"target": target, "routes": [], "status": "hypotheses_ready", "elapsed_seconds": 0}
            on_hypothesis(document, {}, {})
            return document

        with TemporaryDirectory() as directory, patch.object(routes, "infer_routes", side_effect=infer):
            routes.generate_routes(self.formulas, self.config, self.schemas, self.prompts, directory,
                                   on_hypothesis=lambda document, route, hypothesis: completed.append(document["formula_id"]))
        self.assertEqual(completed, ["F001", "F002", "F003", "F004"])


if __name__ == "__main__":
    unittest.main()

"""Check real CIF round trips and the fixed-formula CrystalFormer command contract."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from pymatgen.core import Lattice, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.crystalformer_crystal import CrystalFormerCrystalGenerator
from modules.crystal_generation import generate_crystals


class CrystalFormerTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        checkpoint = self.directory / "weights.pkl"
        checkpoint.touch()
        self.config = {"backend": "crystalformer", "model_variant": "crystalformer", "seed": 42,
                       "model_path": str(checkpoint), "project_dir": str(self.directory),
                       "num_samples": 8, "batch_size": 8, "temperature": 1.0, "top_p": 1.0,
                       "symprec": 0.01, "angle_tolerance": 5}
        self.structures = [Structure.from_spacegroup(225, Lattice.cubic(4.2),
                          ["Mg", "O"], [[0, 0, 0], [.5, .5, .5]])]
        self.document = {"formula_id": "F001", "target": {"formula": "MgO"}, "routes": [{
            "route_id": "R01", "structure_hypotheses": [{"hypothesis_id": "H01", "description": "Saved description",
            "space_group_candidates": [{"number": 225}, {"number": 221}]}]}]}
        sampler = patch("modules.crystalformer_crystal.subprocess.run", side_effect=self.convert)
        self.sampler = sampler.start()
        self.addCleanup(sampler.stop)

    def convert(self, command, **kwargs):
        if command[2] == "scripts/awl2struct.py":
            folder = Path(command[command.index("--output_path") + 1])
            pd.DataFrame({"cif": [str(s.as_dict()) for s in self.structures]}).to_csv(
                folder / "output_MgO_struct.csv", index=False)

    def test_factory_filters_and_preserves_all_valid_cifs(self):
        wrong_composition = self.structures[0].copy()
        wrong_composition.replace_species({"Mg": "Ca"})
        wrong_group = Structure(Lattice.cubic(4.2), ["Mg", "O"], [[0, 0, 0], [.5, .5, .5]])
        self.structures = [wrong_composition, wrong_group, *self.structures, *self.structures]
        result = generate_crystals([self.document], self.config, "", self.directory / "outputs")[0]
        folder = self.directory / "outputs/F001/R01/H01"
        self.assertEqual(result["accepted_count"], 2)
        self.assertFalse(result["description_constraints_applied"])
        self.assertEqual(result["actual_space_group"], 225)
        self.assertEqual(len(list(folder.glob("candidate_*.cif"))), 2)
        self.assertEqual((folder / "candidate.cif").read_bytes(), (folder / "candidate_003.cif").read_bytes())
        self.assertEqual(Structure.from_file(result["cif_file"]).composition.reduced_formula, "MgO")
        self.assertEqual(len(json.loads((folder / "generation_failures.json").read_text())), 2)
        command = self.sampler.call_args_list[0].args[0]
        for option, expected in (("--formula", "MgO"), ("--spacegroup", "225"), ("--num_samples", "8"),
                                 ("--batchsize", "8"), ("--restore_path", self.config["model_path"])):
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertEqual(self.sampler.call_args_list[0].kwargs["env"]["JAX_PLATFORM_NAME"], "cpu")

    def test_no_matching_samples_records_failure_and_removes_stale_cif(self):
        self.structures = []
        folder = self.directory / "outputs/F001/R01/H01"
        folder.mkdir(parents=True)
        (folder / "candidate.cif").write_text("stale")
        results = generate_crystals([self.document], self.config, "", self.directory / "outputs", raise_on_failure=False)
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("no accepted MgO / SG 225", results[0]["error"])
        self.assertFalse((folder / "candidate.cif").exists())
        self.assertEqual(self.sampler.call_count, 2)

    def test_missing_checkpoint_never_starts_randomly_initialized_sampler(self):
        self.config["model_path"] = str(self.directory / "missing.pkl")
        with self.assertRaisesRegex(AssertionError, "Missing CrystalFormer checkpoint"):
            CrystalFormerCrystalGenerator(self.config, "")
        self.sampler.assert_not_called()


if __name__ == "__main__":
    unittest.main()

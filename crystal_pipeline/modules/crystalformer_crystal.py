"""Use the fixed-formula CrystalFormer notebook's sampling and conversion commands."""
import os
import shutil
import subprocess
import sys
import time
from ast import literal_eval
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .chemistry import composition_key
from .common import save_json
from .crystal_generation import geometry_metrics


class CrystalFormerCrystalGenerator:
    def __init__(self, config, prompt):
        self.config = config
        assert Path(config["model_path"]).is_file(), f"Missing CrystalFormer checkpoint: {config['model_path']}"

    def generate(self, description, folder, seed, expected_formula, expected_space_group):
        folder = Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        for path in [*folder.glob("candidate*.cif"), folder / "metadata.json", folder / "generation_failures.json"]:
            path.unlink(missing_ok=True)
        (folder / "description.txt").write_text(description, encoding="utf-8")
        started = time.perf_counter()
        process_env = {**os.environ, "JAX_PLATFORM_NAME": "cpu", "PYTHONIOENCODING": "utf-8"}
        with (folder / "sampling.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-u", "main.py", "--optimizer", "none",
                 "--restore_path", self.config["model_path"], "--formula", expected_formula,
                 "--spacegroup", str(expected_space_group), "--num_samples", str(self.config["num_samples"]),
                 "--batchsize", str(self.config["batch_size"]), "--num_io_process", "1",
                 "--seed", str(seed), "--temperature", str(self.config["temperature"]),
                 "--top_p", str(self.config["top_p"]), "--save_path", folder.as_posix() + "/"],
                cwd=self.config["project_dir"], env=process_env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        with (folder / "conversion.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-u", "scripts/awl2struct.py", "--output_path", folder.as_posix() + "/",
                 "--formula", expected_formula, "--num_io_process", "1"],
                cwd=self.config["project_dir"], env=process_env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        rows = pd.read_csv(folder / f"output_{expected_formula}_struct.csv")
        candidates, failures = [], []
        for index, value in enumerate(rows["cif"], 1):
            cif_path = folder / f"candidate_{index:03d}.cif"
            try:
                structure = Structure.from_dict(literal_eval(value))
                CifWriter(structure, symprec=self.config["symprec"], refine_struct=False).write_file(cif_path)
                structure = Structure.from_file(cif_path)
                if composition_key(structure.composition) != composition_key(expected_formula):
                    raise ValueError(f"Generated composition {structure.composition} does not match {expected_formula}")
                symmetry = SpacegroupAnalyzer(structure, symprec=self.config["symprec"],
                                               angle_tolerance=self.config["angle_tolerance"])
                actual_group = symmetry.get_space_group_number()
                if actual_group != expected_space_group:
                    raise ValueError(f"Actual space group {actual_group} does not match {expected_space_group}")
                geometry_warnings, minimum_distance = geometry_metrics(structure)
                candidates.append({
                    "cif_file": str(cif_path), "composition_filtered_index": index,
                    "actual_formula": structure.composition.reduced_formula, "actual_space_group": actual_group,
                    "actual_space_group_symbol": symmetry.get_space_group_symbol(), "atom_count": len(structure),
                    "formula_matches": True, "space_group_matches": True,
                    "geometry_warnings": geometry_warnings, "minimum_interatomic_distance_A": minimum_distance,
                })
            except (ValueError, RuntimeError) as error:
                cif_path.unlink(missing_ok=True)
                failures.append({"composition_filtered_index": index, "error": str(error)})
        save_json(folder / "generation_failures.json", failures)
        if not candidates:
            raise ValueError(f"CrystalFormer produced no accepted {expected_formula} / SG {expected_space_group} "
                             f"candidates from {self.config['num_samples']} samples; inspect sampling.log, "
                             "conversion.log and generation_failures.json")
        shutil.copyfile(candidates[0]["cif_file"], folder / "candidate.cif")
        result = {
            **candidates[0], "cif_file": str(folder / "candidate.cif"),
            "seed": seed, "model_variant": "crystalformer", "model": "CrystalFormer-CSP",
            "model_path": self.config["model_path"], "device": "cpu",
            "expected_formula": expected_formula, "expected_space_group": expected_space_group,
            "conditioning_fields": ["formula", "space_group_number"], "description_constraints_applied": False,
            "lattice_source": "CrystalFormer-CSP pretrained model",
            "num_samples": self.config["num_samples"], "composition_matched_count": len(rows),
            "accepted_count": len(candidates), "candidates": candidates,
            "selection": "first candidate passing composition, actual space group and geometry checks",
            "elapsed_seconds": time.perf_counter() - started, "status": "generated_unverified",
        }
        save_json(folder / "metadata.json", result)
        return result

    def close(self):
        pass  # Each subprocess releases its own model and JAX runtime.

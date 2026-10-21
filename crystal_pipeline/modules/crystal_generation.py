"""Geometry checks and batch generation for the current CrystalFormer backend."""
from pathlib import Path
import numpy as np
from .common import save_json


def geometry_metrics(structure):
    """Check periodic distances between distinct atoms; short bonds are warnings."""
    if len(structure) < 2:
        return [], None
    distances = structure.distance_matrix[np.triu_indices(len(structure), k=1)]
    if not np.all(np.isfinite(distances)):
        raise ValueError("Interatomic distances must be finite")
    minimum = float(distances.min())
    if minimum < 0.001:
        raise ValueError(f"Overlapping atoms under periodic boundaries: {minimum:g} Å < 0.001 Å")
    warnings = [f"Short interatomic distance: {minimum:g} Å < 0.5 Å; geometry is unverified"] if minimum < 0.5 else []
    return warnings, minimum


def generate_crystals(documents, config, prompt, output_dir, generator=None, results=None, raise_on_failure=True):
    owns_generator = generator is None
    if owns_generator:
        from .crystalformer_crystal import CrystalFormerCrystalGenerator
        generator = CrystalFormerCrystalGenerator(config, prompt)
    if results is None:
        results = []
    failures = []
    index = len(results)
    try:
        for document in documents:
            for route in document["routes"]:
                for hypothesis in route.get("structure_hypotheses", []):
                    identity = {"formula_id": document["formula_id"], "route_id": route["route_id"],
                                "hypothesis_id": hypothesis["hypothesis_id"]}
                    folder = Path(output_dir) / identity["formula_id"] / identity["route_id"] / identity["hypothesis_id"]
                    seed = config["seed"] + index
                    index += 1
                    print(f"Crystal {identity['formula_id']}/{identity['route_id']}/{identity['hypothesis_id']}: generating...", flush=True)
                    try:
                        result = generator.generate(hypothesis["description"], folder, seed,
                            document["target"]["formula"], hypothesis["space_group_candidates"][0]["number"])
                    except Exception as error:
                        (folder / "candidate.cif").unlink(missing_ok=True)
                        result = {"status": "failed", "error": str(error), "seed": seed,
                                  "model_variant": "crystalformer",
                                  "model_path": config.get("model_path"), "model": config.get("model")}
                        save_json(folder / "metadata.json", result)
                        failures.append(identity)
                    results.append({**identity, **result})
                    save_json(Path(output_dir) / "all_results.json", results)
                    print(f"Crystal {identity['formula_id']}/{identity['route_id']}/{identity['hypothesis_id']}: {result['status']}", flush=True)
    finally:
        if owns_generator:
            generator.close()
    if failures and raise_on_failure:
        raise ValueError(f"{len(failures)} crystal generations failed; inspect {Path(output_dir) / 'all_results.json'} and sampling.log")
    return results

"""Run real ELM, database retrieval and CrystalFormer inference without an API key."""
import sys
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "crystal_pipeline"))

from modules.configuration import load_config
from modules.crystalformer_crystal import CrystalFormerCrystalGenerator
from modules.formula_generation import generate_formulas
from modules.retrieval import Retriever

config = load_config(root / "crystal_pipeline/config.json")
output = root / "crystal_pipeline/outputs" / datetime.now().strftime("smoke_%Y%m%d_%H%M%S")
config["elm"]["num_formulas"] = 1
formulas = generate_formulas(config["elm"], output / "formulas.json")
assert len(formulas) == 1
print(f"ELM: {formulas[0]['formula']}", flush=True)
references = Retriever(config["routes"]["database_file"]).retrieve("TiOF2", {"Ti": 4, "O": -2, "F": -1}, 3, 3)
assert references
print(f"RAG: {len(references)} references", flush=True)
result = CrystalFormerCrystalGenerator(config["crystalformer"], "").generate(
    "LiCu2OF3, space group 14", output / "crystalformer", 42, "LiCu2OF3", 14,
)
assert result["formula_matches"] and result["space_group_matches"]
print(f"CrystalFormer: {result['accepted_count']} accepted candidates; CIF: {result['cif_file']}", flush=True)

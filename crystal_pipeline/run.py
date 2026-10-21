"""Run all three stages with one annotated config.json."""
import argparse
from pathlib import Path

from pipeline import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formula → RAG routes and descriptions → selected CIF generator")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--resume-run", type=Path, help="Continue a saved batch without changing config.json")
    parser.add_argument("--stage", choices=("all", "descriptions", "crystals"), default="all",
                        help="Run the whole pipeline, save descriptions only, or generate CIFs from saved descriptions")
    parser.add_argument("--crystal-model", choices=("crystalformer",),
                        help="The current version uses local CrystalFormer-CSP")
    args = parser.parse_args()
    run(args.config, resume_run=args.resume_run, stage=args.stage, crystal_model=args.crystal_model)

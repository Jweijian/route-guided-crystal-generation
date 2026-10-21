"""Read the single annotated JSON configuration and resolve portable paths."""
import json
from pathlib import Path

from jsonschema import validate
from .deepseek_api import api_settings


def without_comments(value):
    if isinstance(value, dict):
        return {key: without_comments(item) for key, item in value.items() if key != "_comments"}
    if isinstance(value, list):
        return [without_comments(item) for item in value]
    return value


def load_config(path):
    path = Path(path).resolve()
    config = without_comments(json.loads(path.read_text(encoding="utf-8")))
    api_section = "api" if "api" in config else "chatgpt"
    if api_section in config:
        config[api_section] = api_settings(config[api_section])
        config["routes"].update(config[api_section])
    validate(config["pipeline"]["formula_source"], {"enum": ["elm", "file"]})
    validate(config["elm"]["formula_type"], {"enum": ["ionic", "alloy"]})
    for section, names in {"elm": ["num_formulas", "max_sampling_attempts", "max_count", "max_length"],
                           "routes": ["max_routes", "max_structures_per_route", "max_workers", "top_n",
                                      "reference_limit", "max_output_tokens", "repair_attempts"],
                           "crystalformer": ["num_samples", "batch_size"]}.items():
        for name in names:
            validate(config[section][name], {"type": "integer", "minimum": 1})
    validate(config["routes"].get("formula_workers", 1), {"type": "integer", "minimum": 1})
    for section, names in {"pipeline": ["formula_input_file", "output_dir"],
                           "elm": ["checkpoint"], "routes": ["database_file"],
                           "crystalformer": ["project_dir", "model_path"]}.items():
        for name in names:
            config[section][name] = str((path.parent / config[section][name]).resolve())
    for name in ("prompt_files", "schema_files"):
        config["routes"][name] = {key: str((path.parent / value).resolve())
                                  for key, value in config["routes"][name].items()}
    if config["pipeline"]["resume_run"]:
        config["pipeline"]["resume_run"] = str((path.parent / config["pipeline"]["resume_run"]).resolve())
    return config

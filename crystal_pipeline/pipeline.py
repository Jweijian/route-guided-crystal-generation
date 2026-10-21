"""Formula → local RAG / LLM → English description → selected CIF generator."""
import copy
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from modules.common import save_json
from modules.configuration import load_config
from modules.deepseek_api import redact_config, validate_api


def run(config_path, resume_run=None, stage="all", crystal_model=None):
    if stage not in ("all", "descriptions", "crystals"):
        raise ValueError("stage must be all, descriptions or crystals")
    if crystal_model not in (None, "crystalformer"):
        raise ValueError("This version uses crystalformer")
    config = load_config(config_path)
    api_section = "api" if "api" in config else "chatgpt"
    if api_section in config and stage != "crystals":
        validate_api(config[api_section])
    if resume_run is not None:
        config["pipeline"]["resume_run"] = str(Path(resume_run).resolve())
    options = config["pipeline"]
    folder = (Path(options["resume_run"]) if options["resume_run"] else
              Path(options["output_dir"]) / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]))
    if stage == "crystals" and (not options["resume_run"] or not (folder / "formulas.json").is_file()
                               or not (folder / "config_snapshot.json").is_file()
                               or not (folder / "prompts_snapshot.json").is_file()):
        raise ValueError("The crystals stage requires --resume-run pointing to a batch with saved descriptions")
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = redact_config(config)
    # Resume must preserve formula order and all generation settings.
    saved_snapshot = folder / "config_snapshot.json"
    if saved_snapshot.exists():
        previous = json.loads(saved_snapshot.read_text(encoding="utf-8"))
        previous["pipeline"]["resume_run"] = ""
        comparable = copy.deepcopy(snapshot)
        comparable["pipeline"]["resume_run"] = ""
        # CrystalFormer settings have their own CIF-stage snapshot and do not change saved descriptions.
        previous.pop("crystalformer", None)
        comparable.pop("crystalformer", None)
        if previous != comparable:
            raise ValueError("Run configuration has changed; clear resume_run to create a new run")
    save_json(saved_snapshot, snapshot)
    prompt_files = config["routes"]["prompt_files"]
    prompts = {key: Path(value).read_text(encoding="utf-8") for key, value in prompt_files.items()}
    schemas = {key: json.loads(Path(value).read_text(encoding="utf-8"))
               for key, value in config["routes"]["schema_files"].items()}
    crystal_prompt = ""
    prompt_snapshot = {**prompts, "crystal": crystal_prompt}
    if (folder / "prompts_snapshot.json").exists():
        if json.loads((folder / "prompts_snapshot.json").read_text(encoding="utf-8")) != prompt_snapshot:
            raise ValueError("Prompts have changed; use a new run directory")
    save_json(folder / "prompts_snapshot.json", prompt_snapshot)
    status = {"status": "running", "stage": "formulas"}
    save_json(folder / "status.json", status)
    generator = None
    try:
        if (folder / "formulas.json").exists():
            formulas = json.loads((folder / "formulas.json").read_text(encoding="utf-8"))["formulas"]
        elif options["formula_source"] == "elm":
            from modules.formula_generation import generate_formulas
            formulas = generate_formulas(config["elm"], folder / "formulas.json")
        else:
            formulas = json.loads(Path(options["formula_input_file"]).read_text(encoding="utf-8"))["formulas"]
            save_json(folder / "formulas.json", {"formulas": formulas})
        from modules.chemistry import validate_target
        for formula in formulas:
            validate_target(formula["formula"], formula["oxidation_states"])
        if not formulas:
            raise ValueError("Formula input must contain at least one formula")
        print(f"Formulas: {', '.join(f['formula'] for f in formulas)}", flush=True)
        from modules.crystal_generation import generate_crystals
        from modules.crystalformer_crystal import CrystalFormerCrystalGenerator as CrystalGenerator
        crystal_config = {**config["crystalformer"], **config["crystal"], "backend": "crystalformer",
                          "model_variant": "crystalformer", "model": "CrystalFormer-CSP", "device": "cpu"}
        crystal_prompt = ""
        variant = "crystalformer"
        crystal_dir = "crystals_crystalformer"
        structures = []

        def generate_immediately(document, route, hypothesis):
            nonlocal generator
            status.update(stage="crystals", crystal_model=variant)
            save_json(folder / "status.json", status)
            if generator is None:
                save_json(folder / crystal_dir / "config_snapshot.json", redact_config(crystal_config))
                generator = CrystalGenerator(crystal_config, crystal_prompt)
            candidate = {"formula_id": document["formula_id"], "target": document["target"],
                         "routes": [{**route, "structure_hypotheses": [hypothesis]}]}
            generate_crystals([candidate], crystal_config, crystal_prompt, folder / crystal_dir,
                              generator=generator, results=structures, raise_on_failure=False)
            status["stage"] = "routes"
            save_json(folder / "status.json", status)

        status["stage"] = "routes"
        save_json(folder / "status.json", status)
        if stage == "crystals":
            from modules.routes import load_descriptions
            documents = load_descriptions(formulas, config["routes"], schemas, prompts, folder / "routes")
        else:
            from modules.routes import generate_routes
            if stage == "all":
                documents = generate_routes(formulas, config["routes"], schemas, prompts, folder / "routes",
                                            on_hypothesis=generate_immediately)
            else:
                documents = generate_routes(formulas, config["routes"], schemas, prompts, folder / "routes")
        if generator is not None:
            generator.close()
            generator = None
        statistics = []
        failures = [{"formula_id": d["formula_id"], "formula": d["target"]["formula"],
                     "errors": d.get("failures", ["Saved route description is incomplete"])}
                    for d in documents if d["status"] != "hypotheses_ready"]
        save_json(folder / "route_failures.json", failures)
        for document in documents:
            response_dir = folder / "routes" / document["formula_id"] / "responses"
            responses = [json.loads(p.read_text(encoding="utf-8")) for p in response_dir.glob("*.json")
                         if not p.name.endswith("_error.json")]
            statistics.append({"formula": document["target"]["formula"],
                "route_count": len(document["routes"]),
                "structure_hypothesis_count": sum(len(r.get("structure_hypotheses", [])) for r in document["routes"]),
                "route_wall_seconds": document["elapsed_seconds"], "api_requests": len(responses),
                "validation_failures": len(list(response_dir.glob("*_error.json"))),
                "cumulative_api_seconds": sum(r["elapsed_seconds"] for r in responses),
                "output_tokens": sum(r["usage"].get("completion_tokens", 0) for r in responses)})
        if stage != "crystals":
            save_json(folder / "run_statistics.json", statistics)
        descriptions = []
        for document in documents:
            descriptions.append(f"## {document['formula_id']}: {document['target']['formula']}")
            if document["status"] != "hypotheses_ready":
                descriptions.append("\nDescription failures: " + " | ".join(document.get("failures", [])))
            for route in document["routes"]:
                descriptions.append(f"\n**{route['route_id']} ({route['route_basis']}) — Route:** {route['route_summary']}")
                for hypothesis in route.get("structure_hypotheses", []):
                    descriptions.append(f"\n**{hypothesis['hypothesis_id']} — Structure:** {hypothesis['description']}")
        (folder / "descriptions.md").write_text("\n".join(descriptions) + "\n", encoding="utf-8")
        hypothesis_count = sum(len(r.get("structure_hypotheses", [])) for d in documents for r in d["routes"])
        if stage == "descriptions" or not hypothesis_count:
            status.update(status="partial_failure" if failures else "descriptions_ready", stage="descriptions", formula_count=len(formulas),
                          route_count=sum(len(d["routes"]) for d in documents),
                          hypothesis_count=hypothesis_count, failed_formula_count=len(failures))
            save_json(folder / "status.json", status)
            print(f"Descriptions finished: {status['route_count']} routes, {status['hypothesis_count']} valid structures, "
                  f"{len(failures)} formulas with failures.\nResults: {folder}", flush=True)
            return folder
        if stage == "crystals":
            status.update(stage="crystals", crystal_model=variant)
            save_json(folder / "status.json", status)
            save_json(folder / crystal_dir / "config_snapshot.json", redact_config(crystal_config))
            structures = generate_crystals(documents, crystal_config, crystal_prompt, folder / crystal_dir,
                                           raise_on_failure=False)
        report = []
        if failures:
            report.append(f"{len(failures)} formulas have route/description failures; see [failure list](route_failures.json).")
        for document in documents:
            report.append(f"## {document['formula_id']}: {document['target']['formula']}")
            for route in document["routes"]:
                report.append(f"\n**{route['route_id']} ({route['route_basis']}) — Route:** {route['route_summary']}")
                for h in route.get("structure_hypotheses", []):
                    item = next(s for s in structures if (s["formula_id"], s["route_id"], s["hypothesis_id"]) ==
                                (document["formula_id"], route["route_id"], h["hypothesis_id"]))
                    if item.get("status") == "failed":
                        report.append(f"\n**{h['hypothesis_id']} — Structure:** {h['description']}\n\nCIF generation failed: {item['error']}")
                        continue
                    relative = f"{crystal_dir}/{item['formula_id']}/{item['route_id']}/{item['hypothesis_id']}/candidate.cif"
                    report.append(f"\n**{h['hypothesis_id']} — Structure:** {h['description']}\n\n"
                                  f"[CIF]({relative}) — actual formula {item['actual_formula']}; "
                                  f"actual space group {item['actual_space_group']}; "
                                  f"formula matches: {item['formula_matches']}; space group matches: {item['space_group_matches']}.")
        (folder / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        failed_crystals = sum(s.get("status") == "failed" for s in structures)
        status.update(status="partial_failure" if failures or failed_crystals else "complete", stage="complete", formula_count=len(formulas),
                      route_count=sum(len(d["routes"]) for d in documents), crystal_count=len(structures) - failed_crystals,
                      failed_crystal_count=failed_crystals, failed_formula_count=len(failures))
        save_json(folder / "status.json", status)
        print(f"Completed: {status['route_count']} routes, {status['crystal_count']} CIF candidates, {failed_crystals} failed crystals.\nResults: {folder}", flush=True)
        return folder
    except Exception as error:
        status.update(status="failed", error=str(error))
        save_json(folder / "status.json", status)
        print(f"Failed at {status['stage']}: {error}\nSaved results: {folder}", flush=True)
        raise
    finally:
        if generator is not None:
            generator.close()

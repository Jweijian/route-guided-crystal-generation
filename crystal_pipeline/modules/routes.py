import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from jsonschema import ValidationError, validate

from . import chemistry
from .chemistry import validate_target, validate_routes, validate_hypotheses
from .common import save_json
from .deepseek_api import complete_chat, create_session
from .retrieval import Retriever


def configure(config, schemas, prompts):
    global MAX_WORKERS, MAX_ROUTES, MAX_STRUCTURES_PER_ROUTE, PROMPT_VERSION
    global ROUTES_RESPONSE, HYPOTHESES_RESPONSE, SYSTEM_PROMPT, ROUTE_PROMPT, HYPOTHESIS_PROMPT
    global MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT, REPAIR_ATTEMPTS, HTTP_TRUST_ENV, HTTP_PROXY_URL, SETTINGS_SIGNATURE
    global REPAIR_PROMPT, FRESH_ALTERNATIVE_PROMPT
    MAX_WORKERS = config["max_workers"]
    MAX_ROUTES = config["max_routes"]
    MAX_STRUCTURES_PER_ROUTE = config["max_structures_per_route"]
    PROMPT_VERSION = config["prompt_version"]
    ROUTES_RESPONSE, HYPOTHESES_RESPONSE = copy.deepcopy(schemas["routes"]), copy.deepcopy(schemas["structures"])
    ROUTES_RESPONSE["properties"]["routes"]["maxItems"] = MAX_ROUTES
    HYPOTHESES_RESPONSE["properties"]["structure_hypotheses"]["maxItems"] = MAX_STRUCTURES_PER_ROUTE
    chemistry.configure(HYPOTHESES_RESPONSE)
    SYSTEM_PROMPT, ROUTE_PROMPT, HYPOTHESIS_PROMPT = prompts["system"], prompts["routes"], prompts["structures"]
    REPAIR_PROMPT, FRESH_ALTERNATIVE_PROMPT = prompts["repair"], prompts["fresh_alternative"]
    MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT = config["max_output_tokens"], config["request_timeout_seconds"]
    REPAIR_ATTEMPTS, HTTP_TRUST_ENV = config["repair_attempts"], config["http_trust_env"]
    HTTP_PROXY_URL = config.get("proxy_url", "")
    settings = {k: v for k, v in config.items() if k != "api_key"}
    SETTINGS_SIGNATURE = hashlib.sha256(json.dumps([settings, schemas, prompts], sort_keys=True).encode()).hexdigest()


class JsonLLM:
    def __init__(self, response_dir, api_key, model, base_url, thinking=True, reasoning_effort="low"):
        self.response_dir = Path(response_dir)
        self.api_key, self.model = api_key, model
        self.reasoning_effort = reasoning_effort
        self.base_url, self.thinking = base_url.rstrip("/"), thinking
        self.session = create_session({"http_trust_env": HTTP_TRUST_ENV, "proxy_url": HTTP_PROXY_URL})

    def call(self, label, instruction, payload, schema, semantic_check=lambda value: None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction + "\nJSON schema:\n" + json.dumps(schema, ensure_ascii=False)
                     + "\nInput data:\n" + json.dumps(payload, ensure_ascii=False)}]
        fingerprint = hashlib.sha256(json.dumps([self.model, self.thinking, self.reasoning_effort, MAX_OUTPUT_TOKENS, messages], ensure_ascii=False).encode()).hexdigest()
        for attempt in range(REPAIR_ATTEMPTS, 0, -1):
            path = self.response_dir / f"{label}_{attempt}.json"
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint and cached["finish_reason"] == "stop":
                    try:
                        value = json.loads(cached["raw_response"])
                        json.dumps(value, allow_nan=False)
                        validate(value, schema)
                        semantic_check(value)
                        return value
                    except (ValueError, TypeError, KeyError, ValidationError):
                        pass
        initial_messages = copy.deepcopy(messages)
        for attempt in range(1, REPAIR_ATTEMPTS + 1):
            started = time.perf_counter()
            data = complete_chat(self.session, self.api_key, self.model, self.base_url, messages,
                                 MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT, self.reasoning_effort)
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            save_json(self.response_dir / f"{label}_{attempt}.json", {
                "model": data.get("model", self.model), "thinking": self.thinking,
                "reasoning_effort": self.reasoning_effort, "fingerprint": fingerprint,
                "raw_response": content, "reasoning_content": choice["message"].get("reasoning_content"),
                "finish_reason": choice["finish_reason"], "usage": data.get("usage", {}),
                "elapsed_seconds": time.perf_counter() - started})
            try:
                if choice["finish_reason"] != "stop":
                    raise ValueError("Incomplete response: " + choice["finish_reason"])
                value = json.loads(content)
                json.dumps(value, allow_nan=False)
                validate(value, schema)
                semantic_check(value)
                return value
            except (ValueError, TypeError, KeyError, ValidationError) as error:
                save_json(self.response_dir / f"{label}_{attempt}_error.json", {"validation_error": str(error)})
                if attempt == REPAIR_ATTEMPTS:
                    raise ValueError(f"{label}: no valid output after {REPAIR_ATTEMPTS} attempts; inspect {self.response_dir}") from error
                if attempt == 2:
                    messages = copy.deepcopy(initial_messages)
                    messages.append({"role": "user", "content": FRESH_ALTERNATIVE_PROMPT.format(error=str(error))})
                    continue
                messages.extend([{"role": "assistant", "content": content},
                    {"role": "user", "content": REPAIR_PROMPT.format(error=str(error))}])


def compact_references(references):
    result = []
    for reference in references:
        entry = {k: v for k, v in reference.items() if k != "record"}
        entry["record"] = {k: reference["record"][k] for k in
            ("formula", "description", "synthesis_method", "precursors", "conditions", "evidence_snippet")
            if k in reference["record"]}
        result.append(entry)
    return result


def infer_routes(target, references, llm, output_dir, on_hypothesis=None):
    started = time.perf_counter()
    target = validate_target(target["formula"], target["oxidation_states"])
    result = {"schema_version": "3.0", "prompt_version": PROMPT_VERSION,
              "model": llm.model, "thinking": llm.thinking, "reasoning_effort": llm.reasoning_effort,
              "max_workers": MAX_WORKERS, "max_routes": MAX_ROUTES,
              "max_structures_per_route": MAX_STRUCTURES_PER_ROUTE, "settings_signature": SETTINGS_SIGNATURE, "target": target,
              "references": references, "route_assessments": {}, "routes": [], "status": "in_progress"}
    saved_path = Path(output_dir) / "routes.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        saved_references = [dict(reference, route_sentence=None) for reference in saved["references"]]
        current_references = [dict(reference, route_sentence=None) for reference in references]
        if saved["target"] != target or saved_references != current_references or saved["model"] != llm.model or saved["thinking"] != llm.thinking:
            raise ValueError("Saved batch does not match target, references, model, or mode")
        if saved.get("reasoning_effort", "high") != llm.reasoning_effort:
            raise ValueError("Saved batch uses a different reasoning effort; use a new batch directory")
        if (saved.get("max_routes", 6), saved.get("max_structures_per_route", 3)) != (MAX_ROUTES, MAX_STRUCTURES_PER_ROUTE):
            raise ValueError("Saved batch uses different route/structure limits; use a new batch directory")
        if saved.get("settings_signature") != SETTINGS_SIGNATURE:
            raise ValueError("Saved batch uses different configuration or prompts; use a new run directory")
        result["routes"] = saved["routes"]
        result["route_assessments"] = saved["route_assessments"]
        fields = ROUTES_RESPONSE["properties"]["routes"]["items"]["properties"]
        for route in result["routes"]:
            try:
                route_value = {"routes": [{k: route[k] for k in fields}], "assessment": "Resumed route"}
                validate(route_value, ROUTES_RESPONSE)
                validate_routes(route_value, references, route["route_basis"], target)
            except (ValueError, TypeError, KeyError, ValidationError) as error:
                raise ValueError(f"Saved route {route.get('route_id', '?')} is invalid: {error}. Use a new run directory") from error
    else:
        model_limit = MAX_ROUTES // 2 if references else MAX_ROUTES
        rag_limit = MAX_ROUTES - model_limit
        for basis, refs, limit in [("model_inference", [], model_limit),
                                   ("rag_inference", references, rag_limit)]:
            if limit == 0:
                continue
            route_schema = copy.deepcopy(ROUTES_RESPONSE)
            route_schema["properties"]["routes"]["maxItems"] = limit
            value = llm.call("routes_" + basis, ROUTE_PROMPT + f"\nReturn 1 to {limit} routes at most.",
                {"target": target, "route_basis": basis, "max_routes": limit, "references": compact_references(refs)},
                route_schema, lambda value, b=basis, rs=refs: validate_routes(value, rs, b, target))
            result["routes"].extend(value["routes"])
            result["route_assessments"][basis] = value["assessment"]
    fields = ROUTES_RESPONSE["properties"]["routes"]["items"]["properties"]
    validate({"routes": [{k: r[k] for k in fields} for r in result["routes"]],
              "assessment": "Combined routes"}, ROUTES_RESPONSE)
    pending = []
    for index, route in enumerate(result["routes"], 1):
        route["route_id"] = f"R{index:02d}"
        route_input = {k: route[k] for k in fields}
        candidate = None
        if "structure_hypotheses" in route and "assessment" in route:
            candidate = {"structure_hypotheses": route["structure_hypotheses"], "assessment": route["assessment"]}
            try:
                validate_hypotheses(candidate, route_input, target, references)
            except (ValueError, ValidationError):
                candidate = None
        if candidate is None:
            route.pop("structure_hypotheses", None)
            route.pop("assessment", None)
            pending.append(route)
        else:
            route.update(candidate)
            route.pop("hypothesis_error", None)
    save_json(saved_path, result)

    if on_hypothesis is not None:
        for route in result["routes"]:
            for hypothesis in route.get("structure_hypotheses", []):
                on_hypothesis(result, route, hypothesis)

    def infer_structure(route):
        # Sessions and response labels are independent; routes.json is written only by the main thread.
        worker = JsonLLM(llm.response_dir, llm.api_key, llm.model, llm.base_url,
                         thinking=llm.thinking, reasoning_effort=llm.reasoning_effort)
        route_input = {k: route[k] for k in fields}
        selected = [r for r in references if r["reference_id"] in route["reference_ids"]]
        try:
            return worker.call("hypotheses_" + route["route_id"],
                HYPOTHESIS_PROMPT + f"\nReturn 1 to {MAX_STRUCTURES_PER_ROUTE} structures at most.",
                {"target": target, "route": route_input, "max_structures": MAX_STRUCTURES_PER_ROUTE,
                 "references": compact_references(selected)},
                HYPOTHESES_RESPONSE,
                lambda value: validate_hypotheses(value, route_input, target, references))
        finally:
            worker.session.close()

    failures = []
    def finish_route(route, obtain):
        succeeded = False
        try:
            route.update(obtain())
            route.pop("hypothesis_error", None)
            succeeded = True
        except Exception as error:
            route["hypothesis_error"] = str(error)
            failures.append(f"{route['route_id']}: {error}")
        save_json(saved_path, result)
        if succeeded and on_hypothesis is not None:
            for hypothesis in route["structure_hypotheses"]:
                on_hypothesis(result, route, hypothesis)
        print(f"{route['route_id']}: {'complete' if succeeded else 'failed'}", flush=True)

    if on_hypothesis is not None:
        # Finish the current route's CIFs before requesting another description.
        for route in pending:
            finish_route(route, lambda: infer_structure(route))
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(infer_structure, route): route for route in pending}
            for future in as_completed(futures):
                finish_route(futures[future], future.result)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = "partial_failure" if failures else "hypotheses_ready"
    result["failures"] = failures
    save_json(saved_path, result)
    if failures:
        print("Some routes failed; continuing with completed candidates. " + " | ".join(failures), flush=True)
    return result



def load_descriptions(formulas, config, schemas, prompts, output_dir):
    """Read and validate completed descriptions without retrieval or API requests."""
    configure(config, schemas, prompts)
    documents = []
    fields = ROUTES_RESPONSE["properties"]["routes"]["items"]["properties"]
    for index, entry in enumerate(formulas, 1):
        formula_id = f"F{index:03d}"
        path = Path(output_dir) / formula_id / "routes.json"
        if not path.is_file():
            path = Path(output_dir) / formula_id / "failure.json"
        if not path.is_file():
            raise ValueError(f"Missing saved descriptions for {formula_id}; run the descriptions stage first")
        document = json.loads(path.read_text(encoding="utf-8"))
        target = validate_target(entry["formula"], entry["oxidation_states"])
        if document["status"] not in ("hypotheses_ready", "partial_failure", "failed") or document["target"] != target:
            raise ValueError(f"{formula_id}: saved descriptions are incomplete or do not match the target")
        if document.get("settings_signature") != SETTINGS_SIGNATURE:
            raise ValueError(f"{formula_id}: saved descriptions use different configuration, schemas or prompts")
        if document["status"] == "failed":
            document["formula_id"] = formula_id
            documents.append(document)
            continue
        route_ids = [route["route_id"] for route in document["routes"]]
        if len(set(route_ids)) != len(route_ids):
            raise ValueError(f"{formula_id}: duplicate saved route IDs")
        validate({"routes": [{k: route[k] for k in fields} for route in document["routes"]],
                  "assessment": "Saved routes"}, ROUTES_RESPONSE)
        for route in document["routes"]:
            route_input = {k: route[k] for k in fields}
            validate_routes({"routes": [route_input]}, document["references"], route["route_basis"], target)
            if document["status"] == "partial_failure" and "structure_hypotheses" not in route:
                continue
            candidate = {"structure_hypotheses": route["structure_hypotheses"], "assessment": route["assessment"]}
            validate_hypotheses(candidate, route_input, target, document["references"])
            route.update(candidate)
        document["formula_id"] = formula_id
        documents.append(document)
    return documents


def generate_routes(formulas, config, schemas, prompts, output_dir, on_hypothesis=None):
    configure(config, schemas, prompts)
    retriever = Retriever(config["database_file"])

    def infer_formula(index, entry):
        target = validate_target(entry["formula"], entry["oxidation_states"])
        formula_id = f"F{index:03d}"
        folder = Path(output_dir) / formula_id
        refs = retriever.retrieve(target["formula"], target["oxidation_states"],
                                  top_n=config["top_n"], limit=config["reference_limit"])
        save_json(folder / "references.json", refs)
        llm = JsonLLM(folder / "responses", config["api_key"], config["model"],
                      config["api_base_url"], thinking=True, reasoning_effort=config["reasoning_effort"])
        started = time.perf_counter()
        def consume(document, route, hypothesis):
            document["formula_id"] = formula_id
            try:
                on_hypothesis(document, route, hypothesis)
            except (ValueError, requests.RequestException) as error:
                raise RuntimeError(f"Crystal stage failed for {formula_id}: {error}") from error

        try:
            if on_hypothesis is None:
                document = infer_routes(target, refs, llm, folder)
            else:
                document = infer_routes(target, refs, llm, folder, on_hypothesis=consume)
        except (ValueError, requests.RequestException) as error:
            document = {"target": target, "references": refs, "routes": [], "status": "failed",
                        "elapsed_seconds": time.perf_counter() - started, "failures": [str(error)],
                        "settings_signature": SETTINGS_SIGNATURE}
            save_json(folder / "failure.json", document)
            print(f"{formula_id}: failed; continuing. {error}", flush=True)
        finally:
            llm.session.close()
        document["formula_id"] = formula_id
        if document["status"] != "failed":
            (folder / "failure.json").unlink(missing_ok=True)
        return document

    # CIF callbacks share one generator, so only the description-only path runs formulas concurrently.
    workers = config.get("formula_workers", 1)
    documents = []
    if on_hypothesis is not None or workers == 1:
        for index, entry in enumerate(formulas, 1):
            documents.append(infer_formula(index, entry))
            save_json(Path(output_dir) / "all_results.json", documents)
    else:
        documents = [None] * len(formulas)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(infer_formula, index, entry): index - 1
                       for index, entry in enumerate(formulas, 1)}
            for future in as_completed(futures):
                documents[futures[future]] = future.result()
                # One writer maintains stable formula order even when requests finish out of order.
                save_json(Path(output_dir) / "all_results.json", [d for d in documents if d is not None])
    return documents

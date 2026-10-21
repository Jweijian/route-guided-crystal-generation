import copy
import json
import re
from functools import lru_cache

import numpy as np
from jsonschema import ValidationError, validate
from pymatgen.core import Composition, Element
from pymatgen.symmetry.groups import SpaceGroup


def configure(structure_schema):
    global HYPOTHESES_RESPONSE
    HYPOTHESES_RESPONSE = structure_schema




def validate_target(formula, oxidation_states):
    """检查元素价态覆盖与形式电荷平衡，返回标准化化学式。"""
    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()
    if not amounts or not all(a > 0 and np.isfinite(a) for a in amounts.values()):
        raise ValueError("Formula must have positive finite element amounts")
    if set(amounts) != set(oxidation_states):
        raise ValueError("oxidation_states must cover exactly the formula elements")
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
        for v in oxidation_states.values()
    ):
        raise ValueError("Oxidation states must be finite numbers")
    if abs(sum(amounts[e] * oxidation_states[e] for e in amounts)) > 1e-8:
        raise ValueError("Formula and oxidation states are not charge-balanced")
    return {"formula": comp.reduced_formula, "oxidation_states": oxidation_states}


def composition_key(formula):
    return tuple(sorted(Composition(formula).reduced_composition.get_el_amt_dict().items()))


@lru_cache(maxsize=1)
def space_group_aliases():
    aliases = {
        SpaceGroup.from_int_number(i).symbol.replace("_", "").replace(" ", ""): i
        for i in range(1, 231)
    }
    for symbol, data in SpaceGroup.sg_encoding.items():
        aliases[symbol.replace("_", "").replace(" ", "")] = data["int_number"]
        aliases[data["full_symbol"].replace("_", "").replace(" ", "")] = data["int_number"]
    for abbreviated, full in SpaceGroup.abbrev_sg_mapping.items():
        aliases[abbreviated.replace("_", "")] = aliases[full.replace("_", "")]
    return aliases


def space_group_numbers(text):
    aliases = space_group_aliases()
    symbols = re.findall(r"(?<![A-Za-z0-9])([PABCIFR][0-9a-z_/-]+)(?![A-Za-z0-9])", text)
    return list(
        dict.fromkeys(aliases[s.replace("_", "")] for s in symbols if s.replace("_", "") in aliases)
    )


def prepare_document(document):
    """Preserve legacy target symmetry hints explicitly as unverified exploration."""
    document = copy.deepcopy(document)
    names = {
        "三斜": "triclinic",
        "单斜": "monoclinic",
        "正交": "orthorhombic",
        "四方": "tetragonal",
        "三方": "trigonal",
        "菱方": "trigonal",
        "六方": "hexagonal",
        "立方": "cubic",
    }
    for route in document["routes"]:
        for h in route.get("structure_hypotheses", []):
            # Only the opening target statement, never the later reference discussion.
            opening = re.split(r"[。\n]|\.\s", h["description"], maxsplit=1)[0]
            opening = re.split(r"参考|类比|reference|analog", opening, maxsplit=1, flags=re.I)[0]
            hinted_groups = space_group_numbers(opening)
            groups = h["space_group_candidates"]
            if not groups and hinted_groups:
                groups.extend(
                    {
                        "number": g,
                        "reason": "从目标结构描述提取用于探索；未核实该晶相或路线可合成性。",
                        "reference_ids": [],
                        "basis": "exploratory",
                    }
                    for g in hinted_groups
                )
            elif hinted_groups and not set(hinted_groups) <= {g["number"] for g in groups}:
                raise ValueError(
                    f"{route['route_id']}/{h['hypothesis_id']}: description and space_group_candidates disagree"
                )
            for g in groups:
                g.setdefault("basis", "analogy_inference")
            # Cell axes/settings do not define the crystal system.
            system_text = re.sub(
                r"(?:六方|菱方|三方)\s*(?:晶胞|坐标轴|轴设置)|\b(?:hexagonal|rhombohedral|trigonal)\s+(?:unit\s+)?(?:cell|axes|setting)\b",
                "",
                opening,
                flags=re.I,
            )
            systems = {v for k, v in names.items() if k in system_text}
            systems.update(
                v for v in names.values() if re.search(r"\b" + v + r"\b", system_text, re.I)
            )
            system_terms = "|".join([*names, *names.values()])
            alternatives = {
                names.get(term.lower(), term.lower())
                for pair in re.findall(
                    rf"({system_terms})\s*(?:或者|或|/|\bor\b)\s*({system_terms})",
                    system_text,
                    re.I,
                )
                for term in pair
            }
            group_systems = {SpaceGroup.from_int_number(g["number"]).crystal_system for g in groups}
            h.setdefault(
                "crystal_system",
                (
                    next(iter(systems))
                    if len(systems) == 1
                    else next(iter(group_systems)) if len(group_systems) == 1 else None
                ),
            )
            if h["crystal_system"] is None and len(systems) == 1:
                h["crystal_system"] = next(iter(systems))
            description_conflict = systems - {h["crystal_system"]} and not (
                h["crystal_system"] in alternatives and systems <= alternatives
            )
            if h["crystal_system"] and (
                group_systems - {h["crystal_system"]} or description_conflict
            ):
                classifications = {
                    g["number"]: SpaceGroup.from_int_number(g["number"]).crystal_system
                    for g in groups
                }
                raise ValueError(
                    f"{route['route_id']}/{h['hypothesis_id']}: crystal_system={h['crystal_system']} conflicts with "
                    f"space-group classification {classifications} or target opening {sorted(systems)}. "
                    "Correct crystal_system and description using these classifications; a trigonal structure may use hexagonal cell axes."
                )
    return document


def check_references(ids, allowed):
    if not set(ids) <= set(allowed):
        raise ValueError(f"Unknown reference IDs: {set(ids) - set(allowed)}")


def validate_routes(value, references, expected_basis, target=None):
    """分别检查自主推断与 RAG 推断的来源标签和引用。"""
    allowed = {r["reference_id"] for r in references}
    ids = [r["route_id"] for r in value["routes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate route_id")
    for route in value["routes"]:
        if route["route_basis"] != expected_basis:
            raise ValueError("route_basis must match the requested inference source")
        if expected_basis == "model_inference" and route["reference_ids"]:
            raise ValueError("Model-only routes must not claim RAG references")
        if expected_basis == "rag_inference" and not route["reference_ids"]:
            raise ValueError("RAG routes must cite at least one retrieved reference")
        check_references(route["reference_ids"], allowed)
        check_reaction(route, target)
        for step in route["steps"]:
            check_references(step["reference_ids"], route["reference_ids"])
            if step["evidence_type"] == "reported" and not step["reference_ids"]:
                raise ValueError("Reported step requires a reference")


def standardize_description(h, target):
    group = h["space_group_candidates"][0]
    symmetry = SpaceGroup.from_int_number(group["number"])
    if symmetry.crystal_system != h["crystal_system"]:
        raise ValueError("Crystal system must match the selected space-group number")
    for field in ("structure_type", "structural_details"):
        if re.search(r"[\u4e00-\u9fff]", h[field]):
            raise ValueError("Structure type and details must be written in English")
    symbol = symmetry.symbol.replace("_", "")
    h["description"] = (
        f"{target['formula']} is {h['structure_type']} structured and crystallizes in the "
        f"{h['crystal_system']} {symbol} space group (No. {group['number']}). "
        + h["structural_details"].strip()
    )


def _validate_hypothesis_batch(value, route, target, references=None):
    """校验最终目标组成、引用、元素配位及候选空间群的一致性。"""
    validate(value, HYPOTHESES_RESPONSE)
    for hypothesis in value["structure_hypotheses"]:
        standardize_description(hypothesis, target)
    value["structure_hypotheses"] = prepare_document(
        {"routes": [{**route, "structure_hypotheses": value["structure_hypotheses"]}]}
    )["routes"][0]["structure_hypotheses"]
    validate(value, HYPOTHESES_RESPONSE)
    ids = [h["hypothesis_id"] for h in value["structure_hypotheses"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate hypothesis_id within route")
    for h in value["structure_hypotheses"]:
        # 旧响应按是否填写引用标注来源，不为模型补造证据。
        h.setdefault("hypothesis_basis", "rag_inference" if h["evidence"] else "model_inference")
        if h["hypothesis_basis"] == "rag_inference" and not h["evidence"]:
            raise ValueError("RAG-based structure hypothesis must cite route references; use model_inference for exploration without retrieved evidence")
        if h["hypothesis_basis"] == "model_inference" and h["evidence"]:
            raise ValueError("Model-only structure hypothesis must have empty evidence; use rag_inference when citing retrieved support")
        if route["route_basis"] == "model_inference" and h["hypothesis_basis"] != "model_inference":
            raise ValueError("Model-only route has no retrieved evidence for a RAG structure hypothesis")
        if h["hypothesis_basis"] == "model_inference" and any(
            g["basis"] != "exploratory" or g["reference_ids"] for g in h["space_group_candidates"]
        ):
            raise ValueError("Model-only hypothesis space groups must be exploratory without reference IDs")
        comp = Composition(h["product_formula"])
        elements = set(comp.get_el_amt_dict())
        if not elements or not all(a > 0 for a in comp.get_el_amt_dict().values()):
            raise ValueError("Invalid product_formula")
        if composition_key(comp) != composition_key(target["formula"]):
            raise ValueError(
                "product_formula must match target formula composition and stoichiometry; "
                "intermediates and secondary phases belong only in assessment or uncertainties"
            )
        h["product_formula"] = target["formula"]
        for entry in h["coordination"]:
            if entry["center"] not in elements or entry["neighbor"] not in elements:
                raise ValueError("Coordination elements not in product_formula")
        groups = [g["number"] for g in h["space_group_candidates"]]
        if len(groups) != len(set(groups)):
            raise ValueError("Duplicate candidate space groups")
        for g in h["space_group_candidates"]:
            check_references(g["reference_ids"], route["reference_ids"])
            if g.get("basis") != "exploratory" and not g["reference_ids"]:
                raise ValueError(
                    "Evidence-based space group requires references; use exploratory for unverified proposals"
                )
            if g["basis"] == "database_reference" and references is not None:
                records = {r["reference_id"]: r for r in references}
                if not any(
                    composition_key(records[r]["formula"]) == composition_key(h["product_formula"])
                    and g["number"]
                    in space_group_numbers(
                        records[r]["record"].get("description", "").split(".")[0]
                    )
                    for r in g["reference_ids"]
                ):
                    raise ValueError(
                        "Database space group must cite the same product composition and stated space group; use analogy_inference or exploratory otherwise"
                    )
        check_references([e["reference_id"] for e in h["evidence"]], route["reference_ids"])
        check_chemical_consistency(h, target, references or [])

def check_chemical_consistency(h, target, references):
    prototype = h["prototype"]
    if prototype is not None:
        mapping = prototype["site_mapping"]
        amounts = Composition(target["formula"]).get_el_amt_dict()
        if set(mapping) != set(amounts):
            raise ValueError("Prototype mapping must cover all target elements")
        mapped = {}
        for element, amount in amounts.items():
            mapped[mapping[element]] = mapped.get(mapping[element], 0) + amount
        if composition_key(Composition(mapped)) != composition_key(prototype["formula"]):
            raise ValueError("Prototype stoichiometry does not match mapped target; remove unsupported prototype")
    elif re.search(r"[A-Z][A-Za-z0-9()]*型|钙钛矿|反?萤石|[A-Z][a-z]?\d*[A-Z][A-Za-z0-9()]*-type|rock[ -]salt|halite|perovskite|fluorite|zinc[ -]blende|wurtzite", h["structure_type"], re.I):
        raise ValueError(
            f"{h['structure_type']}: a named prototype requires prototype.formula and a stoichiometrically valid site_mapping. "
            "Replace this candidate with a compatible explicit topology; simply adding hypothetical does not fix a mismatched prototype."
        )
    for entry in h["coordination"]:
        geometry = entry["geometry"]
        for pattern, number in [(r"立方八面体|cuboctahed", 12),
                                (r"四面体|tetrahed", 4), (r"八面体|(?<![A-Za-z])octahed", 6)]:
            if re.search(pattern, geometry, re.I) and entry["number"] != number:
                raise ValueError(
                    f"{entry['center']}-{entry['neighbor']}: {geometry} requires {number} neighbors, "
                    f"not {entry['number']}. Geometry refers to this neighbor subset."
                )
            if re.search(pattern, geometry, re.I):
                break
    amounts = Composition(target["formula"]).get_el_amt_dict()
    pairs = {}
    for entry in h["coordination"]:
        pairs.setdefault((entry["center"], entry["neighbor"]), []).append(entry["number"])
    for (center, neighbor), counts in pairs.items():
        reverse = pairs.get((neighbor, center), [])
        if len(counts) == len(reverse) == 1:
            left, right = amounts[center] * counts[0], amounts[neighbor] * reverse[0]
            if abs(left - right) > 1e-8:
                raise ValueError(
                    f"Reciprocal coordination mismatch: {center}-{neighbor} gives {left:g} contacts per formula, "
                    f"but {neighbor}-{center} gives {right:g}. Use stoichiometrically consistent counts."
                )
    text = json.dumps(h, ensure_ascii=False).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-"))
    for element, number, sign in re.findall(r"([A-Z][a-z]?)([1-9][0-9]*)([+-])(?![A-Za-z0-9])", text):
        if element in target["oxidation_states"]:
            charge = int(number) * (1 if sign == "+" else -1)
            if charge != target["oxidation_states"][element]:
                raise ValueError(f"{element}: stated charge conflicts with input oxidation state")
    records = {r["reference_id"]: r for r in references}
    for evidence in h["evidence"]:
        if evidence["basis"] == "database_reference" and composition_key(records[evidence["reference_id"]]["formula"]) != composition_key(target["formula"]):
            raise ValueError("Different compositions are analogy_inference, not direct target evidence")


def reaction_species_key(formula):
    """Preserve molecular counts; reduced composition cannot identify a reagent."""
    amounts = Composition(formula, strict=True).get_el_amt_dict()
    if not amounts or not all(a > 0 and np.isfinite(a) for a in amounts.values()):
        raise ValueError(f"Reaction species must have positive finite element amounts: {formula}")
    for element in amounts:
        Element(element)
    return tuple(sorted(amounts.items()))


def check_reaction(route, target):
    reaction = route["balanced_reaction"]
    if reaction is None:
        if re.search(r"\d+(?:\.\d+)?\s*[:：]\s*\d", route["route_summary"]):
            raise ValueError("Explicit precursor ratio requires a balanced reaction")
        return
    precursor_keys = {reaction_species_key(formula) for formula in route["precursors"]}
    reactant_keys = {reaction_species_key(species["formula"]) for species in reaction["reactants"]}
    if precursor_keys != reactant_keys:
        raise ValueError(
            "Precursor species must match all reactants, including consumed gases. "
            f"Missing precursors={reactant_keys - precursor_keys}; extra precursors={precursor_keys - reactant_keys}. "
            "Put unconsumed solvents, inert atmospheres and additives in step notes instead."
        )
    budgets = []
    net_target = 0.0
    target_key = composition_key(target["formula"]) if target else None
    target_amounts = Composition(target["formula"]).get_el_amt_dict() if target else {}
    for side in ("reactants", "products"):
        budget = {}
        for species in reaction[side]:
            coefficient = species["amount"]
            if (not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool)
                    or not np.isfinite(coefficient) or coefficient <= 0):
                raise ValueError("Reaction coefficients must be positive finite numbers")
            amounts = dict(reaction_species_key(species["formula"]))
            for element, amount in amounts.items():
                budget[element] = budget.get(element, 0) + amount * coefficient
            if target and composition_key(Composition(amounts)) == target_key:
                element = next(iter(target_amounts))
                units = coefficient * amounts[element] / target_amounts[element]
                net_target += units if side == "products" else -units
        if not budget or not all(np.isfinite(amount) for amount in budget.values()):
            raise ValueError("Reaction element totals must be nonempty and finite")
        budgets.append(budget)
    if any(abs(budgets[0].get(e, 0) - budgets[1].get(e, 0)) > 1e-8 for e in set(budgets[0]) | set(budgets[1])):
        raise ValueError(
            f"Reaction mass balance failed: reactant totals={budgets[0]}, product totals={budgets[1]}. "
            "Balance every element including byproducts, or set balanced_reaction=null and remove definite precursor ratios. Keep an exploratory route."
        )
    if target and (not np.isfinite(net_target) or net_target <= 0):
        raise ValueError("Balanced reaction must have positive net production of the target; identity reactions are invalid")


def validate_hypotheses(value, route, target, references=None):
    """Retain individually consistent hypotheses; request repair if none survive."""
    validate(value, HYPOTHESES_RESPONSE)
    ids = [h["hypothesis_id"] for h in value["structure_hypotheses"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate hypothesis_id within route")
    accepted, rejected = [], []
    for h in value["structure_hypotheses"]:
        candidate = {"structure_hypotheses": [copy.deepcopy(h)], "assessment": value["assessment"]}
        try:
            _validate_hypothesis_batch(candidate, route, target, references)
            accepted.extend(candidate["structure_hypotheses"])
        except (ValueError, ValidationError) as error:
            rejected.append(f"{h['hypothesis_id']}: {str(error).splitlines()[0]}")
    if not accepted:
        raise ValueError("No consistent candidate remains; propose at least one alternative. " + "; ".join(rejected))
    value["structure_hypotheses"] = accepted
    if rejected:
        value["assessment"] += " Excluded inconsistent candidates: " + "; ".join(rejected)

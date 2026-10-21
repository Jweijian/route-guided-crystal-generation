"""Render the synthesis fields of a database record as one English sentence."""

import re


def condition_text(value):
    text = {str: str, list: lambda items: "; ".join(condition_text(item) for item in items),
            type(None): lambda _: "not specified"}[type(value)](value)
    text = re.sub(r"\s+", " ", text).strip()
    text = {"null": "not specified", "none": "not specified", "n/a": "not specified"}.get(text.casefold(), text)
    # Separate clauses without altering decimal points or chemical formulae.
    return re.sub(r"(?<!\d)[.!?](?=\s|$)|[。！？]", ";", text).rstrip(" ;") or "not specified"


def route_to_sentence(record):
    """Preserve reported alternatives and missing values without inferring a recipe."""
    conditions = record["conditions"]
    return (
        f"The database record for {record['formula']} describes a synthesis route "
        f"by {condition_text(record['synthesis_method'])} "
        f"using precursors listed as {condition_text(record['precursors'])}, "
        f"with temperature recorded as {condition_text(conditions['temperature'])}, "
        f"duration as {condition_text(conditions['time'])}, "
        f"atmosphere as {condition_text(conditions['atmosphere'])}, "
        f"additional processing conditions as {condition_text(conditions['other'])}, "
        f"and equipment listed as {condition_text(record['equipment'])}."
    )

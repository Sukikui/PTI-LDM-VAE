from __future__ import annotations

import json
import os
from typing import Any


def _load_attribute_json(attribute_file: str) -> dict[str, dict[str, float]]:
    """Load attribute JSON mapping filenames to attribute dictionaries.

    Args:
        attribute_file (str): Path to a JSON file.

    Returns:
        dict[str, dict[str, float]]: Mapping filename -> {attribute_name: value}.
    """
    if not os.path.exists(attribute_file):
        raise FileNotFoundError(f"Attribute file not found: {attribute_file}")

    with open(attribute_file, encoding="utf-8") as file:
        return json.load(file)


def select_attribute_sources(
    attribute_file: str | dict[str, str],
    data_source: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load attribute mappings depending on data source.

    Args:
        attribute_file (str | dict[str, str]): JSON path or source-to-path mapping.
        data_source (str): Data source key (edente/dente/both).

    Returns:
        dict[str, dict[str, dict[str, float]]]: Mapping from source -> attribute dict.

    Notes:
        When ``data_source`` is ``both`` and ``attribute_file`` is a single path, the same mapping is reused for
        both ``edente`` and ``dente`` entries.
    """
    if isinstance(attribute_file, str):
        mapping = _load_attribute_json(attribute_file)
        if data_source == "both":
            return {"edente": mapping, "dente": mapping}
        return {data_source: mapping}

    if isinstance(attribute_file, dict):
        return {source: _load_attribute_json(path) for source, path in attribute_file.items()}

    raise ValueError("attribute_file must be a string or mapping from source to file.")


def _normalize_attributes(
    attributes: dict[str, float],
    normalize_cfg: dict[str, Any] | None,
) -> dict[str, float]:
    """Normalize attribute values when requested.

    Args:
        attributes (dict[str, float]): Attribute values for one sample.
        normalize_cfg (dict[str, Any] | None): Normalization config with ``enabled`` and ``divisor``.

    Returns:
        dict[str, float]: Normalized attributes.
    """
    if not normalize_cfg or not normalize_cfg.get("enabled", False):
        return attributes

    divisor = float(normalize_cfg.get("divisor", 1.0))
    if divisor == 0:
        raise ValueError("Normalization divisor must be non-zero.")

    return {key: float(value) / divisor for key, value in attributes.items()}


def filter_attributes_for_paths(
    paths: list[str],
    attribute_sources: dict[str, dict[str, dict[str, float]]],
    attribute_keys: list[str],
    normalize_cfg: dict[str, Any] | None,
) -> list[dict[str, float]]:
    """Extract and normalize attributes for a list of image paths.

    Args:
        paths (list[str]): Image file paths.
        attribute_sources (dict[str, dict[str, dict[str, float]]]): Source -> attribute mapping.
        attribute_keys (list[str]): Attribute names to keep.
        normalize_cfg (dict[str, Any] | None): Optional normalization configuration.

    Returns:
        list[dict[str, float]]: Attributes aligned with ``paths`` order.
    """
    attributes: list[dict[str, float]] = []
    for path in paths:
        base = os.path.basename(path)
        if "edente" in path:
            source_key = "edente"
        elif "dente" in path:
            source_key = "dente"
        else:
            raise ValueError(f"Cannot identify data source from path: {path}")

        mapping = attribute_sources.get(source_key, {})
        attribute_dict = mapping.get(base)
        if attribute_dict is None:
            raise FileNotFoundError(f"Attribute entry missing for {base} in source {source_key}")

        filtered = {key: float(attribute_dict[key]) for key in attribute_keys if key in attribute_dict}
        if len(filtered) != len(attribute_keys):
            missing = set(attribute_keys).difference(filtered)
            raise KeyError(f"Missing attributes for {base}: {missing}")

        filtered = _normalize_attributes(filtered, normalize_cfg)
        attributes.append(filtered)
    return attributes

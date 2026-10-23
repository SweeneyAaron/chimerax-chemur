"""Serialize Chemur interaction rows to JSON or CSV.

Operates on the plugin's own ``InteractionRow``/``Endpoint`` objects (see
runner.py), i.e. exactly what the results tree shows after atom mapping and the
UI filters. This is deliberately separate from the engine's
``AnalysisResult.to_json()/to_csv()`` (in src/chemur/models.py), which serialize
the pre-mapping engine model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def resolve_output_path(path, selected_filter, *, allowed=("json", "csv")):
    """Resolve an explicit format choice, refusing contradictory suffixes."""
    choices = [name for name in allowed if name in selected_filter.lower()]
    if len(choices) != 1:
        raise ValueError("Choose one export format: %s" % ", ".join(allowed))
    format_name = choices[0]
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix("." + format_name)
    elif target.suffix.lower() != "." + format_name:
        raise ValueError("The filename extension must match the selected %s format." % format_name.upper())
    return target, format_name


def write_native_json(result, path):
    """Export the complete assigned engine result, independent of display filters."""
    kwargs = {"include_raw": False} if hasattr(result, "raw_candidates") else {}
    result.to_json(path, **kwargs)


def write_native_csv(result, path):
    """Export the complete assigned engine result, independent of display filters."""
    kwargs = {"include_raw": False} if hasattr(result, "raw_candidates") else {}
    result.to_csv(path, **kwargs)


def _endpoint_key(endpoint):
    atoms = (getattr(endpoint, "center_atoms", None)
             or getattr(endpoint, "ring_atoms", None) or [endpoint.atom])
    return frozenset(id(atom) for atom in atoms)


def _endpoint_dict(ep):
    """One partner of an interaction as a flat dict."""
    return {
        "molecule_type": ep.molecule_type,
        "chain_id": ep.chain_id,
        "residue_name": ep.residue_name,
        "residue_id": ep.residue_id,
        "atom_name": ep.atom.name,
        "atom_label": ep.atom_label,
    }


def rows_to_records(rows):
    """Turn ``InteractionRow`` objects into JSON-ready dicts (one per row)."""
    records = []
    by_pair = {}
    for row in rows:
        key = (row.interaction_type, frozenset((_endpoint_key(row.endpoint1),
                                               _endpoint_key(row.endpoint2))))
        metadata = dict(row.extra)
        if key in by_pair:
            # Overlapping enclosures may share a displayed edge. Keep the
            # provenance of every group while exporting the physical pair once.
            existing = by_pair[key]
            groups = list(existing["metadata"].get("enclosure_groups", []))
            for group in metadata.get("enclosure_groups", []):
                if group not in groups:
                    groups.append(group)
            if groups:
                existing["metadata"]["enclosure_groups"] = groups
                existing["enclosure_groups"] = groups
            continue
        record = {
            "interaction_type": row.interaction_type,
            "distance": row.distance,
            "angle": row.angle,
            "offset": row.offset,
            "partner1": _endpoint_dict(row.endpoint1),
            "partner2": _endpoint_dict(row.endpoint2),
            "geometry": dict(row.extra),
            "metadata": metadata,
            "enclosure_groups": list(metadata.get("enclosure_groups", [])),
        }
        records.append(record)
        by_pair[key] = record
    return records


def write_json(rows, path):
    """Write ``rows`` to ``path`` as a JSON array of interaction records."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows_to_records(rows), handle, indent=2)
        handle.write("\n")


# Fixed columns; geometry keys (which vary by interaction type) are appended.
_BASE_FIELDS = ["interaction_type", "distance", "angle", "offset", "metadata", "enclosure_groups"]
_PARTNER_FIELDS = ["molecule_type", "chain_id", "residue_name", "residue_id",
                   "atom_name", "atom_label"]


def write_csv(rows, path):
    """Write ``rows`` to ``path`` as CSV, flattening both partners into columns."""
    records = rows_to_records(rows)

    partner_cols = []
    for partner in ("partner1", "partner2"):
        partner_cols.extend("%s_%s" % (partner, field) for field in _PARTNER_FIELDS)
    # Union of geometry keys across all rows, kept in first-seen order.
    geometry_keys = []
    for record in records:
        for key, value in record["geometry"].items():
            if key not in geometry_keys and key not in _BASE_FIELDS and not isinstance(value, (dict, list, tuple)):
                geometry_keys.append(key)

    fieldnames = _BASE_FIELDS + partner_cols + geometry_keys

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            flat = {field: record[field] for field in _BASE_FIELDS}
            flat["metadata"] = json.dumps(record["metadata"], sort_keys=True)
            flat["enclosure_groups"] = json.dumps(record["enclosure_groups"], sort_keys=True)
            for partner in ("partner1", "partner2"):
                for field in _PARTNER_FIELDS:
                    flat["%s_%s" % (partner, field)] = record[partner][field]
            for key in geometry_keys:
                flat[key] = record["geometry"].get(key)
            writer.writerow(flat)

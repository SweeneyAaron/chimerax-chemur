"""Serialize ChemeleonX interaction rows to JSON or CSV.

Operates on the plugin's own ``InteractionRow``/``Endpoint`` objects (see
runner.py), i.e. exactly what the results tree shows after atom mapping and the
UI filters. This is deliberately separate from the engine's
``AnalysisResult.to_json()/to_csv()`` (in src/chemeleonx/models.py), which serialize
the pre-mapping engine model.
"""

from __future__ import annotations

import csv
import json


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
    for row in rows:
        records.append({
            "interaction_type": row.interaction_type,
            "distance": row.distance,
            "angle": row.angle,
            "offset": row.offset,
            "partner1": _endpoint_dict(row.endpoint1),
            "partner2": _endpoint_dict(row.endpoint2),
            "geometry": dict(row.extra),
        })
    return records


def write_json(rows, path):
    """Write ``rows`` to ``path`` as a JSON array of interaction records."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows_to_records(rows), handle, indent=2)
        handle.write("\n")


# Fixed columns; geometry keys (which vary by interaction type) are appended.
_BASE_FIELDS = ["interaction_type", "distance", "angle", "offset"]
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
        for key in record["geometry"]:
            if key not in geometry_keys:
                geometry_keys.append(key)

    fieldnames = _BASE_FIELDS + partner_cols + geometry_keys

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            flat = {field: record[field] for field in _BASE_FIELDS}
            for partner in ("partner1", "partner2"):
                for field in _PARTNER_FIELDS:
                    flat["%s_%s" % (partner, field)] = record[partner][field]
            for key in geometry_keys:
                flat[key] = record["geometry"].get(key)
            writer.writerow(flat)

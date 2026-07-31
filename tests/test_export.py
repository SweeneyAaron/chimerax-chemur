"""Tests for the ChimeraX plugin's interaction export (src/export.py).

The module only depends on the stdlib and duck-typed row objects, so we load it
directly from its file path (the bundle is not importable outside ChimeraX).
"""

import csv
import importlib.util
import json
from pathlib import Path

_EXPORT_PATH = (Path(__file__).resolve().parent.parent
                / "src" / "export.py")
_spec = importlib.util.spec_from_file_location("chemur_chimerax_export", _EXPORT_PATH)
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)


class _Atom:
    def __init__(self, name):
        self.name = name


class _Endpoint:
    def __init__(self, molecule_type, chain_id, residue_name, residue_id, atom_name):
        self.atom = _Atom(atom_name)
        self.molecule_type = molecule_type
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.residue_name = residue_name

    @property
    def atom_label(self):
        return "/%s %s %s %s" % (self.chain_id, self.residue_name,
                                 self.residue_id, self.atom.name)


class _Row:
    def __init__(self, interaction_type, ep1, ep2, distance, angle, offset, extra):
        self.interaction_type = interaction_type
        self.endpoint1 = ep1
        self.endpoint2 = ep2
        self.distance = distance
        self.angle = angle
        self.offset = offset
        self.extra = extra


def _rows():
    return [
        _Row("hbond",
             _Endpoint("ligand", "A", "LIG", "1", "O1"),
             _Endpoint("protein", "A", "ASP", "42", "OD1"),
             2.9, 160.0, None, {"donor_angle": 155.0}),
        _Row("pipi_stack",
             _Endpoint("ligand", "A", "LIG", "1", "C3"),
             _Endpoint("protein", "A", "PHE", "99", "CG"),
             3.8, None, 1.2, {"geometry": "parallel"}),
    ]


def test_rows_to_records():
    records = export.rows_to_records(_rows())
    assert len(records) == 2
    first = records[0]
    assert first["interaction_type"] == "hbond"
    assert first["distance"] == 2.9
    assert first["offset"] is None
    assert first["partner1"]["atom_label"] == "/A LIG 1 O1"
    assert first["partner2"]["residue_name"] == "ASP"
    assert first["geometry"] == {"donor_angle": 155.0}


def test_write_json(tmp_path):
    path = tmp_path / "interactions.json"
    export.write_json(_rows(), path)
    data = json.loads(path.read_text())
    assert [r["interaction_type"] for r in data] == ["hbond", "pipi_stack"]
    assert data[1]["partner2"]["atom_name"] == "CG"


def test_write_csv(tmp_path):
    path = tmp_path / "interactions.csv"
    export.write_csv(_rows(), path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    # Geometry keys from both rows become their own columns (union).
    assert "donor_angle" in rows[0]
    assert "geometry" in rows[0]
    assert rows[0]["partner1_atom_label"] == "/A LIG 1 O1"
    assert rows[0]["donor_angle"] == "155.0"
    assert rows[0]["geometry"] == ""  # absent on the hbond row
    assert rows[1]["geometry"] == "parallel"
    assert rows[1]["offset"] == "1.2"

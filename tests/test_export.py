"""Tests for the ChimeraX plugin's interaction export (src/export.py).

The module only depends on the stdlib and duck-typed row objects, so we load it
directly from its file path (the bundle is not importable outside ChimeraX).
"""

import csv
import importlib.util
import json
from pathlib import Path
import pytest

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


def test_grouped_display_metadata_roundtrips_and_shared_legs_are_unique(tmp_path):
    first = _rows()[0]
    first.interaction_type = "hydrophobic_enclosure"
    first.extra = {"center_atom_id": 10, "legs": [{"atom_ids": [10, 20], "distance": 3.5}],
                   "enclosure_groups": [{"center_atom_id": 10}]}
    duplicate = _Row(first.interaction_type, first.endpoint2, first.endpoint1,
                     3.5, None, None, {"enclosure_groups": [{"center_atom_id": 20}]})
    records = export.rows_to_records([first, duplicate])
    assert len(records) == 1
    assert records[0]["enclosure_groups"] == [{"center_atom_id": 10}, {"center_atom_id": 20}]
    assert first.extra["enclosure_groups"] == [{"center_atom_id": 10}]
    path = tmp_path / "contacts.csv"
    export.write_csv([first, duplicate], path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert json.loads(rows[0]["metadata"]) == records[0]["metadata"]
    assert json.loads(rows[0]["enclosure_groups"]) == records[0]["enclosure_groups"]
    assert "legs" not in rows[0]  # Nested data lives in the JSON metadata column.


def test_native_reports_preserve_group_and_omit_raw_candidates(tmp_path):
    from chemur.models import AnalysisResult, AssignedInteraction, CandidateInteraction
    group = AssignedInteraction("hydrophobic_enclosure", (), (0, 1, 2, 3, 4), (),
                                3.5, 135, "inter", 1,
                                metadata={"center_atom_id": 0, "neighbor_atom_ids": [1, 2, 3, 4]})
    raw = CandidateInteraction("hydrophobic", (), (0, 1), (), 3.0, 140, "inter")
    result = AnalysisResult([], [], [], [group], raw_candidates=[raw])
    export.write_native_json(result, tmp_path / "native.json")
    payload = json.loads((tmp_path / "native.json").read_text())
    assert "raw_candidates" not in payload
    assert payload["interactions"][0]["atom_ids"] == [0, 1, 2, 3, 4]
    export.write_native_csv(result, tmp_path / "native.csv")
    with (tmp_path / "native.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["assigned"] == "True"
    assert json.loads(rows[0]["metadata"]) == group.metadata
    assert result.raw_candidates == [raw]


def test_native_trajectory_reports_use_engine_signatures(tmp_path):
    from chemur.models import AssignedInteraction
    from chemur.trajectory import FrameInteractions, TrajectoryResult
    interaction = AssignedInteraction("hydrophobic_enclosure", (), (0, 1, 2, 3, 4), (),
                                      3.5, 135, "inter", 1,
                                      metadata={"center_atom_id": 0, "neighbor_atom_ids": [1, 2, 3, 4]})
    result = TrajectoryResult([], [], [7], [FrameInteractions(7, [interaction])])
    export.write_native_json(result, tmp_path / "trajectory.json")
    payload = json.loads((tmp_path / "trajectory.json").read_text())
    assert payload["rows"][0]["metadata"] == interaction.metadata
    export.write_native_csv(result, tmp_path / "trajectory.csv")
    with (tmp_path / "trajectory.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["frame"] == "7"
    assert json.loads(rows[0]["metadata"]) == interaction.metadata


@pytest.mark.parametrize("format_name", ["JSON", "CSV"])
def test_output_format_choice_appends_missing_suffix(format_name):
    path, actual = export.resolve_output_path("interactions", "%s (*.%s)" % (format_name, format_name.lower()))
    assert actual == format_name.lower()
    assert path == Path("interactions." + actual)
    with pytest.raises(ValueError, match="match the selected"):
        export.resolve_output_path("interactions." + ("csv" if actual == "json" else "json"), format_name)


def test_image_export_format_is_explicit(tmp_path):
    target, kind = export.resolve_output_path(tmp_path / 'diagram', 'SVG image (*.svg)',
                                             allowed=('png', 'svg', 'pdf'))
    assert kind == 'svg' and target.name == 'diagram.svg'
    with pytest.raises(ValueError, match='extension'):
        export.resolve_output_path(tmp_path / 'diagram.png', 'PDF document (*.pdf)',
                                   allowed=('png', 'svg', 'pdf'))

"""Tests for the docked-pose comparison matrix/column assembly (pure logic).

``pose_compare`` has no ChimeraX/Qt imports at module load (the engine + ChimeraX
calls live inside ``compute_pose_comparison``), so it loads outside ChimeraX the
same way ``test_runner_mapping`` loads ``runner``. These cover the residue-column
ordering, multi-type cell accumulation, sequence-context expansion, residue
labels, and the two hit-extraction paths (mapped rows vs the engine-result
fallback).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"


def _load(sub):
    # Distinct from test_runner_mapping's synthetic package so the two loaders
    # never share package state (collection order would otherwise matter).
    pkg_name = "chemur_chimerax_plugin_pose_test"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_SRC)]
        sys.modules[pkg_name] = pkg
    full = f"{pkg_name}.{sub}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, _SRC / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


pc = _load("pose_compare")
H = pc.Hit


# ----- light stand-ins -----------------------------------------------------

class _Endpoint:
    def __init__(self, molecule_type, chain_id, residue_id, residue_name, atom=None):
        self.molecule_type = molecule_type
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.residue_name = residue_name
        self.atom = atom


class _Row:
    def __init__(self, itype, ep1, ep2):
        self.interaction_type = itype
        self.endpoint1 = ep1
        self.endpoint2 = ep2


class _Rec:
    def __init__(self, atom_id, mt, chain, rid, rname):
        self.atom_id = atom_id
        self.molecule_type = mt
        self.chain_id = chain
        self.residue_id = rid
        self.residue_name = rname


class _Inter:
    def __init__(self, itype, atom_ids):
        self.interaction_type = itype
        self.atom_ids = tuple(atom_ids)


class _Result:
    def __init__(self, atoms, interactions):
        self.atoms = atoms
        self.interactions = interactions


# ----- small helpers -------------------------------------------------------

def test_resnum_strips_insertion_code():
    assert pc._resnum("133") == 133
    assert pc._resnum("133A") == 133
    assert pc._resnum("-5") == -5
    assert pc._resnum("nonnumeric") is None


def test_one_letter():
    assert pc.one_letter("THR") == "T"
    assert pc.one_letter("thr") == "T"
    assert pc.one_letter("LIG") == "X"
    assert pc.one_letter("") == "X"


def test_protein_ligand_endpoints():
    prot = _Endpoint("protein", "A", "133", "THR")
    lig = _Endpoint("ligand", "B", "1", "LIG")
    water = _Endpoint("solvent", "W", "5", "HOH")
    assert pc._protein_ligand_endpoints(_Row("hbond", prot, lig)) == (prot, lig)
    assert pc._protein_ligand_endpoints(_Row("hbond", lig, prot)) == (prot, lig)
    # protein-protein, protein-solvent (crystallographic water) and ligand-ligand
    # are all excluded — only protein <-> the docked ligand counts.
    assert pc._protein_ligand_endpoints(_Row("hbond", prot, prot)) is None
    assert pc._protein_ligand_endpoints(_Row("hbond", prot, water)) is None
    assert pc._protein_ligand_endpoints(_Row("hydrophobic", lig, lig)) is None


# ----- hit extraction ------------------------------------------------------

def test_hits_from_rows():
    prot = _Endpoint("protein", "A", "133", "THR", atom="P")
    lig = _Endpoint("ligand", "B", "1", "LIG", atom="L")
    row = _Row("hbond", prot, lig)
    hits = pc._hits_from_rows([row])
    assert len(hits) == 1
    h = hits[0]
    assert (h.chain_id, h.number, h.resname, h.itype) == ("A", 133, "THR", "hbond")
    assert h.row is row
    # a ligand-ligand contact and a protein-water contact both contribute nothing.
    water = _Endpoint("solvent", "W", "5", "HOH")
    assert pc._hits_from_rows([_Row("hydrophobic", lig, lig)]) == []
    assert pc._hits_from_rows([_Row("hbond", prot, water)]) == []


def test_hits_from_result_requires_protein_and_ligand():
    atoms = [_Rec(1, "protein", "A", "133", "THR"), _Rec(2, "ligand", "B", "1", "LIG")]
    res = _Result(atoms, [_Inter("hbond", [1, 2])])
    hits = pc._hits_from_result(res)
    assert len(hits) == 1
    assert (hits[0].chain_id, hits[0].number, hits[0].itype) == ("A", 133, "hbond")
    assert hits[0].row is None
    # protein-protein interaction (no ligand atom) is ignored.
    res2 = _Result(
        [_Rec(1, "protein", "A", "133", "THR"), _Rec(3, "protein", "A", "134", "GLY")],
        [_Inter("hbond", [1, 3])])
    assert pc._hits_from_result(res2) == []


# ----- matrix assembly -----------------------------------------------------

def test_build_matrix_accumulates_types_and_rows():
    real_row = object()
    hits0 = [
        H("A", 133, "133", "THR", "hbond", real_row),
        H("A", 133, "133", "THR", "hydrophobic", None),
        H("A", 86, "86", "ASP", "salt_bridge", None),
        H("A", None, "x", "LIG", "hbond", None),   # non-numeric -> dropped
    ]
    hits1 = [
        H("A", 133, "133", "THR", "hbond", None),
        H("B", 5, "5", "TRP", "pipi_stack", None),
    ]
    rows, cell, cell_rows, resname_by_key, types, interacting = pc.build_matrix(
        [(object(), "p0", True), (object(), "p1", True)], [hits0, hits1])

    assert [r.name for r in rows] == ["p0", "p1"]
    assert cell[(0, ("A", 133))] == {"hbond", "hydrophobic"}
    assert cell[(0, ("A", 86))] == {"salt_bridge"}
    assert cell[(1, ("A", 133))] == {"hbond"}
    assert cell[(1, ("B", 5))] == {"pipi_stack"}
    assert interacting == {("A", 133), ("A", 86), ("B", 5)}
    assert types == ["hbond", "hydrophobic", "pipi_stack", "salt_bridge"]
    assert resname_by_key[("A", 133)] == "THR"
    # only the one hit carrying a real row lands in cell_rows.
    assert cell_rows[(0, ("A", 133))] == [real_row]
    assert (1, ("A", 133)) not in cell_rows


# ----- contiguous sequence-window columns ----------------------------------

_REF = {("A", n): name for n, name in [
    (131, "ALA"), (132, "ALA"), (133, "THR"), (134, "GLY"), (135, "LEU"),
    (136, "VAL"), (137, "ILE")]}
_RESNAMES = {("A", 133): "THR", ("A", 135): "LEU"}


def test_columns_context_zero_is_interacting_only():
    cols = pc.build_columns({("A", 133), ("A", 135)}, _RESNAMES, _REF, context=0)
    assert [(c.chain_id, c.number) for c in cols] == [("A", 133), ("A", 135)]
    assert all(not c.is_context for c in cols)


def test_columns_context_fills_and_marks_flanks():
    cols = pc.build_columns({("A", 133), ("A", 135)}, _RESNAMES, _REF, context=1)
    assert [c.number for c in cols] == [132, 133, 134, 135, 136]
    assert [c.is_context for c in cols] == [True, False, True, False, True]


def test_columns_skip_flanks_absent_from_reference():
    ref = {k: v for k, v in _REF.items() if k != ("A", 136)}
    cols = pc.build_columns({("A", 133), ("A", 135)}, _RESNAMES, ref, context=1)
    assert [c.number for c in cols] == [132, 133, 134, 135]  # 136 not in reference


def test_columns_sorted_across_chains():
    interacting = {("B", 5), ("A", 133)}
    resnames = {("A", 133): "THR", ("B", 5): "TRP"}
    ref = {("A", 133): "THR", ("B", 5): "TRP"}
    cols = pc.build_columns(interacting, resnames, ref, context=0)
    assert [(c.chain_id, c.number) for c in cols] == [("A", 133), ("B", 5)]


def test_column_labels():
    cols = pc.build_columns({("A", 133)}, {("A", 133): "THR"},
                            {("A", 133): "THR"}, context=0)
    c = cols[0]
    assert c.label1 == "T133"
    assert c.label3 == "Thr133"


# ----- PoseComparison context rebuild --------------------------------------

def test_rebuild_columns_changes_window_without_engine():
    comp = pc.PoseComparison(
        rows=[], cell={}, cell_rows={}, types=[], receptor_name="r",
        reference_index=_REF, resname_by_key={("A", 133): "THR"},
        interacting={("A", 133)}, context=0)
    assert [c.number for c in comp.columns] == [133]
    comp.rebuild_columns(2)
    assert comp.context == 2
    assert [c.number for c in comp.columns] == [131, 132, 133, 134, 135]


def _consensus_comparison():
    rows = [pc.PoseRow(object(), "p%d" % i, True) for i in range(4)]
    cell = {
        (0, ("A", 10)): {"hbond"}, (1, ("A", 10)): {"hbond"},
        (2, ("A", 10)): {"hbond"},                       # hbond 3/4 = 0.75
        (0, ("A", 11)): {"salt_bridge"}, (1, ("A", 11)): {"salt_bridge"},  # 2/4 = 0.5
    }
    return pc.PoseComparison(rows, cell, {}, ["hbond", "salt_bridge"], "r",
                             {}, {}, set(), 0)


def test_consensus_majority_threshold():
    comp = _consensus_comparison()
    out = pc.compute_consensus(comp, threshold=0.5)
    assert out == {("A", 10): {"hbond": 0.75}, ("A", 11): {"salt_bridge": 0.5}}


def test_consensus_strict_threshold_drops_partial():
    comp = _consensus_comparison()
    # 0.7: keeps hbond (0.75), drops salt_bridge (0.5).
    assert pc.compute_consensus(comp, threshold=0.7) == {("A", 10): {"hbond": 0.75}}
    # 0.8 and 1.0: nothing is made by that fraction of the 4 poses.
    assert pc.compute_consensus(comp, threshold=0.8) == {}
    assert pc.compute_consensus(comp, threshold=1.0) == {}


def test_consensus_respects_type_filter():
    comp = _consensus_comparison()
    out = pc.compute_consensus(comp, include_types={"hbond"}, threshold=0.5)
    assert out == {("A", 10): {"hbond": 0.75}}


def test_consensus_empty_when_no_poses():
    comp = pc.PoseComparison([], {}, {}, [], "r", {}, {}, set(), 0)
    assert pc.compute_consensus(comp) == {}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

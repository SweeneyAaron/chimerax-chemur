"""Tests for the ChimeraX plugin's ChemeleonX-atom -> ChimeraX-atom back-mapping.

The plugin used to resolve atoms purely by ``(chain, residue_id, residue_name,
atom_name)``. On large multi-copy assemblies (e.g. 6PUW, the HIV-integrase
intasome) that name key is not unique, so an endpoint could snap to the same-named
atom in a *different* copy 10+ Å away and draw a bogus long interaction line.

These tests exercise the coordinate-primary resolver in ``runner.py`` without
needing a running ChimeraX: the module is loaded as a synthetic package and the
ChimeraX/ChemeleonX objects are replaced with light stubs that expose only the
attributes the resolver touches.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"


def _load_runner():
    """Load ``runner.py`` (and its pure ``colors`` dependency) outside ChimeraX."""
    pkg_name = "chemeleonx_chimerax_plugin_under_test"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_SRC)]
        sys.modules[pkg_name] = pkg
        for sub in ("colors", "runner"):
            spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.{sub}", _SRC / f"{sub}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"{pkg_name}.{sub}"] = mod
            spec.loader.exec_module(mod)
    return sys.modules[f"{pkg_name}.runner"]


runner = _load_runner()


class _Residue:
    def __init__(self, chain_id, number, name, insertion_code=""):
        self.chain_id = chain_id
        self.number = number
        self.name = name
        self.insertion_code = insertion_code


class _CxAtom:
    """Minimal stand-in for a ChimeraX Atom."""

    def __init__(self, name, coord, residue):
        self.name = name
        self.coord = coord
        self.residue = residue


class _Structure:
    def __init__(self, atoms):
        self.atoms = atoms


class _Rec:
    """Minimal stand-in for a ChemeleonX ``AtomRecord``."""

    def __init__(self, name, coord, chain_id, residue_id, residue_name):
        self.name = name
        self.coord = coord
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.residue_name = residue_name


def _two_copy_structure():
    """Two ligand copies sharing an identical name key but far apart in space."""
    res_a = _Residue("B", 1, "KLQ")
    res_b = _Residue("B", 1, "KLQ")  # same chain/number/name -> colliding key
    atom_a = _CxAtom("C1", (0.0, 0.0, 0.0), res_a)
    atom_b = _CxAtom("C1", (20.0, 20.0, 20.0), res_b)
    return _Structure([atom_a, atom_b]), atom_a, atom_b


def test_coordinate_resolves_correct_copy_not_first():
    """The wrong-copy bug: name key collides, coordinate must win."""
    structure, atom_a, atom_b = _two_copy_structure()
    coord_index, name_index = runner._chimerax_atom_lookup(structure)

    # The name index keeps the first copy (the old, buggy behaviour).
    assert name_index[("B", "1", "KLQ", "C1")] is atom_a

    # A ChemeleonX record for the *second* copy must resolve to atom_b by coordinate,
    # not collapse onto atom_a 34 Å away.
    rec = _Rec("C1", (20.0, 20.0, 20.0), "B", "1", "KLQ")
    assert runner._resolve_atom(rec, coord_index, name_index) is atom_b


def test_unmappable_atom_returns_none_instead_of_wrong_line():
    """No coordinate match and only a far name-key hit -> refuse to map."""
    structure, atom_a, _ = _two_copy_structure()
    coord_index, name_index = runner._chimerax_atom_lookup(structure)

    rec = _Rec("C1", (50.0, 50.0, 50.0), "B", "1", "KLQ")  # matches no real atom
    assert runner._resolve_atom(rec, coord_index, name_index) is None


def test_name_fallback_within_tolerance():
    """Sub-rounding coordinate drift misses the coord key but the name key,
    validated within EPS, still resolves."""
    res = _Residue("A", 10, "SER")
    atom = _CxAtom("OG", (1.000, 2.000, 3.000), res)
    structure = _Structure([atom])
    coord_index, name_index = runner._chimerax_atom_lookup(structure)

    # 0.001 Å off -> different quantised key (coord miss) but well within EPS.
    rec = _Rec("OG", (1.001, 2.000, 3.000), "A", "10", "SER")
    assert runner._coord_key(rec.coord) not in coord_index
    assert runner._resolve_atom(rec, coord_index, name_index) is atom


def test_exact_match_resolves():
    res = _Residue("A", 10, "SER")
    atom = _CxAtom("OG", (1.234, 5.678, 9.012), res)
    structure = _Structure([atom])
    coord_index, name_index = runner._chimerax_atom_lookup(structure)

    rec = _Rec("OG", (1.234, 5.678, 9.012), "A", "10", "SER")
    assert runner._resolve_atom(rec, coord_index, name_index) is atom


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

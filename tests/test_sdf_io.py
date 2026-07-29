"""Tests for the plugin's minimal V2000 SDF writer (sdf_io).

``sdf_io`` has no ChimeraX/Qt imports, so it loads standalone. The RDKit
round-trip (the engine reads our SDF with RDKit) is gated behind importorskip so
the suite still runs where RDKit is absent.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"


def _load_sdf_io():
    name = "chemeleonx_chimerax_plugin_sdf_io"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _SRC / "sdf_io.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


sdf_io = _load_sdf_io()


class _Bond:
    def __init__(self, order=None):
        if order is not None:
            self.order = order


def test_has_bond_orders():
    class S:
        bonds = [_Bond(), _Bond(2.0)]
    assert sdf_io.has_bond_orders(S) is True

    class S2:
        bonds = [_Bond(), _Bond()]
    assert sdf_io.has_bond_orders(S2) is False


def test_bond_order_clamps_and_rounds():
    assert sdf_io._bond_order(_Bond(2.0)) == 2
    assert sdf_io._bond_order(_Bond(4.0)) == 4       # aromatic kept
    assert sdf_io._bond_order(_Bond(1.6)) == 2       # rounds
    assert sdf_io._bond_order(_Bond()) == 1          # missing -> single
    assert sdf_io._bond_order(_Bond(7.0)) == 1       # out of range -> single
    assert sdf_io._bond_order(_Bond(0.0)) == 1


def test_molfile_counts_and_charge_line():
    atoms = [("N", (0.0, 0.0, 0.0), 1), ("C", (1.5, 0.0, 0.0), 0)]
    text = sdf_io.molfile_text(atoms, [(1, 2, 1)], name="TST")
    lines = text.splitlines()
    assert lines[0] == "TST"
    # counts line: 2 atoms, 1 bond
    assert lines[3].startswith("  2  1")
    assert "V2000" in lines[3]
    # the +1 on atom 1 must appear as an M CHG record
    assert any(l.startswith("M  CHG") and "  1   1" in l for l in lines)
    assert lines[-1] == "M  END"


def test_molfile_rejects_oversized():
    atoms = [("C", (0.0, 0.0, 0.0), 0)] * 1000
    with pytest.raises(ValueError):
        sdf_io.molfile_text(atoms, [], name="BIG")


def test_rdkit_roundtrip_preserves_bond_orders():
    Chem = pytest.importorskip("rdkit.Chem", reason="RDKit not installed")
    # CO2: C double-bonded to two O (neutral, sanitizes cleanly).
    atoms = [
        ("C", (0.0, 0.0, 0.0), 0),
        ("O", (1.16, 0.0, 0.0), 0),
        ("O", (-1.16, 0.0, 0.0), 0),
    ]
    bonds = [(1, 2, 2), (1, 3, 2)]
    text = sdf_io.molfile_text(atoms, bonds, name="CO2")

    fh = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w")
    fh.write(text)
    fh.write("$$$$\n")
    fh.close()
    try:
        mol = next(iter(Chem.SDMolSupplier(fh.name, sanitize=True, removeHs=False)))
    finally:
        os.unlink(fh.name)

    assert mol is not None
    assert mol.GetNumAtoms() == 3
    n_double = sum(1 for b in mol.GetBonds()
                   if b.GetBondType() == Chem.BondType.DOUBLE)
    assert n_double == 2


class _Elem:
    def __init__(self, number, name):
        self.number = number
        self.name = name


class _Atom:
    def __init__(self, number, name, coord):
        self.element = _Elem(number, name)
        self.coord = coord


class _CxBond:
    def __init__(self, a1, a2, order):
        self.atoms = (a1, a2)
        self.order = order


class _Struct:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds


def test_write_ligand_sdf_via_rdkit_preserves_double_bonds():
    Chem = pytest.importorskip("rdkit.Chem", reason="RDKit not installed")
    c = _Atom(6, "C", (0.0, 0.0, 0.0))
    o1 = _Atom(8, "O", (1.16, 0.0, 0.0))
    o2 = _Atom(8, "O", (-1.16, 0.0, 0.0))
    struct = _Struct([c, o1, o2], [_CxBond(c, o1, 2.0), _CxBond(c, o2, 2.0)])

    fh = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    fh.close()
    try:
        sdf_io.write_ligand_sdf(struct, fh.name, name="CO2")
        mol = next(iter(Chem.SDMolSupplier(fh.name, True, False)))
    finally:
        os.unlink(fh.name)
    assert mol is not None
    assert mol.GetNumAtoms() == 3
    assert sum(1 for b in mol.GetBonds()
               if b.GetBondType() == Chem.BondType.DOUBLE) == 2


def test_write_ligand_sdf_adds_hydrogens_by_valence():
    Chem = pytest.importorskip("rdkit.Chem", reason="RDKit not installed")
    # Methanol heavy atoms only; RDKit should add the 4 missing hydrogens.
    c = _Atom(6, "C", (0.0, 0.0, 0.0))
    o = _Atom(8, "O", (1.4, 0.0, 0.0))
    struct = _Struct([c, o], [_CxBond(c, o, 1.0)])

    fh = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    fh.close()
    try:
        sdf_io.write_ligand_sdf(struct, fh.name, name="MOH", add_hydrogens=True)
        mol = next(iter(Chem.SDMolSupplier(fh.name, True, False)))
    finally:
        os.unlink(fh.name)
    assert mol is not None
    n_h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    assert n_h == 4  # CH3 + OH


def test_write_ligand_sdf_infers_cationic_nitrogen():
    Chem = pytest.importorskip("rdkit.Chem", reason="RDKit not installed")
    # Quaternary ammonium: N bonded to 4 carbons. ChimeraX drops the +1, leaving an
    # over-valent neutral N that RDKit would reject — write_ligand_sdf must infer it.
    n = _Atom(7, "N", (0.0, 0.0, 0.0))
    cs = [_Atom(6, "C", xyz) for xyz in
          [(1.5, 0, 0), (-1.5, 0, 0), (0, 1.5, 0), (0, -1.5, 0)]]
    struct = _Struct([n] + cs, [_CxBond(n, c, 1.0) for c in cs])

    fh = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    fh.close()
    try:
        sdf_io.write_ligand_sdf(struct, fh.name, name="NME4")
        mol = next(iter(Chem.SDMolSupplier(fh.name, True, False)))
    finally:
        os.unlink(fh.name)
    assert mol is not None, "charged-N ligand must still sanitise"
    natom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    assert natom.GetFormalCharge() == 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

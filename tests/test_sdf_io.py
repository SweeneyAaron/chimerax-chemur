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
    name = "chemur_chimerax_plugin_sdf_io"
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


def _hydroxyl_h(mol):
    """The single hydrogen bonded to the molecule's only oxygen."""
    o = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    h = next(n for n in o.GetNeighbors() if n.GetAtomicNum() == 1)
    conf = mol.GetConformer()
    return tuple(conf.GetAtomPosition(h.GetIdx()))


def _methanol():
    c = _Atom(6, "C", (0.0, 0.0, 0.0))
    o = _Atom(8, "O", (1.4, 0.0, 0.0))
    return _Struct([c, o], [_CxBond(c, o, 1.0)])


def _write(struct, **kwargs):
    Chem = pytest.importorskip("rdkit.Chem", reason="RDKit not installed")
    fh = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    fh.close()
    try:
        sdf_io.write_ligand_sdf(struct, fh.name, **kwargs)
        return next(iter(Chem.SDMolSupplier(fh.name, True, False)))
    finally:
        os.unlink(fh.name)


def test_orient_polar_hydrogens_turns_hydroxyl_toward_acceptor():
    """The bug this guards: an unoriented hydroxyl H silently costs a hydrogen bond.

    RDKit places the OH hydrogen at an arbitrary torsion, so with an acceptor sat
    off the C-O axis the H must swing toward it rather than stay where AddHs left it.
    """
    pytest.importorskip("rdkit.Chem", reason="RDKit not installed")
    # Acceptor placed ~2.7 A from the hydroxyl O, off-axis so only one torsion suits.
    acceptor = (2.0, 2.4, 0.0)
    plain = _write(_methanol(), name="MOH", add_hydrogens=True)
    turned = _write(_methanol(), name="MOH", add_hydrogens=True,
                    acceptor_coords=[acceptor])

    def h_to_acceptor(mol):
        h = _hydroxyl_h(mol)
        return sum((a - b) ** 2 for a, b in zip(h, acceptor)) ** 0.5

    assert h_to_acceptor(turned) < h_to_acceptor(plain), \
        "hydroxyl H should end up closer to the acceptor"
    # It should be a credible hydrogen bond, not merely an improvement.
    assert h_to_acceptor(turned) < 2.2


def test_orient_polar_hydrogens_noop_without_acceptors():
    """No acceptors given, or none in range: the hydrogen must not be invented."""
    pytest.importorskip("rdkit.Chem", reason="RDKit not installed")
    plain = _write(_methanol(), name="MOH", add_hydrogens=True)
    none_given = _write(_methanol(), name="MOH", add_hydrogens=True,
                        acceptor_coords=[])
    far_away = _write(_methanol(), name="MOH", add_hydrogens=True,
                      acceptor_coords=[(50.0, 50.0, 50.0)])
    assert _hydroxyl_h(none_given) == pytest.approx(_hydroxyl_h(plain), abs=1e-6)
    assert _hydroxyl_h(far_away) == pytest.approx(_hydroxyl_h(plain), abs=1e-6)


def test_orient_polar_hydrogens_keeps_phenol_in_plane():
    """A conjugated hydroxyl may flip 180 degrees but must stay in the ring plane."""
    Chem = pytest.importorskip("rdkit.Chem", reason="RDKit not installed")
    mol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"), addCoords=True)
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    conf = mol.GetConformer()

    o = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    ring_c = next(n for n in o.GetNeighbors() if n.GetAtomicNum() == 6)
    ortho = [n for n in ring_c.GetNeighbors() if n.GetIdx() != o.GetIdx()
             and n.GetAtomicNum() == 6]

    def plane_offset():
        """Distance of the hydroxyl H from the plane of C(ring) and its two neighbours."""
        import numpy
        h = next(n for n in o.GetNeighbors() if n.GetAtomicNum() == 1)
        p = [numpy.array(conf.GetAtomPosition(i.GetIdx()))
             for i in (ring_c, ortho[0], ortho[1])]
        normal = numpy.cross(p[1] - p[0], p[2] - p[0])
        normal = normal / numpy.linalg.norm(normal)
        return abs(numpy.dot(numpy.array(conf.GetAtomPosition(h.GetIdx())) - p[0],
                             normal))

    # An acceptor placed well out of the ring plane would drag a freely rotating
    # hydroxyl with it; a conjugated one must refuse to leave the plane. Assert the
    # H ends up genuinely in the plane rather than merely unmoved -- AddHs may have
    # started it out of plane, in which case "unchanged" would still be wrong.
    import numpy
    o = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    off_plane_acceptor = tuple(
        numpy.array(conf.GetAtomPosition(o.GetIdx())) + numpy.array([0.0, 0.0, 2.8]))
    sdf_io.orient_polar_hydrogens(mol, [off_plane_acceptor])
    assert plane_offset() < 0.2, "conjugated hydroxyl H must stay in the ring plane"


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


def test_sdf_uses_scene_coordinates_and_does_not_rotate_existing_hydrogens():
    structure = _methanol()
    hydrogen = _Atom(1, "H", (1.7, -0.9, 0.0))
    structure.atoms.append(hydrogen)
    structure.bonds.append(_CxBond(structure.atoms[1], hydrogen, 1.0))
    for atom in structure.atoms:
        atom.scene_coord = tuple(v + shift for v, shift in zip(atom.coord, (10., 20., 30.)))
    molecule = _write(structure, add_hydrogens=True, acceptor_coords=[(12.0, 22.4, 30.0)])
    conf = molecule.GetConformer()
    for index, atom in enumerate(structure.atoms):
        assert tuple(conf.GetAtomPosition(index)) == pytest.approx(atom.scene_coord, abs=1e-4)


@pytest.mark.parametrize("smiles", ["OCCNC=O", "OCC[NH3+]", "OC=O"])
def test_orientation_rejects_nonacceptors_and_covalent_neighbors(smiles, monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(molecule, randomSeed=17)
    conf = molecule.GetConformer()
    # Keep a nonaccepting amide/cationic nitrogen near OH, while any chemically
    # valid distant carbonyl is outside reach. The acid carbonyl is 1-3 excluded.
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            conf.SetAtomPosition(atom.GetIdx(), (float(atom.GetIdx()) * 10., 0., 0.))
    conf.SetAtomPosition(0, (0., 0., 0.))
    conf.SetAtomPosition(1, (-1.4, 0., 0.))
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 7 or (smiles == "OC=O" and atom.GetIdx() == 2):
            conf.SetAtomPosition(atom.GetIdx(), (1., 2., 0.))
    monkeypatch.setattr(sdf_io, "_hbond_score", lambda *args: pytest.fail("No valid acceptor should be scored"))
    assert sdf_io.orient_polar_hydrogens(molecule, []) == 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

"""Write a ChimeraX structure to an MDL V2000 SDF (molfile) for the engine.

A docked ligand loaded from an SDF carries its bond orders on each ChimeraX
``Bond`` (the SDF reader registers a ``Bond.order`` attribute). ChemeleonX's engine
is template-driven and only recovers ligand chemistry from a SMILES/CCD/SDF, NOT
from a ChimeraX-written mmCIF (which loses bond orders). So to analyse a docked
ligand we re-emit it as an SDF — preserving element, coordinates and bond order —
and hand that to ``chemeleonx.analyze(receptor, ligand_sdf=[...])``, whose RDKit path
reconstructs the full chemistry.

ChimeraX has no SDF *saver*, hence this small writer. Pure string/IO so it is
unit-testable without ChimeraX.
"""

from __future__ import annotations

# V2000 fixed-width fields cap counts at 3 digits.
_MAX_V2000 = 999


def has_bond_orders(structure):
    """True if any of ``structure``'s bonds carries an explicit ``order``.

    Only structures opened from an SDF/MOL file have it (ChimeraX's SDF reader
    sets ``Bond.order``); a ligand loaded another way would otherwise be written
    with all-single bonds, so callers use this to decide whether the SDF route is
    trustworthy.
    """
    for bond in structure.bonds:
        if getattr(bond, "order", None) is not None:
            return True
    return False


def _bond_order(bond):
    o = getattr(bond, "order", None)
    if o is None:
        return 1
    o = int(round(o))
    return o if 1 <= o <= 4 else 1


def molfile_text(atoms, bonds_with_order, name="LIG"):
    """Build a V2000 molfile string (no ``$$$$``).

    ``atoms`` is a list of ``(symbol, (x, y, z), formal_charge)``; ``bonds_with_order``
    a list of ``(i, j, order)`` with 1-based atom indices. Factored out from
    :func:`write_ligand_sdf` so it can be tested without ChimeraX objects.
    """
    n_atoms = len(atoms)
    n_bonds = len(bonds_with_order)
    if n_atoms > _MAX_V2000 or n_bonds > _MAX_V2000:
        raise ValueError("structure too large for a V2000 molfile (>999 atoms/bonds)")

    lines = [name[:80], "  ChemeleonX", ""]
    lines.append("%3d%3d  0  0  0  0  0  0  0  0999 V2000" % (n_atoms, n_bonds))
    for symbol, (x, y, z), _charge in atoms:
        lines.append("%10.4f%10.4f%10.4f %-3s 0  0  0  0  0  0  0  0  0  0  0  0"
                     % (x, y, z, symbol))
    for i, j, order in bonds_with_order:
        lines.append("%3d%3d%3d  0  0  0  0" % (i, j, order))

    # Emit explicit formal charges (M  CHG) so RDKit doesn't have to guess them
    # from valence; only nonzero charges are listed.
    charged = [(idx + 1, c) for idx, (_s, _xyz, c) in enumerate(atoms) if c]
    for start in range(0, len(charged), 8):
        chunk = charged[start:start + 8]
        entry = "".join(" %3d %3d" % (idx, c) for idx, c in chunk)
        lines.append("M  CHG%3d%s" % (len(chunk), entry))

    lines.append("M  END")
    return "\n".join(lines) + "\n"


def write_ligand_sdf(structure, path, *, name="LIG", add_hydrogens=False):
    """Write ``structure`` (a ChimeraX AtomicStructure) to an SDF at ``path``.

    Builds an RDKit molecule from the atoms + stored ``Bond.order`` and writes it
    with RDKit's own SDF writer, so the molblock is guaranteed well-formed and
    aromaticity/valence are handled correctly — the engine reads it back with
    ``SDMolSupplier(sanitize=True)``. With ``add_hydrogens`` explicit hydrogens are
    added by RDKit (valence-consistent; a ChimeraX ``addh`` can instead break the
    SDF's own valence so RDKit then fails to sanitise it). Falls back to a
    hand-written V2000 molfile only if RDKit is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BondType
        from rdkit.Geometry import Point3D
    except ImportError:
        return _write_molfile_fallback(structure, path, name=name)

    order_to_bt = {1: BondType.SINGLE, 2: BondType.DOUBLE,
                   3: BondType.TRIPLE, 4: BondType.AROMATIC}
    atom_list = list(structure.atoms)
    rw = Chem.RWMol()
    rindex = {}
    for a in atom_list:
        ratom = Chem.Atom(int(a.element.number))
        fc = int(getattr(a, "formal_charge", 0) or 0)
        if fc:
            ratom.SetFormalCharge(fc)
        rindex[a] = rw.AddAtom(ratom)
    for bond in structure.bonds:
        a1, a2 = bond.atoms
        i, j = rindex.get(a1), rindex.get(a2)
        if i is None or j is None:
            continue
        bt = order_to_bt.get(_bond_order(bond), BondType.SINGLE)
        rw.AddBond(i, j, bt)
        if bt == BondType.AROMATIC:
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            rw.GetAtomWithIdx(j).SetIsAromatic(True)

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for a in atom_list:
        c = a.coord
        conf.SetAtomPosition(rindex[a], Point3D(float(c[0]), float(c[1]), float(c[2])))
    mol.AddConformer(conf, assignId=True)
    # ChimeraX's SDF reader drops formal charges, so a charged group (e.g. a
    # protonated/quaternary amine) arrives as an over-valent neutral atom that
    # RDKit refuses to sanitise. Infer the unambiguous cationic charges first.
    mol.UpdatePropertyCache(strict=False)
    _infer_formal_charges(mol)
    Chem.SanitizeMol(mol)
    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=True)
    mol.SetProp("_Name", name)
    writer = Chem.SDWriter(path)
    try:
        writer.write(mol)
    finally:
        writer.close()
    return path


# Elements whose over-valence is unambiguously a positive formal charge (rather
# than legitimate expanded valence, as for S/P/halogens). Maps element -> the
# neutral valence to correct from.
_CATION_NEUTRAL_VALENCE = {7: 3, 8: 2}  # N, O


def _infer_formal_charges(mol):
    """Assign the obvious cationic formal charges ChimeraX's SDF reader drops.

    A nitrogen with 4 explicit bonds (or oxygen with 3) is unambiguously +1
    (pyridinium, protonated/quaternary amine, oxonium); without it RDKit raises
    "Explicit valence ... greater than permitted". Expanded-valence elements
    (S, P, halogens) are deliberately left untouched. Call after
    ``UpdatePropertyCache(strict=False)`` and before ``SanitizeMol``.
    """
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            continue
        base = _CATION_NEUTRAL_VALENCE.get(atom.GetAtomicNum())
        if base is None:
            continue
        valence = round(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()))
        excess = valence - base
        if excess > 0:
            atom.SetFormalCharge(excess)


def _write_molfile_fallback(structure, path, *, name="LIG"):
    """Hand-written V2000 SDF (used only when RDKit is unavailable)."""
    atom_list = list(structure.atoms)
    index = {a: i + 1 for i, a in enumerate(atom_list)}
    atoms = [(a.element.name,
              (float(a.coord[0]), float(a.coord[1]), float(a.coord[2])),
              int(getattr(a, "formal_charge", 0) or 0))
             for a in atom_list]
    bonds_with_order = []
    for bond in structure.bonds:
        i = index.get(bond.atoms[0])
        j = index.get(bond.atoms[1])
        if i is not None and j is not None:
            bonds_with_order.append((i, j, _bond_order(bond)))
    with open(path, "w") as fh:
        fh.write(molfile_text(atoms, bonds_with_order, name=name))
        fh.write("$$$$\n")
    return path

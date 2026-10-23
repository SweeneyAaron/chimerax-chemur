"""Write a ChimeraX structure to an MDL V2000 SDF (molfile) for the engine.

A docked ligand loaded from an SDF carries its bond orders on each ChimeraX
``Bond`` (the SDF reader registers a ``Bond.order`` attribute). Chemur's engine
is template-driven and only recovers ligand chemistry from a SMILES/CCD/SDF, NOT
from a ChimeraX-written mmCIF (which loses bond orders). So to analyse a docked
ligand we re-emit it as an SDF — preserving element, coordinates and bond order —
and hand that to ``chemur.analyze(receptor, ligand_sdf=[...])``, whose RDKit path
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

    lines = [name[:80], "  Chemur", ""]
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


def write_ligand_sdf(structure, path, *, name="LIG", add_hydrogens=False,
                     acceptor_coords=None):
    """Write ``structure`` (a ChimeraX AtomicStructure) to an SDF at ``path``.

    Builds an RDKit molecule from the atoms + stored ``Bond.order`` and writes it
    with RDKit's own SDF writer, so the molblock is guaranteed well-formed and
    aromaticity/valence are handled correctly — the engine reads it back with
    ``SDMolSupplier(sanitize=True)``. With ``add_hydrogens`` explicit hydrogens are
    added by RDKit (valence-consistent; a ChimeraX ``addh`` can instead break the
    SDF's own valence so RDKit then fails to sanitise it). Falls back to a
    hand-written V2000 molfile only if RDKit is unavailable.

    ``acceptor_coords`` — an iterable of ``(x, y, z)`` for the receptor's H-bond
    acceptors — additionally orients the rotatable polar hydrogens RDKit just
    added; see :func:`orient_polar_hydrogens` for why that matters. Omitting it
    leaves the hydrogens exactly where RDKit put them.
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
        c = getattr(a, "scene_coord", None)
        if c is None:
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
        original_count = mol.GetNumAtoms()
        mol = Chem.AddHs(mol, addCoords=True)
        if acceptor_coords is not None:
            orient_polar_hydrogens(
                mol, acceptor_coords,
                hydrogen_indices=set(range(original_count, mol.GetNumAtoms())),
            )
    mol.SetProp("_Name", name)
    writer = Chem.SDWriter(path)
    try:
        writer.write(mol)
    finally:
        writer.close()
    return path


# Rotatable polar donors: a hydroxyl/thiol hydrogen turns freely about its
# heavy-heavy bond, so its position carries no information until something orients
# it. Amine and amide N-H are excluded on purpose -- those are fixed by the
# geometry of the heavy atoms, so moving them would be inventing chemistry.
_ROTATABLE_DONORS = (8, 16)  # O, S

# Candidate acceptors are only worth considering within a hydrogen bond's reach.
_ACCEPTOR_SEARCH_RADIUS = 3.6   # A, donor heavy atom to acceptor
_TORSION_STEP = 5               # degrees, for the sp3 scan


def orient_polar_hydrogens(mol, acceptor_coords, *, hydrogen_indices=None):
    """Rotate rotatable polar hydrogens toward nearby acceptors. Returns how many moved.

    ``Chem.AddHs`` places a hydroxyl hydrogen at whatever torsion falls out of the
    heavy-atom geometry, because RDKit knows nothing about the surroundings. That
    position is arbitrary, and a hydroxyl H pointing the wrong way costs a real
    hydrogen bond: the engine sees no O-H...A donor geometry and reports at best a
    carbon-donor weak contact instead. Crystallography does not settle this either
    -- X-ray structures at these resolutions carry no hydrogens at all -- so the
    torsion has to be chosen, and choosing it by the acceptors around it is the
    standard answer.

    Only the hydrogen moves; every heavy atom is left exactly where it was.

    ``acceptor_coords`` is an iterable of ``(x, y, z)`` for the receptor's
    acceptors. The ligand's own acceptors are added automatically, so an
    intramolecular hydrogen bond is not broken to chase a receptor atom.
    """
    import math

    from rdkit.Geometry import Point3D
    from chemur.chemistry import rdkit_atom_annotations

    conf = mol.GetConformer()

    def pos(idx):
        p = conf.GetAtomPosition(idx)
        return (p.x, p.y, p.z)

    targets = [tuple(float(v) for v in c) for c in acceptor_coords]
    ligand_acceptors = {index for index, annotation in rdkit_atom_annotations(mol).items()
                        if annotation.acceptor_capacity > 0}

    moved = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in _ROTATABLE_DONORS:
            continue
        hydrogens = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
        if len(hydrogens) != 1 or len(heavy) != 1:
            continue

        donor, root, hydrogen = atom.GetIdx(), heavy[0].GetIdx(), hydrogens[0].GetIdx()
        if hydrogen_indices is not None and hydrogen not in hydrogen_indices:
            continue
        d_pos, r_pos, h_pos = pos(donor), pos(root), pos(hydrogen)

        # Covalently adjacent (1-2 and 1-3) sites cannot orient a noncovalent H bond.
        excluded = {donor}
        for neighbor in atom.GetNeighbors():
            excluded.add(neighbor.GetIdx())
            excluded.update(other.GetIdx() for other in neighbor.GetNeighbors())
        nearby = [t for t in targets
                  if _distance(d_pos, t) <= _ACCEPTOR_SEARCH_RADIUS]
        nearby += [pos(i) for i in ligand_acceptors
                   if i not in excluded and _distance(d_pos, pos(i)) <= _ACCEPTOR_SEARCH_RADIUS]
        if not nearby:
            continue

        # A hydroxyl on an sp2/aromatic root (phenol, carboxylic acid, enol) is
        # conjugated and near-planar: rotating it out of plane would be wrong, so
        # only the two in-plane torsions are on offer. An sp3 root turns freely.
        #
        # The in-plane pair has to be measured against a real substituent of the
        # root, not against wherever AddHs happened to leave the hydrogen -- if
        # that starting point is itself out of plane, rotating by 0/180 from it
        # just yields two equally wrong positions.
        conjugated = heavy[0].GetIsAromatic() or str(heavy[0].GetHybridization()) == "SP2"
        reference = next((n for n in heavy[0].GetNeighbors()
                          if n.GetIdx() != donor), None)
        if conjugated and reference is not None:
            current = _dihedral(pos(reference.GetIdx()), r_pos, d_pos, h_pos)
            angles = (-current, 180.0 - current)
        else:
            angles = [float(a) for a in range(0, 360, _TORSION_STEP)]

        best, best_score = None, None
        for degrees in angles:
            candidate = _rotate_about_axis(h_pos, r_pos, d_pos, math.radians(degrees))
            score = max(_hbond_score(d_pos, candidate, t) for t in nearby)
            if best_score is None or score > best_score:
                best_score, best = score, candidate
        if best is not None and _distance(best, h_pos) > 1e-6:
            conf.SetAtomPosition(hydrogen, Point3D(*best))
            moved += 1
    return moved


def _distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _hbond_score(donor, hydrogen, acceptor):
    """Rank a candidate H position: prefer a linear D-H...A and a short H...A.

    The weighting is deliberately steep on distance so a hydrogen is not swung
    toward a well-aligned but far-off acceptor in preference to a close one.
    """
    import math

    dh_a = _distance(hydrogen, acceptor)
    v1 = [d - h for d, h in zip(donor, hydrogen)]
    v2 = [a - h for a, h in zip(acceptor, hydrogen)]
    n1 = sum(c * c for c in v1) ** 0.5
    n2 = sum(c * c for c in v2) ** 0.5
    if n1 == 0 or n2 == 0:
        return -1e9
    cosine = sum(c1 * c2 for c1, c2 in zip(v1, v2)) / (n1 * n2)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return angle - 30.0 * dh_a


def _dihedral(p0, p1, p2, p3):
    """Signed dihedral angle p0-p1-p2-p3 in degrees."""
    import math

    b0 = [a - b for a, b in zip(p0, p1)]
    b1 = [a - b for a, b in zip(p2, p1)]
    b2 = [a - b for a, b in zip(p3, p2)]
    n1 = sum(c * c for c in b1) ** 0.5
    if n1 == 0:
        return 0.0
    b1 = [c / n1 for c in b1]
    # Components of b0 and b2 perpendicular to the central b1 axis.
    d0 = sum(a * b for a, b in zip(b0, b1))
    d2 = sum(a * b for a, b in zip(b2, b1))
    v = [a - d0 * b for a, b in zip(b0, b1)]
    w = [a - d2 * b for a, b in zip(b2, b1)]
    x = sum(a * b for a, b in zip(v, w))
    cross = [b1[1] * v[2] - b1[2] * v[1],
             b1[2] * v[0] - b1[0] * v[2],
             b1[0] * v[1] - b1[1] * v[0]]
    y = sum(a * b for a, b in zip(cross, w))
    return math.degrees(math.atan2(y, x))


def _rotate_about_axis(point, axis_start, axis_end, angle):
    """Rotate ``point`` by ``angle`` radians about the axis through the two points."""
    import math

    ax = [e - s for e, s in zip(axis_end, axis_start)]
    norm = sum(c * c for c in ax) ** 0.5
    if norm == 0:
        return point
    ux, uy, uz = (c / norm for c in ax)
    px, py, pz = (p - s for p, s in zip(point, axis_start))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dot = px * ux + py * uy + pz * uz
    # Rodrigues' rotation formula.
    rx = (px * cos_a + (uy * pz - uz * py) * sin_a + ux * dot * (1 - cos_a))
    ry = (py * cos_a + (uz * px - ux * pz) * sin_a + uy * dot * (1 - cos_a))
    rz = (pz * cos_a + (ux * py - uy * px) * sin_a + uz * dot * (1 - cos_a))
    return (rx + axis_start[0], ry + axis_start[1], rz + axis_start[2])


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
              tuple(float(v) for v in getattr(a, "scene_coord", a.coord)),
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

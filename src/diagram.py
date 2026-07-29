"""Build/export a 2D chemical-structure interaction diagram (PoseView/LigPlot style).

Each selected residue/ligand (and its interaction partners) is drawn as a real
RDKit 2D structure (no hydrogens, with formal charges) at a uniform, readable
scale; units are arranged compactly around the ligand by their real 3D
direction, with dashed interaction bonds between the specific interacting atoms.

Chemistry: connectivity from ChimeraX bonds, bond orders from a ChimeraX Mol2
export, formal charges from a standard residue table (local patterns for
ligands). RDKit draws each unit at a fixed bond length to its own transparent
image; matplotlib composites the images and overlays the interaction bonds.
Qt-free for headless testing.
"""

from __future__ import annotations

from .colors import mpl_color

_FALLBACK_FACE = "#e0e0e0"
_FIXED_BOND_PX = 28.0          # on-screen bond length (uniform across all units)
_PPU = _FIXED_BOND_PX / 1.5    # px per RDKit 2D coordinate unit (default bond ~1.5)
_GAP = 2.2                     # gap between a unit and the centre, in coord units

_ORDER_VALUE = {"1": 1.0, "2": 2.0, "3": 3.0, "ar": 1.5, "am": 1.0}

_STD_RESIDUE_CHARGES = {
    "ASP": [("OD2", -1)], "GLU": [("OE2", -1)],
    "LYS": [("NZ", 1)], "ARG": [("NH2", 1)],
    "HIP": [("ND1", 1)],
}
_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "MSE", "SEC", "PYL", "CYX",
    "A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU",
}


class _Unit:
    __slots__ = ("residue", "molecule_type", "ok", "mol", "atom_to_local", "local_xy",
                 "centroid3d", "radius", "label",
                 "image", "png", "img_w", "img_h", "atom_px", "centroid_px",
                 "center", "global_origin")

    def __init__(self, residue):
        self.residue = residue
        self.molecule_type = "other"
        self.ok = False
        self.mol = None
        self.atom_to_local = {}
        self.local_xy = {}
        self.centroid3d = None
        self.radius = 1.0
        self.label = "%s %s" % (residue.name, _res_number(residue))
        self.image = None        # matplotlib float RGBA array (headless renderer)
        self.png = None          # raw PNG bytes (Qt QPixmap, no re-render)
        self.img_w = 0
        self.img_h = 0
        self.atom_px = {}        # ChimeraX atom -> np.array([x, y]) within the unit image
        self.centroid_px = None
        self.center = None       # np.array([x, y]) in coord units (layout)
        self.global_origin = None  # np.array([x, y]) top-left of image in shifted px space


class _Edge:
    """One interaction, resolved to pixel-space endpoints on specific units.

    ``ringA``/``ringB`` (when set) hold an endpoint's aromatic-ring atoms; that
    end of the line terminates at the centroid of those atoms rather than at the
    single ``atomA``/``atomB`` (used as a fallback / obstacle base).
    """

    __slots__ = ("unitA", "atomA", "ringA", "unitB", "atomB", "ringB",
                 "interaction_type", "distance")

    def __init__(self, unitA, atomA, ringA, unitB, atomB, ringB,
                 interaction_type, distance):
        self.unitA = unitA
        self.atomA = atomA
        self.ringA = ringA
        self.unitB = unitB
        self.atomB = atomB
        self.ringB = ringB
        self.interaction_type = interaction_type
        self.distance = distance


class DiagramScene:
    """Renderer-agnostic geometry for a 2D interaction diagram.

    ``atom_pos`` maps every drawn ChimeraX atom to its pixel position; the Qt
    view rebuilds this map from item positions while dragging so edge routing
    stays live. ``extent`` is the canvas ``(width, height)`` in pixels.
    """

    __slots__ = ("units", "edges", "atom_pos", "atom_unit", "extent",
                 "summary", "show_distances")

    def __init__(self, units, edges, atom_pos, atom_unit, extent, summary,
                 show_distances):
        self.units = units
        self.edges = edges
        self.atom_pos = atom_pos        # atom -> np.array([x, y])
        self.atom_unit = atom_unit      # atom -> _Unit
        self.extent = extent            # (width, height) px
        self.summary = summary
        self.show_distances = show_distances


def build_scene(session, structure, rows, selected_residues, *,
                include_hidden=False, show_distances=True, style="default"):
    """Compute the diagram geometry once; return a :class:`DiagramScene`.

    Qt-free, so it backs both the headless matplotlib export and the
    interactive Qt view. ``style`` selects the RDKit structure rendering
    (see :data:`_STRUCTURE_STYLES`).
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    if not selected_residues:
        raise ValueError("Select a ligand or residue(s) first, then export.")

    scoped = _scope_rows(rows, selected_residues, include_hidden)
    if not scoped:
        raise ValueError("No interactions for the current selection.")

    # Units = residues that appear as endpoints; remember each one's molecule type.
    mtype = {}
    interactions_per_res = {}
    for row in scoped:
        for ep in (row.endpoint1, row.endpoint2):
            res = ep.atom.residue
            mtype.setdefault(res, ep.molecule_type)
            interactions_per_res[res] = interactions_per_res.get(res, 0) + 1

    order_lookup = _mol2_bond_orders(session, structure)
    units = []
    for res in mtype:
        u = _build_unit(res, order_lookup, np, Chem)
        u.molecule_type = mtype[res]
        units.append(u)
    by_residue = {u.residue: u for u in units}

    center = _choose_center(units, selected_residues, interactions_per_res)

    # Render before placing: layout keys residue angles off the centre's
    # rendered atom pixels, and per-unit rendering is placement-independent.
    for u in units:
        _render_unit(u, np, Chem, rdMolDraw2D, style=style)
    _place_units(units, center, scoped, np)

    global_xy, extent_px, _fallbacks = _compose_positions(units, np)

    atom_unit = {a: u for u in units for a in u.atom_to_local}
    edges = []
    for row in scoped:
        ra = _resolve_endpoint(row.endpoint1.atom, by_residue, global_xy)
        rb = _resolve_endpoint(row.endpoint2.atom, by_residue, global_xy)
        if ra is None or rb is None:
            continue
        ringA = _ring_in_scene(getattr(row.endpoint1, "ring_atoms", None), global_xy)
        ringB = _ring_in_scene(getattr(row.endpoint2, "ring_atoms", None), global_xy)
        edges.append(_Edge(ra[0], ra[1], ringA, rb[0], rb[1], ringB,
                           row.interaction_type, row.distance))

    summary = {
        "nodes": len(units),
        "edges": len(edges),
        "interactions": len(scoped),
        "types": sorted({e.interaction_type for e in edges}),
        "scoped": True,
        "fallback": sorted(u.label for u in units if not u.ok),
    }
    return DiagramScene(units, edges, global_xy, atom_unit, extent_px, summary,
                        show_distances)


def render_scene_matplotlib(scene, *, title=None):
    """Render a :class:`DiagramScene` to a matplotlib ``Figure`` (headless)."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    units = scene.units
    global_xy = scene.atom_pos
    width, height = scene.extent
    fig = Figure(figsize=(min(22, max(6, width / 150.0)), min(22, max(5, height / 150.0))))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    for u in units:
        if u.ok and u.image is not None:
            ox, oy = u.global_origin
            ax.imshow(u.image, extent=[ox, ox + u.img_w, oy + u.img_h, oy],
                      zorder=2, interpolation="bilinear")
        else:
            p = global_xy.get(next(iter(u.atom_to_local), None))
            if p is not None:
                ax.scatter([p[0]], [p[1]], s=850, marker="o", c=_FALLBACK_FACE,
                           edgecolors="#666666", zorder=2)
        cx, cy = _unit_label_anchor(u, global_xy, np)
        if cx is not None:
            ax.text(cx, cy, u.label, fontsize=8, color="#444444", ha="center",
                    va="bottom", zorder=5)

    types_drawn = set()
    label_edges = scene.show_distances and len(scene.edges) <= 60
    for e in scene.edges:
        pA = _endpoint_pos(e.atomA, e.ringA, global_xy, np)
        pB = _endpoint_pos(e.atomB, e.ringB, global_xy, np)
        if pA is None or pB is None:
            continue
        obs_pos, obs_w = _obstacles_for_edge(e, global_xy, scene.atom_unit, np)
        pts = _route_edge(pA, pB, obs_pos, obs_w, np)
        color = mpl_color(e.interaction_type)
        ax.plot(pts[:, 0], pts[:, 1], ls="--", lw=1.8, color=color,
                alpha=0.9, zorder=3)
        types_drawn.add(e.interaction_type)
        if label_edges and e.distance is not None:
            m = pts[len(pts) // 2]
            ax.text(m[0], m[1], "%.1f" % e.distance, fontsize=7, color=color,
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    handles = [Line2D([0], [0], color=mpl_color(t), ls="--", lw=2, label=t)
               for t in sorted(types_drawn)]
    if handles:
        ax.legend(handles=handles, title="Interaction", loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=9, frameon=False)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def build_diagram(session, structure, rows, selected_residues, *, title=None,
                  include_hidden=False, show_distances=True):
    """Build the diagram; return ``(matplotlib Figure, summary dict)``."""
    scene = build_scene(session, structure, rows, selected_residues,
                        include_hidden=include_hidden, show_distances=show_distances)
    fig = render_scene_matplotlib(scene, title=title)
    return fig, scene.summary


def export_interaction_diagram(session, structure, rows, selected_residues, path, *,
                               title=None, include_hidden=False, show_distances=True, dpi=300):
    fig, summary = build_diagram(session, structure, rows, selected_residues, title=title,
                                 include_hidden=include_hidden, show_distances=show_distances)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    summary["path"] = str(path)
    return summary


# ----- scoping ------------------------------------------------------------

def _scope_rows(rows, selected_residues, include_hidden):
    out = []
    for row in rows:
        pb = row.pseudobond
        if not include_hidden and pb is not None:
            if getattr(pb, "deleted", False) or not getattr(pb, "display", True):
                continue
        if (row.endpoint1.atom.residue not in selected_residues
                and row.endpoint2.atom.residue not in selected_residues):
            continue
        out.append(row)
    return out


# ----- chemistry ----------------------------------------------------------

def _mol2_bond_orders(session, structure):
    import os
    import tempfile
    from chimerax.core.commands import run

    handle = tempfile.NamedTemporaryFile(suffix=".mol2", delete=False)
    handle.close()
    path = handle.name
    bonds = {}
    try:
        run(session, 'save "%s" models #%s' % (path, structure.id_string))
        section = None
        id_to_key = {}
        with open(path) as fh:
            for line in fh:
                if line.startswith("@<TRIPOS>"):
                    section = line.strip()
                    continue
                parts = line.split()
                if not parts:
                    continue
                if section == "@<TRIPOS>ATOM" and len(parts) >= 5:
                    id_to_key[int(parts[0])] = (
                        round(float(parts[2]), 2), round(float(parts[3]), 2), round(float(parts[4]), 2))
                elif section == "@<TRIPOS>BOND" and len(parts) >= 4:
                    ka = id_to_key.get(int(parts[1]))
                    kb = id_to_key.get(int(parts[2]))
                    if ka and kb:
                        bonds[frozenset((ka, kb))] = parts[3]
    except Exception:
        pass
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return bonds


def _key(atom):
    c = atom.scene_coord
    return (round(float(c[0]), 2), round(float(c[1]), 2), round(float(c[2]), 2))


def _build_unit(residue, order_lookup, np, Chem):
    from rdkit.Chem import rdDepictor

    unit = _Unit(residue)
    heavy = [a for a in residue.atoms if a.element.name != "H"]
    if not heavy:
        return unit
    unit.centroid3d = np.mean(np.array([a.scene_coord for a in heavy], dtype=float), axis=0)

    idx = {a: i for i, a in enumerate(heavy)}
    bt_map = {"1": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE, "3": Chem.BondType.TRIPLE,
              "ar": Chem.BondType.AROMATIC, "am": Chem.BondType.SINGLE}
    rw = Chem.RWMol()
    for a in heavy:
        rw.AddAtom(Chem.Atom(int(a.element.number)))
    aromatic = set()
    order_str = {}
    done = set()
    for a in heavy:
        for b in a.bonds:
            o = b.other_atom(a)
            if o not in idx:
                continue
            pair = frozenset((idx[a], idx[o]))
            if pair in done:
                continue
            done.add(pair)
            ostr = order_lookup.get(frozenset((_key(a), _key(o))), "1")
            order_str[pair] = ostr
            bt = bt_map.get(ostr, Chem.BondType.SINGLE)
            rw.AddBond(idx[a], idx[o], bt)
            if bt == Chem.BondType.AROMATIC:
                aromatic.update((idx[a], idx[o]))

    mol = rw.GetMol()
    for i in aromatic:
        mol.GetAtomWithIdx(i).SetIsAromatic(True)
    _assign_formal_charges(residue, heavy, idx, order_str, aromatic, mol)

    flags = (Chem.SanitizeFlags.SANITIZE_ALL
             ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
             ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    try:
        Chem.SanitizeMol(mol, sanitizeOps=flags)
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
    except Exception:
        return _fallback_unit(unit, heavy, np)

    unit.mol = mol
    unit.atom_to_local = idx
    unit.local_xy = {idx[a]: np.array([conf.GetAtomPosition(idx[a]).x,
                                       conf.GetAtomPosition(idx[a]).y]) for a in heavy}
    centroid = np.mean(np.array(list(unit.local_xy.values())), axis=0)
    for k in unit.local_xy:
        unit.local_xy[k] = unit.local_xy[k] - centroid
    unit.radius = max((float(np.linalg.norm(p)) for p in unit.local_xy.values()), default=0.8) or 0.8
    unit.ok = True
    return unit


def _assign_formal_charges(residue, heavy, idx, order_str, aromatic, mol):
    by_name = {a.name: a for a in heavy}
    if "OXT" in by_name:
        mol.GetAtomWithIdx(idx[by_name["OXT"]]).SetFormalCharge(-1)
    if residue.name.upper() in _STANDARD_RESIDUES:
        for atom_name, charge in _STD_RESIDUE_CHARGES.get(residue.name.upper(), ()):
            a = by_name.get(atom_name)
            if a is not None:
                mol.GetAtomWithIdx(idx[a]).SetFormalCharge(charge)
        return

    def order_to(a, o):
        return _ORDER_VALUE.get(order_str.get(frozenset((idx[a], idx[o])), "1"), 1.0)

    def heavy_neighbors(a):
        return [b.other_atom(a) for b in a.bonds if b.other_atom(a) in idx]

    def h_count(a):
        return sum(1 for nb in a.neighbors if nb.element.name == "H")

    charged = set()
    for a in heavy:
        if a.element.name != "C" or idx[a] in aromatic:
            continue
        ns = [nb for nb in heavy_neighbors(a) if nb.element.name == "N"]
        others = [nb for nb in heavy_neighbors(a) if nb.element.name != "N"]
        if len(ns) == 3 and len(others) <= 1:
            doubles = [nb for nb in ns if order_to(a, nb) >= 2.0]
            target = doubles[0] if doubles else max(ns, key=h_count)
            mol.GetAtomWithIdx(idx[target]).SetFormalCharge(1)
            for nb in ns:
                charged.add(idx[nb])

    for a in heavy:
        if a.element.name != "O" or idx[a] in aromatic or idx[a] in charged:
            continue
        nbrs = heavy_neighbors(a)
        if len(nbrs) == 1 and order_to(a, nbrs[0]) == 1.0:
            c = nbrs[0]
            if any(o2 is not a and o2.element.name == "O" and order_to(c, o2) >= 2.0
                   for o2 in heavy_neighbors(c)):
                mol.GetAtomWithIdx(idx[a]).SetFormalCharge(-1)
                charged.add(idx[a])

    for a in heavy:
        if a.element.name != "N" or idx[a] in aromatic or idx[a] in charged:
            continue
        nbrs = heavy_neighbors(a)
        if len(nbrs) != 1 or nbrs[0].element.name != "C" or order_to(a, nbrs[0]) != 1.0:
            continue
        c = nbrs[0]
        is_amide = any(o2.element.name == "O" and order_to(c, o2) >= 2.0
                       for o2 in heavy_neighbors(c))
        if is_amide or sum(1 for x in heavy_neighbors(c) if x.element.name == "N") >= 3:
            continue
        mol.GetAtomWithIdx(idx[a]).SetFormalCharge(1)


def _fallback_unit(unit, heavy, np):
    unit.atom_to_local = {a: 0 for a in heavy}
    unit.local_xy = {0: np.array([0.0, 0.0])}
    unit.radius = 0.9
    unit.ok = False
    return unit


# ----- layout -------------------------------------------------------------

def _choose_center(units, selected_residues, interactions_per_res):
    ligands = [u for u in units if u.molecule_type == "ligand" and u.residue in selected_residues]
    if ligands:
        return max(ligands, key=lambda u: len(u.atom_to_local))
    if len(units) == 1:
        return units[0]
    return max(units, key=lambda u: interactions_per_res.get(u.residue, 0))


def _place_units(units, center, rows, np):
    if not units:
        return
    cents = np.array([u.centroid3d for u in units], dtype=float)
    base2d = _pca_2d(cents, np)
    ci = units.index(center)
    base2d = base2d - base2d[ci]

    center.center = np.array([0.0, 0.0])
    others = [k for k in range(len(units)) if k != ci]
    pos = {ci: np.array([0.0, 0.0])}
    for k in others:
        # Prefer the direction of the centre atom this unit actually bonds to,
        # so its interaction line leaves the ligand beside its real atom; fall
        # back to the 3D PCA direction for units that don't touch the centre.
        v = _preferred_dir(units[k], center, rows, np)
        if v is None:
            v = base2d[k]
        n = float(np.linalg.norm(v))
        direction = v / n if n > 1e-6 else np.array([np.cos(k), np.sin(k)])
        r = center.radius + units[k].radius + _GAP
        pos[k] = direction * r

    # Repulsion to separate units; centre pinned at origin.
    radii = {k: units[k].radius for k in range(len(units))}
    for _ in range(200):
        moved = False
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                d = pos[b] - pos[a]
                dist = float(np.linalg.norm(d))
                need = radii[a] + radii[b] + _GAP
                if dist < need:
                    if dist < 1e-6:
                        d = np.array([np.cos(a + b), np.sin(a + b)]); dist = 1.0
                    step = d / dist * (need - dist) / 2.0
                    if a == ci:
                        pos[b] += 2 * step
                    elif b == ci:
                        pos[a] -= 2 * step
                    else:
                        pos[a] -= step; pos[b] += step
                    moved = True
        pos[ci] = np.array([0.0, 0.0])
        if not moved:
            break

    for k, u in enumerate(units):
        u.center = pos[k]


def _pca_2d(points, np):
    centered = points - points.mean(axis=0)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T
    except np.linalg.LinAlgError:
        return centered[:, :2]


def _preferred_dir(unit, center, rows, np):
    """Layout direction (y-up coord units) from the centre toward the centre
    atom(s) ``unit`` bonds to, or ``None`` if it doesn't touch the centre.

    Uses the centre's rendered atom pixels (y-down), flipped to the y-up layout
    convention used for ``unit.center``.
    """
    if center.centroid_px is None or not center.atom_px:
        return None
    offsets = []
    for row in rows:
        a1, a2 = row.endpoint1.atom, row.endpoint2.atom
        if a1.residue is center.residue and a2.residue is unit.residue:
            ca = a1
        elif a2.residue is center.residue and a1.residue is unit.residue:
            ca = a2
        else:
            continue
        p = center.atom_px.get(ca)
        if p is None:
            p = next((center.atom_px[nb] for nb in ca.neighbors if nb in center.atom_px), None)
        if p is not None:
            offsets.append(p - center.centroid_px)
    if not offsets:
        return None
    d = np.mean(np.array(offsets), axis=0)
    return np.array([d[0], -d[1]])


# ----- per-unit rendering + compositing -----------------------------------

# Chemical-structure styles the user can pick in the diagram (see
# _apply_draw_style); RDKit MolDrawOptions verified available in 2025.09.x.
_STRUCTURE_STYLES = ("default", "bw", "bold", "comic")


def _apply_draw_style(opts, style):
    """Configure RDKit ``MolDrawOptions`` for one of :data:`_STRUCTURE_STYLES`."""
    opts.fixedBondLength = float(_FIXED_BOND_PX)   # uniform scale across units
    opts.setBackgroundColour((1, 1, 1, 0))         # transparent
    if style == "bw":
        opts.useBWAtomPalette()
    elif style == "bold":
        opts.bondLineWidth = 3
        opts.minFontSize = 16
        opts.maxFontSize = 22
    elif style == "comic":
        opts.comicMode = True


def restyle_unit(unit, style):
    """Re-render one unit's structure image in a new style, in place."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    _render_unit(unit, np, Chem, rdMolDraw2D, style=style)


def _render_unit(unit, np, Chem, rdMolDraw2D, style="default"):
    import io
    import matplotlib.image as mpimg

    if not unit.ok or unit.mol is None:
        unit.atom_px = {a: np.array([0.0, 0.0]) for a in unit.atom_to_local}
        unit.centroid_px = np.array([0.0, 0.0])
        unit.img_w = unit.img_h = 0
        return

    xs = [p[0] for p in unit.local_xy.values()]
    ys = [p[1] for p in unit.local_xy.values()]
    bw = (max(xs) - min(xs)) or 1.0
    bh = (max(ys) - min(ys)) or 1.0
    pad = int(_FIXED_BOND_PX)
    w = max(60, int(bw * _PPU) + 2 * pad)
    h = max(60, int(bh * _PPU) + 2 * pad)

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    _apply_draw_style(drawer.drawOptions(), style)
    try:
        prepared = rdMolDraw2D.PrepareMolForDrawing(unit.mol, kekulize=False)
    except Exception:
        prepared = unit.mol
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    unit.png = drawer.GetDrawingText()
    unit.image = mpimg.imread(io.BytesIO(unit.png), format="png")
    unit.img_w, unit.img_h = w, h
    unit.atom_px = {}
    for a, li in unit.atom_to_local.items():
        p = drawer.GetDrawCoords(li)
        unit.atom_px[a] = np.array([float(p.x), float(p.y)])
    unit.centroid_px = np.mean(np.array(list(unit.atom_px.values())), axis=0)


def _compose_positions(units, np):
    """Place each unit's image in a shared pixel canvas; return (global_xy, (W,H), fallbacks)."""
    # target pixel per unit (y flipped so layout-up reads up); origin shifted later.
    targets = {}
    for u in units:
        targets[id(u)] = np.array([u.center[0] * _PPU, -u.center[1] * _PPU])

    raw_origin = {}
    corners = []
    for u in units:
        t = targets[id(u)]
        if u.ok and u.img_w:
            origin = t - u.centroid_px
            raw_origin[id(u)] = origin
            corners.append(origin)
            corners.append(origin + np.array([u.img_w, u.img_h]))
        else:
            raw_origin[id(u)] = t
            corners.append(t - 30)
            corners.append(t + 30)

    corners = np.array(corners)
    margin = 40.0
    minxy = corners.min(axis=0) - margin
    shift = -minxy
    maxxy = corners.max(axis=0) + margin
    width = float(maxxy[0] - minxy[0])
    height = float(maxxy[1] - minxy[1])

    global_xy = {}
    for u in units:
        origin = raw_origin[id(u)] + shift
        u.global_origin = origin
        if u.ok and u.img_w:
            for a, px in u.atom_px.items():
                global_xy[a] = origin + px
        else:
            for a in u.atom_to_local:
                global_xy[a] = (targets[id(u)] + shift)

    fallbacks = [u for u in units if not u.ok]
    return global_xy, (width, height), fallbacks


def _resolve_endpoint(atom, by_residue, global_xy):
    """Map an interaction atom to ``(unit, drawn_atom)`` in pixel space.

    Falls back to a bonded neighbour when the exact atom wasn't drawn (e.g. a
    hydrogen), so the line still starts inside the right structure.
    """
    if atom in global_xy:
        return (by_residue.get(atom.residue), atom)
    unit = by_residue.get(atom.residue)
    if unit is None:
        return None
    for nb in atom.neighbors:
        if nb in global_xy:
            return (by_residue.get(nb.residue, unit), nb)
    return None


def _ring_in_scene(ring_atoms, global_xy):
    """The drawn subset of an endpoint's ring atoms, or None."""
    if not ring_atoms:
        return None
    present = [a for a in ring_atoms if a in global_xy]
    return present if len(present) >= 3 else None


def _endpoint_pos(atom, ring, pos_lookup, np):
    """Pixel position of an edge endpoint: the ring centroid if ``ring`` is set,
    else the single atom's position."""
    if ring:
        pts = [pos_lookup[a] for a in ring if a in pos_lookup]
        if pts:
            return np.mean(np.array(pts, dtype=float), axis=0)
    return pos_lookup.get(atom)


# ----- edge routing (shared by both renderers) ----------------------------

_ROUTE_RADIUS = 15.0       # px; an obstacle nearer than this counts as crossed
_ROUTE_SAMPLES = 24
# Perpendicular bow as a fraction of chord length; 0.0 is the straight segment.
_ROUTE_FRACS = (0.0, 0.16, -0.16, 0.32, -0.32, 0.5, -0.5)


def _obstacles_for_edge(edge, atom_pos, atom_unit, np):
    """Atom positions/weights to avoid for one edge.

    The endpoint atoms (and any ring atoms a π endpoint terminates inside) are
    dropped; atoms of the two endpoint units are down-weighted (a line must pass
    through its own structure to reach its atom/centre), so routing chiefly
    avoids *other* structures.
    """
    ep_units = (id(edge.unitA), id(edge.unitB))
    skip = {edge.atomA, edge.atomB}
    if edge.ringA:
        skip.update(edge.ringA)
    if edge.ringB:
        skip.update(edge.ringB)
    pos = []
    weights = []
    for atom, p in atom_pos.items():
        if atom in skip:
            continue
        u = atom_unit.get(atom)
        pos.append(p)
        weights.append(0.18 if (u is not None and id(u) in ep_units) else 1.0)
    if not pos:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.asarray(pos, dtype=float), np.asarray(weights, dtype=float)


def _route_edge(pA, pB, obstacle_pos, obstacle_w, np):
    """Return an Mx2 polyline from ``pA`` to ``pB`` crossing the fewest atoms.

    Tries the straight segment and quadratic Béziers bowed to either side,
    scoring each by weighted atom crossings; a small bias keeps straighter
    paths when scores tie. Endpoints stay exactly on the interacting atoms.
    """
    pA = np.asarray(pA, dtype=float)
    pB = np.asarray(pB, dtype=float)
    chord = pB - pA
    length = float(np.linalg.norm(chord))
    if length < 1e-6 or len(obstacle_pos) == 0:
        return np.array([pA, pB])
    perp = np.array([-chord[1], chord[0]]) / length
    mid = (pA + pB) / 2.0
    t = np.linspace(0.0, 1.0, _ROUTE_SAMPLES)
    one = 1.0 - t
    best_pts = None
    best_score = None
    for frac in _ROUTE_FRACS:
        ctrl = mid + perp * (frac * length)
        pts = ((one * one)[:, None] * pA
               + (2.0 * one * t)[:, None] * ctrl
               + (t * t)[:, None] * pB)
        score = _crossing_score(pts, obstacle_pos, obstacle_w, np) + abs(frac) * 0.4
        if best_score is None or score < best_score - 1e-9:
            best_score = score
            best_pts = pts
    return best_pts


def _crossing_score(pts, obstacle_pos, obstacle_w, np):
    """Weighted count of obstacles within ``_ROUTE_RADIUS`` of the polyline."""
    seg_a = pts[:-1]
    seg_b = pts[1:]
    dmin = _point_segment_dists(obstacle_pos, seg_a, seg_b, np).min(axis=1)
    hit = dmin < _ROUTE_RADIUS
    if not hit.any():
        return 0.0
    return float(np.sum(obstacle_w[hit] * (1.0 - dmin[hit] / _ROUTE_RADIUS)))


def _point_segment_dists(pts, seg_a, seg_b, np):
    """Distances from each point (Nx2) to each segment (a,b: Sx2) -> NxS."""
    ab = seg_b - seg_a                                   # S x 2
    ab_len2 = np.einsum("sj,sj->s", ab, ab)
    ab_len2 = np.where(ab_len2 < 1e-9, 1e-9, ab_len2)
    ap = pts[:, None, :] - seg_a[None, :, :]             # N x S x 2
    proj = np.clip(np.einsum("nsj,sj->ns", ap, ab) / ab_len2, 0.0, 1.0)
    closest = seg_a[None, :, :] + proj[:, :, None] * ab[None, :, :]
    diff = pts[:, None, :] - closest
    return np.sqrt(np.einsum("nsj,nsj->ns", diff, diff))


def _unit_label_anchor(unit, global_xy, np):
    pts = [global_xy[a] for a in unit.atom_to_local if a in global_xy]
    if not pts:
        return (None, None)
    arr = np.array(pts, dtype=float)
    return (float(arr[:, 0].mean()), float(arr[:, 1].min()) - 10.0)


def _res_number(residue):
    icode = (residue.insertion_code or "").strip()
    return "%d%s" % (residue.number, icode)

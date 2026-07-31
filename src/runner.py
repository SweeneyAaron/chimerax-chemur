"""Bridge between an open ChimeraX structure and the Chemur engine.

Shared by the command (cmd.py) and the GUI tool (tool.py):

  open structure -> (optional addh) -> temp mmCIF -> chemur.analyze()
                 -> map Chemur atoms back to ChimeraX atoms
                 -> (optional) filter by selection / biopolymer-internal
                 -> draw interactions as pseudobonds
                 -> return rows for display
"""

from __future__ import annotations

import os
import tempfile

from .colors import style_for

PBG_NAME = "Chemur interactions"
_PI_MARKERSET_NAME = "Chemur π-centres"
_PI_MARKER_RADIUS = 0.4   # Å; small dot at each ring centre
# ChimeraX's create_marker makes marker atoms element 'H', so a user 'delete H'
# would remove every ring-centre marker (and thus every π pseudobond). Re-element
# the markers to a non-hydrogen placeholder so they survive 'delete H'. He is
# inert, absent from biomolecules, and never a 'delete' target.
_PI_MARKER_ELEMENT = "He"

# Metadata keys (besides distance/angle/offset) that carry geometric detail.
_GEOMETRY_METADATA_KEYS = ("donor_angle", "point_plane_angle", "geometry", "occluded")


class Endpoint:
    """One partner of an interaction: a ChimeraX atom plus grouping info.

    ``ring_atoms`` (when set) holds the aromatic-ring member atoms for a π
    endpoint; the interaction line then terminates at the ring centroid in both
    the 3D view and the 2D diagram. ``atom`` stays a representative ring atom
    (nearest the centroid) for selection/labels/grouping.
    """

    __slots__ = ("atom", "molecule_type", "chain_id", "residue_id", "residue_name",
                 "ring_atoms")

    def __init__(self, atom, molecule_type, chain_id, residue_id, residue_name,
                 ring_atoms=None):
        self.atom = atom
        self.molecule_type = molecule_type
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.residue_name = residue_name
        self.ring_atoms = ring_atoms

    @property
    def residue_key(self):
        return (self.molecule_type, self.chain_id, self.residue_id, self.residue_name)

    @property
    def residue_label(self):
        return "%s %s" % (self.residue_name, self.residue_id)

    @property
    def atom_label(self):
        return "/%s %s %s %s" % (self.chain_id, self.residue_name, self.residue_id, self.atom.name)


class InteractionRow:
    """One resolved interaction, ready for the results tree."""

    __slots__ = ("interaction_type", "endpoint1", "endpoint2", "distance",
                 "angle", "offset", "extra", "pseudobond")

    def __init__(self, interaction_type, endpoint1, endpoint2, distance,
                 angle, offset, extra, pseudobond):
        self.interaction_type = interaction_type
        self.endpoint1 = endpoint1
        self.endpoint2 = endpoint2
        self.distance = distance
        self.angle = angle
        self.offset = offset
        self.extra = extra  # dict of other geometry (donor_angle, geometry, ...)
        self.pseudobond = pseudobond

    def geometry_tooltip(self):
        parts = []
        if self.distance is not None:
            parts.append("distance: %.2f Å" % self.distance)
        if self.angle is not None:
            parts.append("angle: %.1f°" % self.angle)
        if self.offset is not None:
            parts.append("offset: %.2f Å" % self.offset)
        for key, value in self.extra.items():
            parts.append("%s: %s" % (key, _format_value(value)))
        return "\n".join(parts)


class LigandProtonation:
    """Per-residue protonation/templating status for one non-standard residue.

    ``status`` is "ok" when the ligand's SMILES template mapped onto the
    structure (so its protonation state was assigned) or "failed" when it could
    not be parsed/mapped (so the residue contributes no interactions and needs a
    manual SMILES). ``smiles_used`` is the SMILES that actually set the chemistry
    (it encodes the protonation state); ``net_charge`` is the sum of the
    component's per-atom formal charges.
    """

    __slots__ = ("component_id", "name", "chain_id", "residue_id",
                 "status", "smiles_used", "input_smiles", "net_charge", "reason")

    def __init__(self, component_id, name, chain_id, residue_id, status,
                 smiles_used, input_smiles, net_charge, reason):
        self.component_id = component_id
        self.name = name
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.status = status
        self.smiles_used = smiles_used
        self.input_smiles = input_smiles
        self.net_charge = net_charge
        self.reason = reason

    @property
    def residue_label(self):
        chain = self.chain_id or "_"
        return "%s /%s %s" % (self.name, chain, self.residue_id)


class InteractionRun:
    """Result of :func:`run_interactions`: drawable rows plus a ligand report."""

    __slots__ = ("rows", "ligand_report")

    def __init__(self, rows, ligand_report):
        self.rows = rows
        self.ligand_report = ligand_report


class PoseInteractionRun:
    """Result of :func:`run_pose_interactions` for one docked pose.

    ``rows`` are :class:`InteractionRow`s mapped back to live ChimeraX atoms (the
    pseudobond is ``None`` unless drawn). ``result`` is the engine
    :class:`chemur.models.AnalysisResult`, kept so a caller can read interacting
    residues straight from the engine output when atom mapping fails.
    ``fully_mapped`` is ``False`` when the pose ligand's atoms could not be
    resolved to ChimeraX atoms (so ``rows`` is empty and the engine-residue
    fallback applies).
    """

    __slots__ = ("rows", "result", "fully_mapped")

    def __init__(self, rows, result, fully_mapped):
        self.rows = rows
        self.result = result
        self.fully_mapped = fully_mapped


class TrajectoryRun:
    """Result of :func:`run_trajectory`.

    ``result`` is the engine :class:`chemur.trajectory.TrajectoryResult`.
    ``key_to_atoms`` maps each interaction key (``result.interaction_key(...)``)
    to the ChimeraX :class:`Atom` objects involved, so the GUI can select them in
    3D. ``key_to_endpoints`` maps each key to its endpoint atom groups (one inner
    list per interacting component, normally two) so the GUI can test selection
    scope (≥1 end vs both ends selected). ``frame_coordset_ids`` are the ChimeraX
    coordset ids aligned to ``result.frame_indices`` (for scrubbing the 3D view to
    an analysed frame). ``rec_by_id``/``features_by_id``/``id_to_atom`` are the maps
    needed to redraw a single frame's interactions as pseudobonds (see
    :func:`draw_frame_interactions`).
    """

    __slots__ = ("result", "key_to_atoms", "key_to_endpoints", "structure",
                 "frame_coordset_ids", "rec_by_id", "features_by_id", "id_to_atom")

    def __init__(self, result, key_to_atoms, key_to_endpoints, structure,
                 frame_coordset_ids, rec_by_id, features_by_id, id_to_atom):
        self.result = result
        self.key_to_atoms = key_to_atoms
        self.key_to_endpoints = key_to_endpoints
        self.structure = structure
        self.frame_coordset_ids = frame_coordset_ids
        self.rec_by_id = rec_by_id
        self.features_by_id = features_by_id
        self.id_to_atom = id_to_atom


def run_interactions(session, structure, *, protonate=False, protonate_ph=7.4,
                     profile="default", ligand_smiles=None, add_hydrogens=True,
                     rule_overrides=None, selected_only=False,
                     skip_biopolymer_internal=False):
    """Run Chemur on ``structure``, draw interactions, and return rows.

    ``selected_only`` keeps only interactions with at least one selected partner.
    ``skip_biopolymer_internal`` drops protein-protein and nucleic-nucleic contacts.
    ``rule_overrides`` is passed straight to :func:`chemur.analyze`.

    Returns an :class:`InteractionRun` (``.rows`` for display, ``.ligand_report``
    with per-non-standard-residue protonation status).
    """
    from chimerax.core.commands import run
    from chimerax.core.errors import UserError

    analyze = _import_analyze()

    if selected_only and not any(atom.selected for atom in structure.atoms):
        raise UserError("Select some atoms first, or turn off 'selected atoms only'.")

    if add_hydrogens:
        # Chemur needs explicit donor hydrogens to assign H-bond-like interactions.
        run(session, "addh #%s" % structure.id_string)

    # Drop any ring-centre markers left by a previous run *before* saving: the
    # MarkerSet is a child model of the structure, so 'save models #<id>' would
    # otherwise emit it as a second coordinate block and gemmi rejects the file.
    _clear_pi_markers(structure)

    handle = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    handle.close()
    path = handle.name
    try:
        run(session, 'save "%s" models #%s format mmcif' % (path, structure.id_string))
        result = analyze(
            path,
            protonate=protonate,
            protonate_ph_min=protonate_ph,
            protonate_ph_max=protonate_ph,
            profile=profile,
            ligand_smiles=ligand_smiles or None,
            rule_overrides=rule_overrides or None,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    ligand_report = _build_ligand_report(result)

    skipped = result.metadata.get("skipped_ligands") or {}
    if skipped:
        session.logger.warning(
            "Chemur skipped %d ligand(s) it could not template: %s"
            % (len(skipped), ", ".join(sorted(skipped)))
        )

    rec_by_id = {rec.atom_id: rec for rec in result.atoms}
    features_by_id = {f.feature_id: f for f in result.features}
    coord_index, name_index = _chimerax_atom_lookup(structure)
    id_to_atom = {}
    for rec in result.atoms:
        atom = _resolve_atom(rec, coord_index, name_index)
        if atom is not None:
            id_to_atom[rec.atom_id] = atom

    rows = _draw_interactions(
        session, structure, result.interactions,
        rec_by_id, id_to_atom, features_by_id,
        selected_only=selected_only,
        skip_biopolymer_internal=skip_biopolymer_internal,
    )
    return InteractionRun(rows, ligand_report)


def _import_analyze():
    """Import ``chemur.analyze`` or raise a UserError explaining how to install."""
    from chimerax.core.errors import UserError
    try:
        from chemur import analyze
        return analyze
    except ImportError as e:
        import importlib.util
        if importlib.util.find_spec("chemur") is None:
            # Engine genuinely absent from ChimeraX's environment. Install it via
            # ChimeraX's own pip (not '<ChimeraX>/bin/python -m pip', which can land
            # in your user site-packages that ChimeraX does not load).
            raise UserError(
                "The 'chemur' engine is not installed in ChimeraX's Python.\n"
                "It is a declared dependency of this bundle, so normally\n"
                "    toolshed install ChimeraX-Chemur\n"
                "pulls it in. To install it on its own, run:\n"
                "    pip install chemur\n"
                "Verify it landed in ChimeraX with:  pip show chemur"
            ) from e
        # Engine is present but failed to import (e.g. a broken build or a missing
        # dependency). Surface the real error instead of mislabelling it.
        # ChimeraX's `pip` command takes a single requirement and has no
        # --force-reinstall, so a refresh is uninstall-then-install. The restart is
        # load-bearing: sys.modules already holds the broken partial import.
        raise UserError(
            "The 'chemur' engine is installed but failed to import: %s\n"
            "Reinstall it into ChimeraX with:\n"
            "    pip uninstall chemur\n"
            "    pip install chemur\n"
            "then restart ChimeraX."
            % e
        ) from e


def _has_protein(structure):
    """True if ``structure`` contains any amino-acid polymer residue."""
    from chimerax.atomic import Residue
    return any(r.polymer_type == Residue.PT_AMINO for r in structure.residues)


def _addh_and_clear(session, structures, add_hydrogens):
    """Optionally add hydrogens, then clear Chemur π-centre markers before a save.

    The ring-centre MarkerSet is a child model; left in place 'save models' emits a
    second coordinate block that gemmi rejects (see run_interactions).
    """
    from chimerax.core.commands import run
    if add_hydrogens:
        for s in structures:
            run(session, "addh #%s" % s.id_string)
    for s in structures:
        _clear_pi_markers(s)


def save_models_mmcif(session, structures):
    """Save ``structures`` to one temp mmCIF; return its path. Caller deletes it.

    The model spec writes the ``#`` once and comma-joins the ids (``#1,3``); a
    repeated ``#`` (``#1,#3``) is rejected by ChimeraX's spec parser.
    """
    from chimerax.core.commands import run
    handle = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    handle.close()
    path = handle.name
    spec = "#" + ",".join(s.id_string for s in structures)
    run(session, 'save "%s" models %s format mmcif' % (path, spec))
    return path


def run_pose_interactions(session, pose, receptor=None, *, protonate=False,
                          protonate_ph=7.4, profile="default", ligand_smiles=None,
                          add_hydrogens=True, rule_overrides=None,
                          skip_biopolymer_internal=True, draw=False,
                          selected_only=False, receptor_path=None):
    """Analyse one docked ligand pose's interactions with the receptor.

    Routing (auto-detected):

    * **Full receptor+ligand complex** (``pose`` already contains protein): analyse
      the pose on its own from a temp mmCIF.
    * **Ligand-only pose loaded from an SDF** (carries ``Bond.order``): re-emit the
      ligand to a temp SDF — preserving element/coords/bond-order chemistry, which a
      ChimeraX-written mmCIF would lose — and analyse it against the receptor via the
      engine's native ``ligand_sdf`` path. ``receptor_path`` (a pre-saved receptor
      mmCIF) is reused when given, else the receptor is saved here.
    * **Ligand-only pose without bond orders** (not from SDF): merge with the receptor
      in one mmCIF and rely on CCD/user SMILES for chemistry.

    Engine atoms are mapped back to the real ChimeraX atoms (by coordinate, name as
    fallback) across the involved models, so rows can drive 3D selection. Returns a
    :class:`PoseInteractionRun`. ``draw`` is reserved for pseudobond drawing; the
    comparison figure uses ``draw=False``. ``selected_only`` is accepted for signature
    compatibility but ignored.
    """
    from .sdf_io import has_bond_orders, write_ligand_sdf

    analyze = _import_analyze()

    pose_has_protein = _has_protein(pose)
    use_sdf = (not pose_has_protein) and (receptor is not None) and has_bond_orders(pose)

    common = dict(protonate=protonate, protonate_ph_min=protonate_ph,
                  protonate_ph_max=protonate_ph, profile=profile,
                  rule_overrides=rule_overrides or None)
    temp_paths = []
    try:
        if pose_has_protein:
            _addh_and_clear(session, [pose], add_hydrogens)
            path = save_models_mmcif(session, [pose])
            temp_paths.append(path)
            structures = [pose]
            result = analyze(path, ligand_smiles=ligand_smiles or None, **common)

        elif use_sdf:
            # Do NOT run ChimeraX 'addh' on the ligand: it can break the SDF's own
            # valence so RDKit can't sanitise it (the cause of "Could not parse
            # molecule"). Clear stray π-markers only; RDKit adds valence-consistent
            # hydrogens when writing the SDF instead.
            _clear_pi_markers(pose)
            sdf_handle = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
            sdf_handle.close()
            temp_paths.append(sdf_handle.name)
            write_ligand_sdf(pose, sdf_handle.name, name=_ligand_name(pose),
                             add_hydrogens=add_hydrogens)
            rpath = receptor_path
            if rpath is None:
                _addh_and_clear(session, [receptor], add_hydrogens)
                rpath = save_models_mmcif(session, [receptor])
                temp_paths.append(rpath)
            structures = [receptor, pose]
            result = analyze(rpath, ligand_sdf=[sdf_handle.name], **common)

        else:
            structures = [receptor, pose] if receptor is not None else [pose]
            _addh_and_clear(session, structures, add_hydrogens)
            path = save_models_mmcif(session, structures)
            temp_paths.append(path)
            result = analyze(path, ligand_smiles=ligand_smiles or None, **common)
    finally:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    rec_by_id = {rec.atom_id: rec for rec in result.atoms}
    features_by_id = {f.feature_id: f for f in result.features}
    coord_index, name_index = _chimerax_atom_lookup_multi(structures)
    id_to_atom = {}
    for rec in result.atoms:
        atom = _resolve_atom(rec, coord_index, name_index)
        if atom is not None:
            id_to_atom[rec.atom_id] = atom

    ligand_atom_ids = [rec.atom_id for rec in result.atoms
                       if rec.molecule_type == "ligand"]
    ligand_resolved = sum(1 for aid in ligand_atom_ids if aid in id_to_atom)
    fully_mapped = bool(ligand_atom_ids) and ligand_resolved > 0

    if draw:
        draw_on = receptor if (len(structures) == 2 and receptor is not None) else pose
        rows = _draw_interactions(
            session, draw_on, result.interactions,
            rec_by_id, id_to_atom, features_by_id,
            skip_biopolymer_internal=skip_biopolymer_internal,
        )
    else:
        rows = _resolve_rows(
            result.interactions, rec_by_id, id_to_atom, features_by_id,
            skip_biopolymer_internal=skip_biopolymer_internal,
        )
    return PoseInteractionRun(rows, result, fully_mapped)


def _ligand_name(pose):
    """A short, file-safe ligand name for the temp SDF (cosmetic)."""
    base = "".join(c for c in (pose.name or "LIG") if c.isalnum()) or "LIG"
    return base[:16].upper()


def _resolve_rows(interactions, rec_by_id, id_to_atom, features_by_id, *,
                  skip_biopolymer_internal=False, included_types=None):
    """Map ``interactions`` to :class:`InteractionRow`s WITHOUT drawing pseudobonds.

    The non-drawing counterpart of :func:`_draw_interactions`, used by
    :func:`run_pose_interactions` for the comparison figure. Rows carry endpoints
    (with their ChimeraX atoms) but ``pseudobond=None``.
    """
    rows = []
    for inter in interactions:
        if included_types is not None and inter.interaction_type not in included_types:
            continue
        endpoints = _choose_endpoints(inter, rec_by_id, id_to_atom, features_by_id)
        if endpoints is None:
            continue
        ep1, ep2 = endpoints
        if skip_biopolymer_internal and _is_biopolymer_internal(ep1, ep2):
            continue
        rows.append(InteractionRow(
            inter.interaction_type, ep1, ep2,
            inter.distance, inter.angle, inter.offset,
            _geometry_extra(inter), None,
        ))
    return rows


def _draw_interactions(session, structure, interactions, rec_by_id, id_to_atom,
                       features_by_id, *, selected_only=False,
                       skip_biopolymer_internal=False, included_types=None):
    """Map ``interactions`` to ChimeraX atoms, (re)draw the pseudobonds, return rows.

    Clears the structure's Chemur pseudobond groups + π markers first, so calling
    this repeatedly (e.g. per trajectory frame) replaces the previous drawing.
    ``included_types`` (optional set) restricts which interaction types are drawn.
    Shared by :func:`run_interactions` and :func:`draw_frame_interactions`.
    """
    pbg = structure.pseudobond_group(PBG_NAME)
    if pbg.num_pseudobonds:
        pbg.pseudobonds.delete()
    pi_pbg = session.pb_manager.get_group(_pi_group_name(structure), create=True)
    if pi_pbg.num_pseudobonds:
        pi_pbg.pseudobonds.delete()
    _clear_pi_markers(structure)
    marker_holder = {}  # lazily-created MarkerSet + per-ring marker cache

    rows = []
    counts = {}
    unresolved = 0
    filtered = 0
    for inter in interactions:
        if included_types is not None and inter.interaction_type not in included_types:
            filtered += 1
            continue
        endpoints = _choose_endpoints(inter, rec_by_id, id_to_atom, features_by_id)
        if endpoints is None:
            unresolved += 1
            continue
        ep1, ep2 = endpoints

        if selected_only and not (ep1.atom.selected or ep2.atom.selected):
            filtered += 1
            continue
        if skip_biopolymer_internal and _is_biopolymer_internal(ep1, ep2):
            filtered += 1
            continue

        style = style_for(inter.interaction_type)
        pb = _draw_interaction(session, structure, ep1, ep2, style, pbg, pi_pbg, marker_holder)
        if pb is None:
            # Duplicate pair within the same group, or otherwise invalid.
            unresolved += 1
            continue
        rows.append(InteractionRow(
            inter.interaction_type, ep1, ep2,
            inter.distance, inter.angle, inter.offset,
            _geometry_extra(inter), pb,
        ))
        counts[inter.interaction_type] = counts.get(inter.interaction_type, 0) + 1

    # A global (cross-model) pseudobond group only draws once it's a session
    # model; register it if it ended up with π pseudobonds, else drop it.
    if pi_pbg.num_pseudobonds:
        if pi_pbg.id is None:
            session.models.add([pi_pbg])
    elif pi_pbg.id is None:
        pi_pbg.delete()
    else:
        session.models.close([pi_pbg])

    _log_summary(session, structure, counts, len(rows), unresolved, filtered)
    return rows


def run_trajectory(session, structure, *, exclude_solvent=False, exclude_ions=False,
                   frame_start=0, frame_stop=None, frame_stride=1, processes=1,
                   profile="default", ligand_smiles=None, protonate=False,
                   protonate_ph=7.4, rule_overrides=None, progress=None):
    """Analyse Chemur interactions across every selected frame of a trajectory.

    ``structure`` must be a multi-coordset model (an MD trajectory loaded in
    ChimeraX). The loaded coordsets are serialised to a temporary topology mmCIF +
    coordinate stack which the engine reads (and parallelises over). Returns a
    :class:`TrajectoryRun`.
    """
    import numpy as np
    from chimerax.core.commands import run
    from chimerax.core.errors import UserError

    try:
        from chemur.parser import parse_structure
        from chemur.trajectory import ArrayFrameSource, analyze_trajectory
    except ImportError as e:
        # ArrayFrameSource needs only numpy -- there is no optional extra to install
        # here, so this means the engine itself is absent or broken.
        raise UserError(
            "The 'chemur' engine is not available in ChimeraX's Python: %s\n"
            "Install it with:\n"
            "    pip install chemur\n"
            "then restart ChimeraX." % e
        ) from e

    if structure.num_coordsets < 2:
        raise UserError(
            "#%s has a single frame; load a trajectory (multiple coordsets) first."
            % structure.id_string
        )

    # MD frames already carry explicit hydrogens, so we do NOT run 'addh': the real
    # per-frame H positions are better than a single static guess.
    _clear_pi_markers(structure)

    handle = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    handle.close()
    topo_path = handle.name
    coords_handle = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    coords_handle.close()
    coords_path = coords_handle.name
    try:
        # Save the topology at the active frame; the coordinate indices below are
        # built in that same frame so they line up with the written mmCIF.
        run(session, 'save "%s" models #%s format mmcif' % (topo_path, structure.id_string))
        topo_atoms, _ = parse_structure(topo_path)

        # Permutation: backend atom i -> column in structure.atoms (ChimeraX order).
        coord_index, name_index = _chimerax_atom_lookup(structure)
        index_of = {atom: i for i, atom in enumerate(structure.atoms)}
        perm = np.empty(len(topo_atoms), dtype=np.int64)
        unresolved = 0
        for i, rec in enumerate(topo_atoms):
            atom = _resolve_atom(rec, coord_index, name_index)
            if atom is None or atom not in index_of:
                unresolved += 1
                perm[i] = -1
            else:
                perm[i] = index_of[atom]
        if unresolved:
            raise UserError(
                "Could not align %d of %d topology atoms to the trajectory; "
                "the frames cannot be mapped to the structure." % (unresolved, len(topo_atoms))
            )

        # Stream every coordset into a (n_frames, n_atoms, 3) memmap in backend order.
        coordset_ids = list(structure.coordset_ids)
        stack = np.lib.format.open_memmap(
            coords_path, mode="w+", dtype=np.float32,
            shape=(len(coordset_ids), len(topo_atoms), 3),
        )
        saved_cs = structure.active_coordset_id
        try:
            for fi, cid in enumerate(coordset_ids):
                structure.active_coordset_id = cid
                stack[fi] = structure.atoms.coords[perm]
        finally:
            structure.active_coordset_id = saved_cs
        stack.flush()

        result = analyze_trajectory(
            ArrayFrameSource(topo_path, coords_path),
            profile=profile,
            rule_overrides=rule_overrides or None,
            ligand_smiles=ligand_smiles or None,
            protonate=protonate,
            protonate_ph_min=protonate_ph,
            protonate_ph_max=protonate_ph,
            exclude_solvent=exclude_solvent,
            exclude_ions=exclude_ions,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_stride=frame_stride,
            processes=processes,
            progress=progress,
        )
    finally:
        for path in (topo_path, coords_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    # Map the (filtered, renumbered) result atoms back to ChimeraX atoms by
    # residue/atom name (frame independent), then group atoms per interaction key.
    id_to_atom = {}
    for rec in result.atoms:
        atom = name_index.get((rec.chain_id, rec.residue_id, rec.residue_name, rec.name))
        if atom is not None:
            id_to_atom[rec.atom_id] = atom
    comp_of = {rec.atom_id: rec.component_id for rec in result.atoms}

    key_to_atoms = {}
    key_to_endpoints = {}
    for frame in result.frames:
        for inter in frame.interactions:
            key = result.interaction_key(inter)
            if key in key_to_atoms:
                continue
            atoms = [id_to_atom[a] for a in inter.atom_ids if a in id_to_atom]
            key_to_atoms[key] = atoms
            # Group the mapped atoms by their interacting component (endpoint),
            # preserving component order, for selection-scope tests in the GUI.
            groups = {}
            for a in inter.atom_ids:
                atom = id_to_atom.get(a)
                if atom is None:
                    continue
                groups.setdefault(comp_of.get(a), []).append(atom)
            key_to_endpoints[key] = list(groups.values())

    # Perceive features once. Feature perception is deterministic on the (frame
    # independent) topology, so these feature ids match every frame's stored
    # feature_ids and let us reuse the full _choose_endpoints / _draw_interaction
    # pipeline to draw any single frame (see draw_frame_interactions).
    rec_by_id = {rec.atom_id: rec for rec in result.atoms}
    try:
        from chemur.features import perceive_features
        features_by_id = {
            f.feature_id: f for f in perceive_features(result.atoms, result.components)
        }
    except Exception:
        features_by_id = {}

    frame_coordset_ids = [coordset_ids[i] for i in result.frame_indices]
    session.logger.info(
        "Chemur trajectory: analysed %d frames of #%s; %d distinct interactions."
        % (result.n_frames, structure.id_string, len(result.occupancy()))
    )
    return TrajectoryRun(result, key_to_atoms, key_to_endpoints, structure,
                         frame_coordset_ids, rec_by_id, features_by_id, id_to_atom)


def draw_frame_interactions(session, run, frame_index, *, included_types=None):
    """Draw one analysed frame's interactions as pseudobonds; return the rows.

    The structure should already be at ``frame_index``'s coordset (the GUI's frame
    slider sets it) so π ring-centre markers land on the right coordinates. Returns
    a list of :class:`InteractionRow` (empty if the frame wasn't analysed).
    """
    frame = next((f for f in run.result.frames if f.frame_index == frame_index), None)
    if frame is None:
        return []
    return _draw_interactions(
        session, run.structure, frame.interactions,
        run.rec_by_id, run.id_to_atom, run.features_by_id,
        included_types=included_types,
    )


def _build_ligand_report(result):
    """One :class:`LigandProtonation` per non-standard residue in ``result``."""
    input_smiles = result.metadata.get("input_ligand_smiles") or {}
    charge_by_component = {}
    for rec in result.atoms:
        if rec.molecule_type != "ligand":
            continue
        charge_by_component[rec.component_id] = (
            charge_by_component.get(rec.component_id, 0) + (rec.formal_charge or 0)
        )

    report = []
    for component in result.components:
        if component.molecule_type != "ligand":
            continue
        meta = component.metadata
        failed = bool(meta.get("ignored_by_template"))
        report.append(LigandProtonation(
            component_id=component.component_id,
            name=component.name,
            chain_id=component.chain_id,
            residue_id=component.residue_id,
            status="failed" if failed else "ok",
            smiles_used=meta.get("smiles"),
            input_smiles=(
                input_smiles.get(component.component_id.upper())
                or input_smiles.get(component.name.upper())
            ),
            net_charge=0 if failed else charge_by_component.get(component.component_id, 0),
            reason=meta.get("ignored_reason", ""),
        ))
    report.sort(key=lambda r: (r.status != "failed", r.chain_id or "", r.residue_id))
    return report


def _is_biopolymer_internal(ep1, ep2):
    return (
        (ep1.molecule_type == "protein" and ep2.molecule_type == "protein")
        or (ep1.molecule_type == "nucleotide" and ep2.molecule_type == "nucleotide")
    )


def _geometry_extra(inter):
    extra = {}
    for key in _GEOMETRY_METADATA_KEYS:
        if key in inter.metadata and inter.metadata[key] is not None:
            extra[key] = inter.metadata[key]
    return extra


# Max distance (Å) between a Chemur atom and the ChimeraX atom it resolves to.
# Far above 3-dp mmCIF rounding noise, far below any real atom separation, so a
# wrong-copy mapping (which is 10+ Å off) can never pass.
_COORD_EPS = 0.05


def _coord_key(coord):
    """Quantise a coordinate to 1/1000 Å (mmCIF write precision) for hashing."""
    return (round(coord[0] * 1000), round(coord[1] * 1000), round(coord[2] * 1000))


def _sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _chimerax_atom_lookup(structure):
    """Build the indices used to re-attach Chemur atoms to ChimeraX atoms.

    Returns ``(coord_index, name_index)``:

    - ``coord_index`` maps a quantised ``atom.coord`` -> ChimeraX Atom. Coordinates
      are unique per copy, so this resolves the *correct* protomer/ligand copy even
      when several share a chain/residue/atom name (the cause of the >10 Å lines on
      large assemblies like 6PUW).
    - ``name_index`` maps ``(chain_id, residue_id, residue_name, atom_name)`` ->
      ChimeraX Atom, kept only as a fallback. Keys mirror Chemur's parser: chain
      ``name or "_"`` and residue id ``f"{number}{insertion_code}"``.

    ``atom.coord`` is the model-local frame Chemur read from the saved mmCIF (the
    same frame ``_attach_ring`` already compares ring centres in).
    """
    return _chimerax_atom_lookup_multi([structure])


def _chimerax_atom_lookup_multi(structures):
    """Like :func:`_chimerax_atom_lookup` but over several structures' atoms.

    Used when an interaction run combines more than one open model (e.g. a
    receptor plus a docked ligand pose saved into one mmCIF): coordinates are
    globally unique, so the coord_index still resolves each engine atom to the
    correct ChimeraX atom across all the supplied structures, and a pose atom can
    never cross-map to a receptor atom.
    """
    coord_index = {}
    name_index = {}
    for structure in structures:
        for atom in structure.atoms:
            residue = atom.residue
            chain = residue.chain_id or "_"
            icode = (residue.insertion_code or "").strip()
            residue_id = "%d%s" % (residue.number, icode)
            coord_index.setdefault(_coord_key(atom.coord), atom)
            name_index.setdefault((chain, residue_id, residue.name, atom.name), atom)
    return coord_index, name_index


def _resolve_atom(rec, coord_index, name_index):
    """Resolve one Chemur ``AtomRecord`` to a ChimeraX Atom.

    Coordinate match first, name key as a fallback for floating-point boundary
    misses. Either way the result is accepted only if it actually sits on the
    Chemur atom (within ``_COORD_EPS``); otherwise we return ``None`` so the
    endpoint is treated as unresolved rather than drawn to the wrong atom.
    """
    atom = coord_index.get(_coord_key(rec.coord))
    if atom is None:
        atom = name_index.get((rec.chain_id, rec.residue_id, rec.residue_name, rec.name))
    if atom is None:
        return None
    if _sq_dist(atom.coord, rec.coord) > _COORD_EPS ** 2:
        return None
    return atom


def _choose_endpoints(inter, rec_by_id, id_to_atom, features_by_id):
    """Pick the two partner :class:`Endpoint`s for an interaction.

    Prefers two atoms from different Chemur components (the two interacting
    groups); falls back to the first two distinct mapped atoms. A side whose
    feature is an aromatic ring gets its ``ring_atoms`` attached so the line can
    terminate at the ring centroid.
    """
    mapped = []
    for aid in inter.atom_ids:
        atom = id_to_atom.get(aid)
        rec = rec_by_id.get(aid)
        if atom is not None and rec is not None:
            mapped.append((rec, atom))
    if len(mapped) < 2:
        return None

    first_rec, first_atom = mapped[0]
    chosen = None
    for rec, atom in mapped[1:]:
        if rec.component_id != first_rec.component_id and atom is not first_atom:
            chosen = (rec, atom)
            break
    if chosen is None:
        for rec, atom in mapped[1:]:
            if atom is not first_atom:
                chosen = (rec, atom)
                break
    if chosen is None:
        return None

    ring_feats = [features_by_id[fid] for fid in inter.feature_ids
                  if fid in features_by_id and features_by_id[fid].feature_type == "ring"]
    ep1 = _make_endpoint(first_rec, first_atom)
    ep2 = _make_endpoint(*chosen)
    _attach_ring(ep1, first_rec, ring_feats, id_to_atom)
    _attach_ring(ep2, chosen[0], ring_feats, id_to_atom)
    return ep1, ep2


def _attach_ring(endpoint, rec, ring_feats, id_to_atom):
    """If ``rec`` belongs to one of ``ring_feats``, attach the ring's mapped
    atoms and move the representative atom to the one nearest the ring centre."""
    for feat in ring_feats:
        if rec.atom_id not in feat.atom_ids:
            continue
        atoms = [id_to_atom[a] for a in feat.atom_ids if a in id_to_atom]
        if len(atoms) < 3:
            return
        endpoint.ring_atoms = atoms
        cx, cy, cz = feat.center
        endpoint.atom = min(
            atoms,
            key=lambda a: (a.coord[0] - cx) ** 2 + (a.coord[1] - cy) ** 2 + (a.coord[2] - cz) ** 2,
        )
        return


def _make_endpoint(rec, atom):
    return Endpoint(atom, rec.molecule_type, rec.chain_id, rec.residue_id, rec.residue_name)


# ----- pseudobond + ring-centre marker drawing ----------------------------

def _pi_group_name(structure):
    return "%s π #%s" % (PBG_NAME, structure.id_string)


def _clear_pi_markers(structure):
    """Remove any ring-centre MarkerSet left from a previous run."""
    from chimerax.markers import MarkerSet
    for child in list(structure.child_models()):
        if isinstance(child, MarkerSet) and child.name == _PI_MARKERSET_NAME:
            child.delete()


def _draw_interaction(session, structure, ep1, ep2, style, pbg, pi_pbg, holder):
    """Create the pseudobond, routing π (ring) endpoints to centre markers."""
    node1 = _node_for(ep1, style, holder, session, structure)
    node2 = _node_for(ep2, style, holder, session, structure)
    if node1 is None or node2 is None or node1 is node2:
        return None
    group = pi_pbg if (ep1.ring_atoms or ep2.ring_atoms) else pbg
    try:
        pb = group.new_pseudobond(node1, node2)
    except Exception:
        return None
    pb.color = style["color"]
    pb.radius = style["radius"]
    return pb


def _node_for(endpoint, style, holder, session, structure):
    """Atom the pseudobond attaches to: the real atom, or a ring-centre marker
    (one per ring) created in a child MarkerSet that tracks the structure."""
    if not endpoint.ring_atoms:
        return endpoint.atom
    key = frozenset(endpoint.ring_atoms)
    markers = holder.setdefault("markers", {})
    marker = markers.get(key)
    if marker is not None:
        return marker
    ms = holder.get("ms")
    if ms is None:
        from chimerax.markers import MarkerSet
        ms = MarkerSet(session, name=_PI_MARKERSET_NAME)
        structure.add([ms])
        holder["ms"] = ms
    n = len(endpoint.ring_atoms)
    cx = sum(a.coord[0] for a in endpoint.ring_atoms) / n
    cy = sum(a.coord[1] for a in endpoint.ring_atoms) / n
    cz = sum(a.coord[2] for a in endpoint.ring_atoms) / n
    from chimerax.atomic import Element
    marker = ms.create_marker((cx, cy, cz), style["color"], _PI_MARKER_RADIUS)
    # Re-element away from 'H' so 'delete H' leaves the ring-centre markers intact.
    marker.element = Element.get_element(_PI_MARKER_ELEMENT)
    markers[key] = marker
    return marker


def _pose_markerset_name(group_name):
    return "%s π-centres" % group_name


def _pose_node_for(endpoint, style, holder, session):
    """Like :func:`_node_for`, but a ring's centre marker is attached to the
    endpoint atom's OWN structure — a pose interaction spans two models (receptor
    + pose), so each ring marker must live on the model whose ring it is."""
    if not endpoint.ring_atoms:
        return endpoint.atom
    key = frozenset(endpoint.ring_atoms)
    markers = holder.setdefault("markers", {})
    marker = markers.get(key)
    if marker is not None:
        return marker
    structure = endpoint.atom.structure
    ms_by_struct = holder.setdefault("ms", {})
    ms = ms_by_struct.get(structure)
    if ms is None:
        from chimerax.markers import MarkerSet
        ms = MarkerSet(session, name=holder["ms_name"])
        structure.add([ms])
        ms_by_struct[structure] = ms
    n = len(endpoint.ring_atoms)
    cx = sum(a.coord[0] for a in endpoint.ring_atoms) / n
    cy = sum(a.coord[1] for a in endpoint.ring_atoms) / n
    cz = sum(a.coord[2] for a in endpoint.ring_atoms) / n
    from chimerax.atomic import Element
    marker = ms.create_marker((cx, cy, cz), style["color"], _PI_MARKER_RADIUS)
    marker.element = Element.get_element(_PI_MARKER_ELEMENT)
    markers[key] = marker
    return marker


def draw_pose_rows(session, rows, group_name):
    """Draw one pose's interactions as pseudobonds in a named GLOBAL group.

    ``rows`` are :class:`InteractionRow`s from a pose comparison; their endpoint
    atoms are live ChimeraX atoms spanning the receptor + pose models, so a global
    (cross-model) pseudobond group is required. Any previous drawing for the same
    ``group_name`` is cleared first. Returns the group.
    """
    clear_pose_group(session, group_name)
    pbg = session.pb_manager.get_group(group_name, create=True)
    holder = {"ms_name": _pose_markerset_name(group_name)}
    for row in rows:
        try:
            style = style_for(row.interaction_type)
            node1 = _pose_node_for(row.endpoint1, style, holder, session)
            node2 = _pose_node_for(row.endpoint2, style, holder, session)
            if node1 is None or node2 is None or node1 is node2:
                continue
            pb = pbg.new_pseudobond(node1, node2)
            pb.color = style["color"]
            pb.radius = style["radius"]
        except Exception:
            # Skip a row whose atoms were deleted (e.g. the model was closed).
            continue
    if pbg.num_pseudobonds:
        if pbg.id is None:
            session.models.add([pbg])
    elif pbg.id is None:
        pbg.delete()
    return pbg


def clear_pose_group(session, group_name):
    """Remove a pose's pseudobond group and its ring-centre MarkerSets."""
    from chimerax.markers import MarkerSet
    ms_name = _pose_markerset_name(group_name)
    for m in session.models.list():
        if isinstance(m, MarkerSet) and m.name == ms_name:
            m.delete()
    pbg = session.pb_manager.get_group(group_name, create=False)
    if pbg is None:
        return
    if pbg.num_pseudobonds:
        pbg.pseudobonds.delete()
    if pbg.id is not None:
        session.models.close([pbg])
    else:
        pbg.delete()


def _format_value(value):
    if isinstance(value, float):
        return "%.2f" % value
    return str(value)


def _log_summary(session, structure, counts, drawn, unresolved, filtered):
    logger = session.logger
    if not drawn:
        logger.warning("Chemur: no interactions to show for #%s." % structure.id_string)
        return
    parts = ", ".join("%s: %d" % (k, counts[k]) for k in sorted(counts))
    msg = "Chemur drew %d interactions on #%s (%s)." % (drawn, structure.id_string, parts)
    if filtered:
        msg += " %d hidden by filters." % filtered
    if unresolved:
        msg += " %d skipped (atoms could not be mapped)." % unresolved
    logger.info(msg)

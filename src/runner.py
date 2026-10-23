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
from copy import deepcopy
from itertools import count

from .engine_support import (require_engine, check_cancel, analysis_options,
                             projected_interactions, capture_analysis_options)

from .colors import style_for

_DRAW_IDS = count(1)
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

    ``ring_atoms`` holds ring members for a ring-centroid endpoint.
    ``center_atoms`` holds atoms defining another geometric center, such as the
    carbon/nitrogen midpoint of an amide. ``atom`` remains a representative
    atom for labels and residue grouping.
    """

    __slots__ = ("atom", "molecule_type", "chain_id", "residue_id", "residue_name",
                 "ring_atoms", "center_atoms")

    def __init__(self, atom, molecule_type, chain_id, residue_id, residue_name,
                 ring_atoms=None, center_atoms=None):
        self.atom = atom
        self.molecule_type = molecule_type
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.residue_name = residue_name
        self.ring_atoms = ring_atoms
        self.center_atoms = center_atoms

    @property
    def residue_key(self):
        return (self.molecule_type, self.chain_id, self.residue_id, self.residue_name)

    @property
    def residue_label(self):
        return "%s %s" % (self.residue_name, self.residue_id)

    @property
    def atom_label(self):
        if self.center_atoms:
            return "midpoint (%s)" % ", ".join(
                "/%s %s %s%s %s" % (atom.residue.chain_id, atom.residue.name,
                    atom.residue.number, (atom.residue.insertion_code or "").strip(), atom.name)
                for atom in self.center_atoms)
        return "/%s %s %s %s" % (self.chain_id, self.residue_name, self.residue_id, self.atom.name)


class InteractionRow:
    """One resolved interaction, ready for the results tree."""

    __slots__ = ("interaction_type", "endpoint1", "endpoint2", "distance",
                 "angle", "offset", "extra", "pseudobond", "interaction",
                 "atom_ids", "feature_ids", "component_ids", "assignment_order")

    def __init__(self, interaction_type, endpoint1, endpoint2, distance,
                 angle, offset, extra, pseudobond, interaction=None):
        self.interaction_type = interaction_type
        self.endpoint1 = endpoint1
        self.endpoint2 = endpoint2
        self.distance = distance
        self.angle = angle
        self.offset = offset
        self.extra = extra  # dict of other geometry (donor_angle, geometry, ...)
        self.pseudobond = pseudobond
        self.interaction = interaction
        self.atom_ids = tuple(getattr(interaction, "atom_ids", ()))
        self.feature_ids = tuple(getattr(interaction, "feature_ids", ()))
        self.component_ids = tuple(getattr(interaction, "component_ids", ()))
        self.assignment_order = getattr(interaction, "assignment_order", None)

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
                 "status", "smiles_used", "input_smiles", "net_charge", "reason",
                 "assembly_id", "component_ids", "editable", "chemistry_source", "diagnostics")

    def __init__(self, component_id, name, chain_id, residue_id, status,
                 smiles_used, input_smiles, net_charge, reason, *, assembly_id=None,
                 component_ids=None, editable=True, chemistry_source=None, diagnostics=()):
        self.component_id = component_id
        self.name = name
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.status = status
        self.smiles_used = smiles_used
        self.input_smiles = input_smiles
        self.net_charge = net_charge
        self.reason = reason
        self.assembly_id = assembly_id or component_id
        self.component_ids = tuple(component_ids or (component_id,))
        self.editable = editable
        self.chemistry_source = chemistry_source
        self.diagnostics = tuple(diagnostics)

    @property
    def residue_label(self):
        chain = self.chain_id or "_"
        return "%s /%s %s" % (self.name, chain, self.residue_id)


class InteractionRun:
    """A native assigned result plus its separately filtered display projection."""

    def __init__(self, rows, ligand_report, result=None, source_models=(),
                 id_to_atom=None, graphics=None, analysis_options=None):
        self.analysis_options = deepcopy(analysis_options or {})
        self.rows = rows
        self.ligand_report = ligand_report
        self.result = result
        self.source_models = tuple(source_models)
        self.id_to_atom = dict(id_to_atom or {})
        self.graphics = list(graphics or ())
        self.counts = {}
        for row in rows:
            self.counts[row.interaction_type] = self.counts.get(row.interaction_type, 0) + 1


class PoseInteractionRun(InteractionRun):
    """Native pose analysis, with explicit heavy-atom mapping completeness."""

    def __init__(self, rows, result, fully_mapped=False, *, mapping_status=None,
                 source_models=(), id_to_atom=None, graphics=None, pose_component_ids=None,
                 analysis_options=None):
        super().__init__(rows, _build_ligand_report(result), result, source_models,
                         id_to_atom, graphics, analysis_options=analysis_options)
        self.mapping_status = mapping_status or ("complete" if fully_mapped else "unavailable")
        self.pose_component_ids = None if pose_component_ids is None else tuple(pose_component_ids)

    @property
    def fully_mapped(self):
        return self.mapping_status == "complete"


class TrajectoryRun:
    """Native trajectory plus stable topology mapping and frame-aware participants."""

    def __init__(self, result, key_to_atoms, key_to_endpoints, structure,
                 frame_coordset_ids, rec_by_id, features_by_id, id_to_atom,
                 graphics=None, analysis_options=None):
        self.analysis_options = deepcopy(analysis_options or {})
        self.result = result
        self.ligand_report = _build_ligand_report(result)
        self.key_to_atoms = key_to_atoms
        self.key_to_endpoints = key_to_endpoints
        self.structure = structure
        self.source_models = (structure,)
        self.frame_coordset_ids = frame_coordset_ids
        self.rec_by_id = rec_by_id
        self.features_by_id = features_by_id
        self.id_to_atom = id_to_atom
        self.graphics = list(graphics or ())
        self.rows = []
        self._frames = {f.frame_index: f for f in result.frames}

    def frame_participants(self, frame_index=None, key=None):
        frames = self.result.frames if frame_index is None else [self._frames.get(frame_index)]
        mapping = {}
        for frame in frames:
            if frame is None:
                continue
            for inter in frame.interactions:
                ids = inter.atom_ids
                if frame_index is None and inter.interaction_type == "hydrophobic_enclosure":
                    ids = [inter.metadata["center_atom_id"]]
                mapping[self.result.interaction_key(inter)] = [
                    self.id_to_atom[a] for a in ids if a in self.id_to_atom]
        if key is None:
            return mapping
        if key in mapping:
            return mapping[key]
        # Selecting an enclosure in a frame where it is absent still selects
        # its stable center, never a shell borrowed from another frame. The
        # per-frame mapping and endpoint groups remain empty for scope filters.
        if (frame_index in self._frames and key[0] == "hydrophobic_enclosure"):
            return self.frame_participants(None, key)
        return []

    def frame_endpoints(self, frame_index=None, key=None):
        frames = self.result.frames if frame_index is None else [self._frames.get(frame_index)]
        mapping = {}
        for frame in frames:
            if frame is None:
                continue
            for inter in frame.interactions:
                groups = _interaction_groups(inter, self.rec_by_id, self.id_to_atom,
                                             self.features_by_id)
                if frame_index is None and inter.interaction_type == "hydrophobic_enclosure":
                    groups = groups[:1]
                mapping[self.result.interaction_key(inter)] = groups
        return mapping if key is None else mapping.get(key, [])


def _interaction_groups(inter, rec_by_id, id_to_atom, features_by_id=None):
    if inter.interaction_type == "hydrophobic_enclosure":
        center = inter.metadata.get("center_atom_id")
        return [[id_to_atom[center]] if center in id_to_atom else [],
                [id_to_atom[a] for a in inter.metadata.get("neighbor_atom_ids", ())
                 if a in id_to_atom]]
    if inter.interaction_type == "amide_pi":
        features = _amide_pi_features(inter, features_by_id or {})
        if features is None:
            return []
        # A backbone amide spans two residues but is one interaction partner.
        return [[id_to_atom[a] for a in feat.atom_ids if a in id_to_atom]
                for feat in features]
    groups = {}
    for aid in inter.atom_ids:
        if aid in id_to_atom and aid in rec_by_id:
            groups.setdefault(rec_by_id[aid].component_id, []).append(id_to_atom[aid])
    return list(groups.values())


def _map_atoms(result, structures):
    coord_index, name_index = _chimerax_atom_lookup_multi(structures)
    mapped = {}
    for rec in result.atoms:
        atom = _resolve_atom(rec, coord_index, name_index)
        if atom is not None:
            mapped[rec.atom_id] = atom
    # A live atom cannot represent two distinct native atoms.
    owners = {}
    for aid, atom in mapped.items():
        owners.setdefault(id(atom), []).append(aid)
    duplicate_ids = {aid for aids in owners.values() if len(aids) > 1 for aid in aids}
    return {aid: atom for aid, atom in mapped.items() if aid not in duplicate_ids}


def run_interactions(session, structure, *, protonate=False, protonate_ph=7.4,
                     profile="default", ligand_smiles=None, add_hydrogens=True,
                     rule_overrides=None, selected_only=False,
                     skip_biopolymer_internal=False, assembly_overrides=None,
                     hydrophobic_enclosure=None, cancel_check=None):
    """Analyse native assigned data, then map/filter only its displayed projection."""
    from chimerax.core.errors import UserError

    engine = require_engine()
    options = analysis_options(engine, profile=profile, rule_overrides=rule_overrides,
                               hydrophobic_enclosure=hydrophobic_enclosure,
                               assembly_overrides=assembly_overrides)
    settings = capture_analysis_options(engine, options, protonate=protonate,
        protonate_ph_min=protonate_ph, protonate_ph_max=protonate_ph,
        ligand_smiles=ligand_smiles, add_hydrogens=add_hydrogens,
        selected_only=selected_only, skip_biopolymer_internal=skip_biopolymer_internal,
        hydrophobic_enclosure=hydrophobic_enclosure)
    check_cancel(cancel_check)
    if selected_only and not any(atom.selected for atom in structure.atoms):
        raise UserError("Select some atoms first, or turn off 'selected atoms only'.")
    _addh_and_clear(session, [structure], add_hydrogens, cancel_check=cancel_check)
    path = save_models_mmcif(session, [structure])
    try:
        check_cancel(cancel_check)
        result = engine.analyze(path, protonate=protonate,
                                protonate_ph_min=protonate_ph, protonate_ph_max=protonate_ph,
                                ligand_smiles=ligand_smiles or None, **options)
    finally:
        _unlink(path)
    check_cancel(cancel_check)
    ligand_report = _build_ligand_report(result)
    skipped = result.metadata.get("skipped_ligands") or {}
    if skipped:
        session.logger.warning("Chemur skipped %d ligand(s): %s" %
                               (len(skipped), ", ".join(sorted(skipped))))
    id_to_atom = _map_atoms(result, [structure])
    check_cancel(cancel_check)
    graphics = []
    rows = _draw_interactions(
        session, structure, result.interactions,
        {a.atom_id: a for a in result.atoms}, id_to_atom,
        {f.feature_id: f for f in result.features}, selected_only=selected_only,
        skip_biopolymer_internal=skip_biopolymer_internal,
        graphics=graphics, cancel_check=cancel_check)
    run = InteractionRun(rows, ligand_report, result, [structure], id_to_atom, graphics,
                         analysis_options=settings)
    _track_graphics(session, run.source_models, run.graphics)
    if cancel_check is None:
        previous = getattr(structure, "_chemur_command_graphics", ())
        structure._chemur_command_graphics = run.graphics
        _discard_graphics(session, previous)
    return run


def _import_analyze():
    return require_engine().analyze


def _discard_graphics(session, graphics):
    """Best-effort removal of partially created graphics on failed/cancelled draws."""
    for handle in list(graphics):
        if getattr(handle, "deleted", False):
            continue
        try:
            if getattr(handle, "id", None) is None:
                handle.delete()
            else:
                session.models.close([handle])
        except Exception:
            pass


def _track_graphics(session, source_models, graphics):
    from .lifecycle import track_graphics
    return track_graphics(session, source_models, graphics)


def _unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _has_protein(structure):
    """True if ``structure`` contains any amino-acid polymer residue."""
    from chimerax.atomic import Residue
    return any(r.polymer_type == Residue.PT_AMINO for r in structure.residues)


def _addh_and_clear(session, structures, add_hydrogens, *, cancel_check=None):
    """Prepare input hydrogens; old result graphics survive until a new draw."""
    from chimerax.core.commands import run
    if add_hydrogens:
        for structure in structures:
            check_cancel(cancel_check)
            run(session, "addh #%s" % structure.id_string)
    check_cancel(cancel_check)


def save_models_mmcif(session, structures):
    """Save one scene-coordinate snapshot, without child markers or inactive states.

    ChimeraX writes separate data blocks for separate models and emits alternate
    conformers. Chemur consumes one coordinate block: normalize the temporary
    export to exactly the live atoms' active positions, keeping named chemistry.
    No source model, coordinate set, or alternate-location selection is changed.
    """
    from chimerax.mmcif.mmcif_write import write_mmcif
    from .source_chemistry import preserve_source_chemistry
    structures = list(structures)
    source_components = {}
    for model_index, structure in enumerate(structures):
        for atom in structure.atoms:
            residue = atom.residue
            identity = (residue.chain_id or "_", residue.name,
                        "%d%s" % (residue.number, (residue.insertion_code or "").strip()))
            previous = source_components.setdefault(identity, model_index)
            if previous != model_index:
                raise ValueError("Source models contain the same component identity %s. "
                                 "Give the models distinct chain identifiers before joint analysis." %
                                 ":".join(identity))
    snapshots = []
    for model_index, structure in enumerate(structures):
        for atom_index, atom in enumerate(structure.atoms):
            residue = atom.residue
            snapshots.append({
                "id": (model_index, atom_index), "model": model_index,
                "element": atom.element.name.upper(),
                "name": (residue.chain_id or "_", residue.name,
                         "%d%s" % (residue.number, (residue.insertion_code or "").strip()), atom.name),
                "coord": tuple(float(v) for v in _scene_coord(atom)),
                "altloc": str(getattr(atom, "alt_loc", "") or "").strip(),
            })
    handle = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    handle.close()
    try:
        write_mmcif(session, handle.name, models=structures, rel_model=None, all_coordsets=False)
        preserve_source_chemistry(handle.name, structures, logger=session.logger)
        _normalize_mmcif_snapshot(handle.name, snapshots, len(structures))
    except BaseException:
        _unlink(handle.name)
        raise
    return handle.name


def _normalize_mmcif_snapshot(path, snapshots, model_count):
    """Keep one active row per input atom and merge molecular tables into one block."""
    import gemmi
    document = gemmi.cif.read_file(str(path))
    coordinate_blocks = [block for block in document if block.get_mmcif_category("_atom_site.")]
    if not coordinate_blocks:
        raise ValueError("ChimeraX wrote no coordinates for the selected source models")
    block_owner = {block.name: index for index, block in enumerate(coordinate_blocks)}
    by_name = {}
    for atom in snapshots:
        by_name.setdefault(atom["name"], []).append(atom)

    def value(row, *keys, default=""):
        for key in keys:
            v = row.get(key)
            if v not in (None, False, "", ".", "?"):
                return str(v)
        return default

    # These are the coordinate/chemistry categories the engine reads, plus the
    # entity/asymmetry tables needed to make their identifiers unambiguous.
    categories = (
        "_atom_site.", "_chem_comp.", "_chem_comp_atom.", "_chem_comp_bond.",
        "_pdbx_chem_comp_descriptor.", "_struct_conn.", "_struct_asym.",
        "_entity.", "_entity_poly.", "_entity_poly_seq.", "_pdbx_entity_nonpoly.",
        "_pdbx_poly_seq_scheme.", "_pdbx_nonpoly_scheme.", "_pdbx_branch_scheme.",
        "_pdbx_entity_branch.", "_pdbx_entity_branch_list.", "_pdbx_entity_branch_link.",
        "_chemur_source_residue.", "_pdbx_struct_mod_residue.",
    )
    merged = {category: [] for category in categories}
    seen_rows = {category: set() for category in categories}
    seen_atoms = set()
    for block_index, block in enumerate(document):
        owner = block_owner.get(block.name) if len(coordinate_blocks) == model_count else None
        prefix = "s%d_" % (block_index + 1) if len(coordinate_blocks) > 1 else ""
        for category in categories:
            columns = block.get_mmcif_category(category)
            if not columns:
                continue
            for values in zip(*columns.values()):
                row = dict(zip(columns, values))
                if category == "_atom_site.":
                    name = (value(row, "auth_asym_id", "label_asym_id", default="_"),
                            value(row, "auth_comp_id", "label_comp_id"),
                            value(row, "auth_seq_id", "label_seq_id") + value(row, "pdbx_PDB_ins_code"),
                            value(row, "auth_atom_id", "label_atom_id"))
                    coord = tuple(float(row[key]) for key in ("Cartn_x", "Cartn_y", "Cartn_z"))
                    altloc = value(row, "label_alt_id")
                    element = value(row, "type_symbol").upper()
                    matches = [atom for atom in by_name.get(name, ())
                               if (owner is None or atom["model"] == owner)
                               and atom["element"] == element
                               and (not altloc or atom["altloc"] == altloc)
                               and _sq_dist(atom["coord"], coord) <= _COORD_EPS ** 2]
                    if not matches:
                        continue  # An inactive alternate conformer.
                    if len(matches) != 1:
                        raise ValueError("Source atom identity is ambiguous while exporting %s" % (name,))
                    atom = matches[0]
                    if atom["id"] in seen_atoms:
                        continue
                    seen_atoms.add(atom["id"])
                    row["id"] = str(len(merged[category]) + 1)
                    row["pdbx_PDB_model_num"] = "1"
                if prefix:
                    for key, item in list(row.items()):
                        if item in (None, False, "", ".", "?"):
                            continue
                        if (key in {"label_asym_id", "asym_id", "label_entity_id", "entity_id"}
                                or key.endswith("_label_asym_id")
                                or (category in {"_entity.", "_struct_asym.", "_struct_conn."} and key == "id")):
                            row[key] = prefix + str(item)
                # Identical component dictionaries across source models are one
                # dictionary. Conflicting definitions remain visible to Chemur's
                # validator instead of silently choosing one chemical state.
                row_key = tuple(sorted(row.items()))
                if row_key not in seen_rows[category]:
                    seen_rows[category].add(row_key)
                    merged[category].append(row)
    missing = [atom for atom in snapshots if atom["id"] not in seen_atoms]
    if missing:
        raise ValueError("Could not export %d source atom(s) at their active coordinates (first: %s)" %
                         (len(missing), ":".join(missing[0]["name"])))
    combined = gemmi.cif.Document()
    block = combined.add_new_block("chemur_snapshot")
    source_entries = {row.get("source_entry_id") for row in merged["_chemur_source_residue."]
                      if row.get("source_entry_id") not in (None, False, "", ".", "?")}
    block.set_pair("_entry.id", next(iter(source_entries)) if len(source_entries) == 1 else "chemur_snapshot")
    for category, rows in merged.items():
        if not rows:
            continue
        names = list(dict.fromkeys(key for row in rows for key in row))
        block.set_mmcif_category(category, {name: [row.get(name, False) for row in rows] for name in names})
    combined.write_file(str(path))


def _source_atom_snapshot(structures, pose):
    """Freeze input ownership before analysis, independently of later live mapping."""
    snapshot = []
    for structure in structures:
        for atom in structure.atoms:
            residue = atom.residue
            snapshot.append((structure is pose, atom.element.name.upper(),
                (residue.chain_id or "_", residue.name,
                 "%d%s" % (residue.number, (residue.insertion_code or "").strip()), atom.name),
                tuple(float(v) for v in _scene_coord(atom))))
    return tuple(snapshot)


def _pose_component_ids(result, *, whole_complex=False, sdf_path=None, snapshot=None):
    """Identify native targets from input provenance, never from drawable contacts.

    None denotes ambiguous provenance, () a confidently absent native ligand.
    Component-ID collisions across receptor and pose cannot be represented safely
    by a component selection and are explicitly unresolved.
    """
    ligand_atoms = [a for a in result.atoms if a.molecule_type == "ligand" and a.element != "H"]
    if whole_complex:
        selected = {a.component_id for a in ligand_atoms}
    elif sdf_path is not None:
        selected = {a.component_id for a in ligand_atoms
                    if a.metadata.get("source") == "sdf"
                    and a.metadata.get("source_path") == str(sdf_path)}
        selected.update(c.component_id for c in result.components
                        if c.molecule_type == "ligand" and c.metadata.get("source") == "sdf"
                        and c.metadata.get("source_path") == str(sdf_path))
    elif snapshot is not None:
        ownership = {}
        spatial = {}
        def cell(coord):
            import math
            return tuple(math.floor(float(value) / _COORD_EPS) for value in coord)
        for item in snapshot:
            spatial.setdefault((item[1], cell(item[3])), []).append(item)
        for atom in ligand_atoms:
            x, y, z = cell(atom.coord)
            nearby = [item for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                      for item in spatial.get((atom.element.upper(), (x + dx, y + dy, z + dz)), ())]
            matches = [item for item in nearby if _sq_dist(item[3], atom.coord) <= _COORD_EPS ** 2]
            name = (atom.chain_id, atom.residue_name, str(atom.residue_id), atom.name)
            named = [item for item in matches if item[2] == name]
            if named:
                matches = named
            owners = {item[0] for item in matches}
            if len(owners) != 1:
                return None
            ownership.setdefault(atom.component_id, set()).update(owners)
        if any(len(owners) > 1 for owners in ownership.values()):
            return None
        selected = {cid for cid, owners in ownership.items() if owners == {True}}
    else:
        return None
    registry = result.metadata.get("ligand_assemblies") or {}
    for record in registry.values():
        members = set(record.get("component_ids", ()))
        if members & selected:
            selected.update(members)
    return tuple(sorted(selected))


def run_pose_interactions(session, pose, receptor=None, *, protonate=False,
                          protonate_ph=7.4, profile="default", ligand_smiles=None,
                          add_hydrogens=True, rule_overrides=None,
                          skip_biopolymer_internal=True, draw=False,
                          selected_only=False, receptor_path=None,
                          receptor_acceptors=None, assembly_overrides=None,
                          hydrophobic_enclosure=None, cancel_check=None):
    """Analyse a pose while retaining its full native result and mapping status."""
    from .sdf_io import has_bond_orders, write_ligand_sdf

    engine = require_engine()
    common = analysis_options(engine, profile=profile, rule_overrides=rule_overrides,
                              hydrophobic_enclosure=hydrophobic_enclosure,
                              assembly_overrides=assembly_overrides)
    common.update(protonate=protonate, protonate_ph_min=protonate_ph,
                  protonate_ph_max=protonate_ph)
    check_cancel(cancel_check)
    pose_has_protein = _has_protein(pose)
    use_sdf = not pose_has_protein and receptor is not None and has_bond_orders(pose)
    settings = capture_analysis_options(engine, common, ligand_smiles=ligand_smiles,
        add_hydrogens=add_hydrogens, selected_only=selected_only,
        skip_biopolymer_internal=skip_biopolymer_internal, draw=draw,
        hydrophobic_enclosure=hydrophobic_enclosure,
        input_route="sdf" if use_sdf else "complex_mmcif" if pose_has_protein else "merged_mmcif")
    temp_paths = []
    pose_snapshot = None
    sdf_path = None
    try:
        if use_sdf:
            rpath = receptor_path
            if rpath is None:
                _addh_and_clear(session, [receptor], add_hydrogens, cancel_check=cancel_check)
                rpath = save_models_mmcif(session, [receptor])
                temp_paths.append(rpath)
            check_cancel(cancel_check)
            orient_hydrogens = not (protonate or ligand_smiles or assembly_overrides)
            targets = receptor_acceptors if orient_hydrogens else None
            if add_hydrogens and orient_hydrogens and targets is None:
                from .lifecycle import RunCancelled
                try:
                    targets = prepare_receptor_acceptors(rpath, engine=engine,
                        assembly_overrides=assembly_overrides, cancel_check=cancel_check)
                except RunCancelled:
                    raise
                except Exception as exc:
                    session.logger.warning("Chemur could not type receptor acceptors; "
                                           "receptor-guided hydrogen orientation was skipped: %s" % exc)
                    targets = ()
            sdf_handle = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
            sdf_handle.close()
            temp_paths.append(sdf_handle.name)
            sdf_path = sdf_handle.name
            check_cancel(cancel_check)
            write_ligand_sdf(pose, sdf_handle.name, name=_ligand_name(pose),
                             add_hydrogens=add_hydrogens,
                             acceptor_coords=_nearby_acceptors(targets, pose))
            structures = [receptor, pose]
            check_cancel(cancel_check)
            result = engine.analyze(rpath, ligand_sdf=[sdf_handle.name],
                                    ligand_smiles=ligand_smiles or None, **common)
        else:
            structures = [pose] if pose_has_protein or receptor is None else [receptor, pose]
            _addh_and_clear(session, structures, add_hydrogens, cancel_check=cancel_check)
            pose_snapshot = _source_atom_snapshot(structures, pose)
            path = save_models_mmcif(session, structures)
            temp_paths.append(path)
            check_cancel(cancel_check)
            result = engine.analyze(path, ligand_smiles=ligand_smiles or None, **common)
    finally:
        for path in temp_paths:
            _unlink(path)
    # In particular, never touch models after a closed-model cancellation.
    check_cancel(cancel_check)
    target_components = _pose_component_ids(result, whole_complex=pose_has_protein,
                                           sdf_path=sdf_path, snapshot=pose_snapshot)
    id_to_atom = _map_atoms(result, structures)
    ligand_ids = [a.atom_id for a in result.atoms if a.molecule_type == "ligand"
                  and target_components is not None and a.component_id in target_components
                  and a.element != "H" and not a.metadata.get("ignored_by_template")]
    rec_by_id = {a.atom_id: a for a in result.atoms}
    required_ids = set(ligand_ids)
    for interaction in projected_interactions(result.interactions):
        if set(ligand_ids).intersection(interaction.atom_ids):
            required_ids.update(aid for aid in interaction.atom_ids
                                if aid in rec_by_id and rec_by_id[aid].element != "H")
    resolved = sum(aid in id_to_atom for aid in ligand_ids)
    status = ("complete" if ligand_ids and required_ids.issubset(id_to_atom)
              else "partial" if resolved else "unavailable")
    features_by_id = {f.feature_id: f for f in result.features}
    graphics = []
    check_cancel(cancel_check)
    if draw:
        draw_on = receptor if len(structures) == 2 else pose
        rows = _draw_interactions(session, draw_on, result.interactions,
            rec_by_id, id_to_atom, features_by_id, selected_only=selected_only,
            skip_biopolymer_internal=skip_biopolymer_internal,
            graphics=graphics, cancel_check=cancel_check)
    else:
        rows = _resolve_rows(result.interactions, rec_by_id, id_to_atom, features_by_id,
                             selected_only=selected_only,
                             skip_biopolymer_internal=skip_biopolymer_internal)
    run = PoseInteractionRun(rows, result, mapping_status=status,
                             source_models=structures, id_to_atom=id_to_atom, graphics=graphics,
                             pose_component_ids=target_components, analysis_options=settings)
    if draw:
        _track_graphics(session, run.source_models, run.graphics)
    return run


def _ligand_name(pose):
    """A short, file-safe ligand name for the temp SDF (cosmetic)."""
    base = "".join(c for c in (pose.name or "LIG") if c.isalnum()) or "LIG"
    return base[:16].upper()


# Only receptor atoms within this far of the pose can accept from it; the rest are
# not worth carrying into the per-hydrogen scoring.
_ACCEPTOR_CUTOFF = 6.0
def prepare_receptor_acceptors(path, *, engine=None, assembly_overrides=None,
                               cancel_check=None):
    """Typed scene-coordinate receptor snapshot matching the SDF analysis route."""
    engine = engine or require_engine()
    check_cancel(cancel_check)
    atoms, components = engine.parse_structure(path, include_ligands=False)
    engine.prepare_ligand_chemistry(atoms, components, sdf_mode=True,
                                   assembly_overrides=assembly_overrides, fetch_ccd=False)
    check_cancel(cancel_check)
    return tuple(f.center for f in engine.perceive_features(atoms, components)
                 if f.feature_type == "acceptor")


def _nearby_acceptors(acceptor_coords, pose):
    """Restrict an already typed snapshot to the scene-coordinate pose neighborhood."""
    if acceptor_coords is None:
        return None
    pose_coords = [_scene_coord(atom) for atom in pose.atoms]
    if not pose_coords:
        return []
    cutoff2 = _ACCEPTOR_CUTOFF ** 2
    return [tuple(float(v) for v in coord) for coord in acceptor_coords
            if any(_sq_dist(coord, p) <= cutoff2 for p in pose_coords)]


def _resolve_rows(interactions, rec_by_id, id_to_atom, features_by_id, *,
                  skip_biopolymer_internal=False, included_types=None,
                  selected_only=False):
    """Map actual unique legs; native interactions are never view-filtered."""
    rows = []
    for inter in projected_interactions(interactions):
        if included_types is not None and inter.interaction_type not in included_types:
            continue
        endpoints = _choose_endpoints(inter, rec_by_id, id_to_atom, features_by_id)
        if endpoints is None:
            continue
        ep1, ep2 = endpoints
        if selected_only and not any(
                atom.selected for ep in endpoints
                for atom in (ep.center_atoms or [ep.atom])):
            continue
        if skip_biopolymer_internal and _is_biopolymer_internal(ep1, ep2):
            continue
        rows.append(InteractionRow(inter.interaction_type, ep1, ep2,
            inter.distance, inter.angle, inter.offset, _geometry_extra(inter), None, inter))
    return rows


def _draw_interactions(session, structure, interactions, rec_by_id, id_to_atom,
                       features_by_id, *, selected_only=False,
                       skip_biopolymer_internal=False, included_types=None,
                       graphics=None, cancel_check=None):
    """Replace display graphics after successful analysis and cancellation checks."""
    check_cancel(cancel_check)
    projected = projected_interactions(interactions)
    rows = _resolve_rows(projected, rec_by_id, id_to_atom, features_by_id,
        selected_only=selected_only, skip_biopolymer_internal=skip_biopolymer_internal,
        included_types=included_types)
    check_cancel(cancel_check)
    draw_id = next(_DRAW_IDS)
    pbg = structure.pseudobond_group("%s (%d)" % (PBG_NAME, draw_id))
    pi_pbg = session.pb_manager.get_group("%s (%d)" % (_pi_group_name(structure), draw_id), create=True)
    marker_holder = {"graphics": graphics, "ms_name": "%s (%d)" % (_PI_MARKERSET_NAME, draw_id)}
    if graphics is not None:
        graphics[:] = [pbg, pi_pbg]
    counts = {}
    try:
        for row in rows:
            check_cancel(cancel_check)
            pb = _draw_interaction(session, structure, row.endpoint1, row.endpoint2,
                style_for(row.interaction_type), pbg, pi_pbg, marker_holder)
            row.pseudobond = pb
            counts[row.interaction_type] = counts.get(row.interaction_type, 0) + 1
        check_cancel(cancel_check)
    except BaseException:
        _discard_graphics(session, [pbg, pi_pbg, *marker_holder.get("ms", {}).values()])
        if graphics is not None:
            graphics.clear()
        raise
    if pi_pbg.num_pseudobonds:
        if pi_pbg.id is None:
            session.models.add([pi_pbg])
    else:
        if pi_pbg.id is None:
            pi_pbg.delete()
        else:
            session.models.close([pi_pbg])
        if graphics is not None:
            graphics[:] = [g for g in graphics if g is not pi_pbg]
    drawn = sum(row.pseudobond is not None for row in rows)
    enclosure_groups = {(group.get("center_atom_id"), group.get("assignment_order"))
                        for row in rows for group in row.extra.get("enclosure_groups", ())}
    _log_summary(session, structure, counts, drawn, len(rows) - drawn,
                 max(0, len(projected) - len(rows)), enclosure_groups=len(enclosure_groups))
    return rows


def run_trajectory(session, structure, *, exclude_solvent=False, exclude_ions=False,
                   frame_start=0, frame_stop=None, frame_stride=1, processes=1,
                   profile="default", ligand_smiles=None, protonate=False,
                   protonate_ph=7.4, rule_overrides=None, progress=None,
                   assembly_overrides=None, hydrophobic_enclosure=None, cancel_check=None):
    """Analyse scene-coordinate frames, keeping topology atom identity stable."""
    import numpy as np
    from chimerax.core.errors import UserError

    engine = require_engine()
    options = analysis_options(engine, profile=profile, rule_overrides=rule_overrides,
                               hydrophobic_enclosure=hydrophobic_enclosure,
                               assembly_overrides=assembly_overrides)
    settings = capture_analysis_options(engine, options, ligand_smiles=ligand_smiles,
        protonate=protonate, protonate_ph_min=protonate_ph, protonate_ph_max=protonate_ph,
        add_hydrogens=False, exclude_solvent=exclude_solvent, exclude_ions=exclude_ions,
        frame_start=frame_start, frame_stop=frame_stop, frame_stride=frame_stride,
        processes=processes, hydrophobic_enclosure=hydrophobic_enclosure)
    options.pop("include_raw")  # Trajectories retain assigned interactions only.
    check_cancel(cancel_check)
    if structure.num_coordsets < 2:
        raise UserError("#%s has a single frame; load a trajectory first." % structure.id_string)
    topo_path = save_models_mmcif(session, [structure])
    coords_handle = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    coords_handle.close()
    coords_path = coords_handle.name
    stack = None
    try:
        topo_atoms, _ = engine.parse_structure(topo_path)
        check_cancel(cancel_check)
        coord_index, name_index = _chimerax_atom_lookup(structure)
        index_of = {atom: i for i, atom in enumerate(structure.atoms)}
        topo_mapping = {}
        used_atoms = set()
        perm = []
        for rec in topo_atoms:
            atom = _resolve_atom(rec, coord_index, name_index)
            if atom is None or atom not in index_of or atom in used_atoms:
                raise UserError("Could not uniquely align topology atom %s to the trajectory." % rec.label)
            topo_mapping[rec.atom_id] = atom
            used_atoms.add(atom)
            perm.append(index_of[atom])
        coordset_ids = list(structure.coordset_ids)
        stack = np.lib.format.open_memmap(coords_path, mode="w+", dtype=np.float32,
                                         shape=(len(coordset_ids), len(topo_atoms), 3))
        saved_cs = structure.active_coordset_id
        try:
            for fi, cid in enumerate(coordset_ids):
                check_cancel(cancel_check)
                structure.active_coordset_id = cid
                stack[fi] = np.asarray([_scene_coord(a) for a in structure.atoms])[perm]
        finally:
            if not getattr(structure, "deleted", False):
                structure.active_coordset_id = saved_cs
        stack.flush()
        del stack
        stack = None

        def report_progress(done, total):
            check_cancel(cancel_check)
            if progress is not None:
                progress(done, total)
            check_cancel(cancel_check)

        check_cancel(cancel_check)
        result = engine.analyze_trajectory(engine.ArrayFrameSource(topo_path, coords_path),
            ligand_smiles=ligand_smiles or None, protonate=protonate,
            protonate_ph_min=protonate_ph, protonate_ph_max=protonate_ph,
            exclude_solvent=exclude_solvent, exclude_ions=exclude_ions,
            frame_start=frame_start, frame_stop=frame_stop, frame_stride=frame_stride,
            processes=processes, progress=report_progress, **options)
    finally:
        if stack is not None:
            del stack
        _unlink(topo_path)
        _unlink(coords_path)
    check_cancel(cancel_check)
    # The required engine preserves topology IDs when excluding water/ions.
    original = {a.atom_id: a for a in topo_atoms}
    id_to_atom = {}
    for rec in result.atoms:
        before = original.get(rec.atom_id)
        if before is None or (rec.component_id, rec.name, rec.element) != (
                before.component_id, before.name, before.element):
            raise UserError("The engine changed trajectory atom identity; update Chemur and retry.")
        id_to_atom[rec.atom_id] = topo_mapping[rec.atom_id]
    rec_by_id = {a.atom_id: a for a in result.atoms}
    features_by_id = {f.feature_id: f for f in engine.perceive_features(result.atoms, result.components)}
    # Compatibility unions are useful for a whole-trajectory selection. Per-frame
    # filters MUST use frame_participants/frame_endpoints below.
    key_to_atoms, key_to_endpoints = {}, {}
    for frame in result.frames:
        for inter in frame.interactions:
            key = result.interaction_key(inter)
            values = key_to_atoms.setdefault(key, [])
            for aid in inter.atom_ids:
                atom = id_to_atom.get(aid)
                if atom is not None and atom not in values:
                    values.append(atom)
            groups = _interaction_groups(inter, rec_by_id, id_to_atom, features_by_id)
            existing = key_to_endpoints.setdefault(key, [[] for _ in groups])
            for index, group in enumerate(groups):
                if index == len(existing):
                    existing.append([])
                for atom in group:
                    if atom not in existing[index]:
                        existing[index].append(atom)
    frame_coordset_ids = [coordset_ids[i] for i in result.frame_indices]
    session.logger.info("Chemur trajectory: analysed %d frames of #%s; %d distinct interactions." %
                       (result.n_frames, structure.id_string, len(result.occupancy())))
    return TrajectoryRun(result, key_to_atoms, key_to_endpoints, structure,
                         frame_coordset_ids, rec_by_id, features_by_id, id_to_atom,
                         analysis_options=settings)


def draw_frame_interactions(session, run, frame_index, *, included_types=None,
                            selected_only=False, skip_biopolymer_internal=False,
                            cancel_check=None):
    check_cancel(cancel_check)
    frame = run._frames.get(frame_index)
    if frame is None:
        return []
    graphics = []
    rows = _draw_interactions(session, run.structure, frame.interactions,
        run.rec_by_id, run.id_to_atom, run.features_by_id, included_types=included_types,
        selected_only=selected_only, skip_biopolymer_internal=skip_biopolymer_internal,
        graphics=graphics, cancel_check=cancel_check)
    _discard_graphics(session, run.graphics)
    run.graphics[:] = graphics
    run.rows = rows
    _track_graphics(session, run.source_models, run.graphics)
    return rows


def _build_ligand_report(result):
    """Report validated/unresolved chemical ligands, including connected assemblies."""
    metadata = getattr(result, "metadata", {})
    result_components = getattr(result, "components", ())
    inputs = metadata.get("input_ligand_smiles") or {}
    registry = dict(metadata.get("ligand_assemblies") or {})
    registry.update(metadata.get("chemical_products") or {})
    for component in result_components:
        record = component.metadata.get("chemical_product") or component.metadata.get("ligand_assembly")
        if isinstance(record, dict) and record.get("assembly_id"):
            registry.setdefault(record["assembly_id"], record)
    components = {c.component_id: c for c in result_components}
    atoms = {a.atom_id: a for a in getattr(result, "atoms", ())}
    covered = set()
    report = []
    for assembly_id, record in registry.items():
        member_ids = tuple(record.get("component_ids", ()))
        members = [components[cid] for cid in member_ids if cid in components]
        if not members:
            continue
        covered.update(member_ids)
        owner = members[0]
        failed = record.get("status") != "validated"
        ids = record.get("atom_ids", ())
        diagnostics = record.get("diagnostics", ())
        reason = record.get("reason") or record.get("error") or "; ".join(map(str, diagnostics))
        if failed and not reason:
            reason = owner.metadata.get("ignored_reason", "Assembly chemistry is unresolved")
        # Connected or covalently bound products require a complete assembly
        # override, not a name-based monomer SMILES edited in a single cell.
        editable = (len(members) == 1 and not record.get("attachment_component_ids")
                    and record.get("product_kind", "ligand") == "ligand")
        report.append(LigandProtonation(
            assembly_id, "+".join(c.name for c in members), owner.chain_id,
            ", ".join(c.residue_id for c in members), "failed" if failed else "ok",
            record.get("smiles"), inputs.get(assembly_id.upper()) or inputs.get(owner.component_id.upper())
            or (inputs.get(owner.name.upper()) if len(members) == 1 else None),
            0 if failed else sum(atoms[aid].formal_charge for aid in ids
                                if aid in atoms and not atoms[aid].metadata.get("ignored_by_template")),
            reason, assembly_id=assembly_id, component_ids=member_ids, editable=editable,
            chemistry_source=record.get("chemistry_source"), diagnostics=diagnostics))
    for component in result_components:
        if component.molecule_type != "ligand" or component.component_id in covered:
            continue
        meta = component.metadata
        failed = bool(meta.get("ignored_by_template"))
        report.append(LigandProtonation(
            component.component_id, component.name, component.chain_id, component.residue_id,
            "failed" if failed else "ok", meta.get("smiles"),
            inputs.get(component.component_id.upper()) or inputs.get(component.name.upper()),
            0 if failed else sum(atoms[aid].formal_charge for aid in component.atom_ids
                                if aid in atoms and not atoms[aid].metadata.get("ignored_by_template")),
            meta.get("ignored_reason", ""), chemistry_source=meta.get("template_source")))
    report.sort(key=lambda r: (r.status != "failed", r.chain_id or "", r.residue_id))
    return report


def _is_biopolymer_internal(ep1, ep2):
    return (
        (ep1.molecule_type == "protein" and ep2.molecule_type == "protein")
        or (ep1.molecule_type == "nucleotide" and ep2.molecule_type == "nucleotide")
    )


def _geometry_extra(inter):
    return deepcopy(inter.metadata)


_COORD_EPS = 0.05


def _scene_coord(atom):
    # The fallback supports lightweight test objects without a model transform.
    return atom.scene_coord if hasattr(atom, "scene_coord") else atom.coord


def _coord_key(coord):
    return tuple(round(float(v) * 1000) for v in coord)


def _sq_dist(a, b):
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def _chimerax_atom_lookup(structure):
    return _chimerax_atom_lookup_multi([structure])


def _chimerax_atom_lookup_multi(structures):
    """Keep collision buckets, in the scene frame used by all engine inputs."""
    coord_index, name_index = {}, {}
    for structure in structures:
        for atom in structure.atoms:
            residue = atom.residue
            residue_id = "%d%s" % (residue.number, (residue.insertion_code or "").strip())
            name = (residue.chain_id or "_", residue_id, residue.name, atom.name)
            coord_index.setdefault(_coord_key(_scene_coord(atom)), []).append(atom)
            name_index.setdefault(name, []).append(atom)
    return coord_index, name_index


def _resolve_atom(rec, coord_index, name_index):
    """Reject ambiguous matches instead of silently choosing a copy or altloc."""
    candidates = list(coord_index.get(_coord_key(rec.coord), ()))
    name = (rec.chain_id, rec.residue_id, rec.residue_name, rec.name)
    for atom in name_index.get(name, ()):
        if atom not in candidates:
            candidates.append(atom)
    candidates = [atom for atom in candidates
                  if _sq_dist(_scene_coord(atom), rec.coord) <= _COORD_EPS ** 2
                  and (not hasattr(rec, "element") or not hasattr(atom, "element")
                       or atom.element.name.upper() == rec.element.upper())]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    named = [atom for atom in candidates if atom in name_index.get(name, ())]
    if len(named) == 1:
        return named[0]
    return None


def _choose_endpoints(inter, rec_by_id, id_to_atom, features_by_id):
    """Pick the two partner :class:`Endpoint`s for an interaction.

    Respect the engine's measured salt-bridge pair and the two feature roles of
    an amide–π contact. Other contacts prefer atoms from different components,
    attaching ring membership where needed for ring-centroid drawing.
    """
    if inter.interaction_type == "salt_bridge":
        atom_ids = inter.metadata.get("distance_atom_ids")
        if not atom_ids or len(atom_ids) != 2:
            return None
        endpoints = []
        for aid in atom_ids:
            rec, atom = rec_by_id.get(aid), id_to_atom.get(aid)
            if rec is None or atom is None:
                return None
            endpoints.append(_make_endpoint(rec, atom))
        if endpoints[0].atom is endpoints[1].atom:
            return None
        return tuple(endpoints)
    if inter.interaction_type == "amide_pi":
        return _amide_pi_endpoints(inter, rec_by_id, id_to_atom, features_by_id)

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


def _amide_pi_features(inter, features_by_id):
    features = [features_by_id[fid] for fid in inter.feature_ids if fid in features_by_id]
    amides = [feat for feat in features if feat.feature_type == "amide"]
    rings = [feat for feat in features if feat.feature_type == "ring"]
    return (amides[0], rings[0]) if len(amides) == len(rings) == 1 else None


def _amide_pi_endpoints(inter, rec_by_id, id_to_atom, features_by_id):
    features = _amide_pi_features(inter, features_by_id)
    if features is None:
        return None
    amide, ring = features
    center_ids = [amide.metadata.get(key) for key in ("carbon_atom_id", "nitrogen_atom_id")]
    if (len(set(center_ids)) != 2 or any(aid not in amide.atom_ids for aid in center_ids)
            or len(ring.atom_ids) < 3):
        return None
    # Partial centers would produce a line with a different distance from the
    # engine measurement, so omit a contact whose geometry cannot be mapped.
    if any(aid not in id_to_atom or aid not in rec_by_id
           for aid in (*center_ids, *ring.atom_ids)):
        return None
    center_atoms = [id_to_atom[aid] for aid in center_ids]
    if center_atoms[0] is center_atoms[1]:
        return None
    ep1 = _make_endpoint(rec_by_id[center_ids[0]], center_atoms[0])
    ep1.center_atoms = center_atoms
    ring_rec = rec_by_id[ring.atom_ids[0]]
    ep2 = _make_endpoint(ring_rec, id_to_atom[ring_rec.atom_id])
    _attach_ring(ep2, ring_rec, [ring], id_to_atom)
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
        cx, cy, cz = (sum(_scene_coord(a)[i] for a in atoms) / len(atoms) for i in range(3))
        endpoint.atom = min(
            atoms,
            key=lambda a: _sq_dist(_scene_coord(a), (cx, cy, cz)),
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
    """Create the pseudobond, routing geometric centers to marker atoms."""
    node1 = _node_for(ep1, style, holder, session, structure)
    node2 = _node_for(ep2, style, holder, session, structure)
    if node1 is None or node2 is None or node1 is node2:
        return None
    cross_model = getattr(ep1.atom, "structure", structure) is not getattr(ep2.atom, "structure", structure)
    group = pi_pbg if (ep1.ring_atoms or ep2.ring_atoms or ep1.center_atoms
                       or ep2.center_atoms or cross_model) else pbg
    try:
        pb = group.new_pseudobond(node1, node2)
    except Exception:
        return None
    pb.color = style["color"]
    pb.radius = style["radius"]
    return pb


def _node_for(endpoint, style, holder, session, structure):
    """Center markers belong to their atoms' model, in that model's local frame."""
    if not (endpoint.center_atoms or endpoint.ring_atoms):
        return endpoint.atom
    return _ring_marker(endpoint, style, holder, session,
                        getattr(endpoint.atom, "structure", structure), holder.get("ms_name", _PI_MARKERSET_NAME))


def _ring_marker(endpoint, style, holder, session, structure, name):
    atoms = endpoint.center_atoms or endpoint.ring_atoms
    key = frozenset(atoms)
    markers = holder.setdefault("markers", {})
    if key in markers:
        return markers[key]
    models = holder.setdefault("ms", {})
    ms = models.get(structure)
    if ms is None:
        from chimerax.markers import MarkerSet
        ms = MarkerSet(session, name=name)
        structure.add([ms])
        models[structure] = ms
        if holder.get("graphics") is not None:
            holder["graphics"].append(ms)
    center = tuple(sum(_scene_coord(a)[i] for a in atoms) / len(atoms)
                   for i in range(3))
    if hasattr(structure, "scene_position"):
        center = structure.scene_position.inverse() * center
    else:
        center = tuple(sum(a.coord[i] for a in atoms) / len(atoms)
                       for i in range(3))
    from chimerax.atomic import Element
    marker = ms.create_marker(center, style["color"], _PI_MARKER_RADIUS)
    marker.element = Element.get_element(_PI_MARKER_ELEMENT)
    markers[key] = marker
    return marker


def _pose_markerset_name(group_name):
    return "%s π-centres" % group_name


def _pose_node_for(endpoint, style, holder, session):
    if not (endpoint.center_atoms or endpoint.ring_atoms):
        return endpoint.atom
    return _ring_marker(endpoint, style, holder, session, endpoint.atom.structure,
                        holder["ms_name"])


def draw_pose_rows(session, rows, group_name, *, graphics=None, cancel_check=None):
    """Draw one pose's interactions as pseudobonds in a named GLOBAL group.

    ``rows`` are :class:`InteractionRow`s from a pose comparison; their endpoint
    atoms are live ChimeraX atoms spanning the receptor + pose models, so a global
    (cross-model) pseudobond group is required. Each draw owns unique handles;
    the caller retires the previous drawing after success. Returns the group.
    """
    check_cancel(cancel_check)
    unique_name = "%s (%d)" % (group_name, next(_DRAW_IDS))
    pbg = session.pb_manager.get_group(unique_name, create=True)
    holder = {"ms_name": _pose_markerset_name(unique_name), "graphics": graphics}
    if graphics is not None:
        graphics[:] = [pbg]
    for row in rows:
        check_cancel(cancel_check)
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
    if graphics is not None:
        _track_graphics(session, tuple({ep.atom.structure for row in rows
                                      for ep in (row.endpoint1, row.endpoint2)}), graphics)
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


def _log_summary(session, structure, counts, drawn, unresolved, filtered, *, enclosure_groups=0):
    logger = session.logger
    if not drawn:
        logger.warning("Chemur: no interactions to show for #%s." % structure.id_string)
        return
    parts = ", ".join("%s: %d" % (k, counts[k]) for k in sorted(counts))
    msg = "Chemur drew %d contacts on #%s (%s)." % (drawn, structure.id_string, parts)
    if enclosure_groups:
        msg += " %d hydrophobic enclosure group(s), %d enclosure contact(s)." % (
            enclosure_groups, counts.get("hydrophobic_enclosure", 0))
    if filtered:
        msg += " %d hidden by filters." % filtered
    if unresolved:
        msg += " %d skipped (atoms could not be mapped)." % unresolved
    logger.info(msg)

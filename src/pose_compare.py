"""Compare Chemur interactions of several docked ligand poses against one receptor.

For each selected pose model, runs the engine on the pose (merged with the
receptor when the pose is ligand-only, or standalone when it already contains the
protein) and collects, per protein residue, which interaction type(s) that pose's
ligand makes there. :class:`PoseComparison` is the residue (column) x pose (row)
matrix that :mod:`pose_plot` renders as an MSA-like grid.

The matrix assembly (:func:`build_matrix`) and the contiguous sequence-window
column layout (:func:`build_columns`) are pure functions so they can be unit
tested without ChimeraX or the engine.
"""

from __future__ import annotations

import os
import re
from collections import namedtuple

from .lifecycle import RunCancelled

# One protein-residue contact made by one pose. ``number`` is the integer residue
# number used for ordering/columns (None when the residue id has no leading
# digits); ``residue_id`` keeps the full id (e.g. "133A") for labels; ``row`` is
# the :class:`~.runner.InteractionRow` for 3D selection, or None for engine-only
# (unmapped) poses.
Hit = namedtuple("Hit", ["chain_id", "number", "residue_id", "resname", "itype", "row"])

_NUM_RE = re.compile(r"^\s*(-?\d+)")

# Minimal three-to-one residue code map (standard amino acids); anything else
# falls back to "X". Kept local so build_columns stays pure/testable.
_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def one_letter(resname):
    """Single-letter code for a residue name, or "X" if unknown."""
    return _AA3TO1.get((resname or "").upper(), "X")


def _resnum(residue_id):
    """Leading integer of a residue id ("133", "133A" -> 133); None if non-numeric."""
    m = _NUM_RE.match(str(residue_id))
    return int(m.group(1)) if m else None


def _protein_ligand_endpoints(row):
    """``(protein_ep, ligand_ep)`` for a protein–ligand row, else ``None``.

    The partner must be ``molecule_type == "ligand"`` so the figure shows only the
    docked ligand's own contacts; protein–protein, protein–solvent (e.g. the
    receptor's crystallographic waters) and protein–ion rows are excluded.
    """
    e1, e2 = row.endpoint1, row.endpoint2
    if e1.molecule_type == "protein" and e2.molecule_type == "ligand":
        return e1, e2
    if e2.molecule_type == "protein" and e1.molecule_type == "ligand":
        return e2, e1
    return None


class ResidueColumn:
    """One column of the MSA grid: a protein residue (or context residue)."""

    __slots__ = ("chain_id", "number", "residue_id", "resname", "key", "label1", "label3",
                 "is_context")

    def __init__(self, chain_id, number, resname, is_context):
        self.chain_id = chain_id
        self.residue_id = str(number)
        self.number = _resnum(self.residue_id)
        self.resname = resname
        self.key = (chain_id, self.residue_id)
        self.label1 = "%s%s" % (one_letter(resname), self.residue_id)
        self.label3 = "%s%s" % ((resname or "?").title(), self.residue_id)
        self.is_context = is_context


class PoseRow:
    """One row of the MSA grid: a docked pose."""

    __slots__ = ("pose", "name", "mapping_status", "analysis_status", "diagnostic")

    def __init__(self, pose, name, fully_mapped=False, *, mapping_status=None,
                 analysis_status="success", diagnostic=""):
        self.pose = pose
        self.name = name
        self.mapping_status = mapping_status or ("complete" if fully_mapped else "unavailable")
        if self.mapping_status == "none":
            self.mapping_status = "unavailable"
        self.analysis_status = analysis_status
        self.diagnostic = diagnostic

    @property
    def fully_mapped(self):
        return self.mapping_status == "complete"

    @property
    def eligible_for_consensus(self):
        return self.analysis_status == "success"

    @property
    def display_name(self):
        status = self.analysis_status
        if status != "success":
            return "%s [%s]" % (self.name, status)
        if self.mapping_status != "complete":
            return "%s [%s]" % (self.name, "partial 3D" if self.mapping_status == "partial" else "no 3D")
        return self.name


class PoseComparison:
    """Residue (column) x pose (row) interaction matrix.

    ``columns`` are protein residues shown as a contiguous sequence window;
    ``rows`` are the analysed poses (selection order). ``cell[(row_index,
    col_key)]`` is the set of interaction types that pose makes with that residue;
    ``cell_rows[(row_index, col_key)]`` are the backing InteractionRows (for 3D
    selection; empty for engine-only poses). ``types`` is every interaction type
    present (legend + per-type filter). The cached inputs let the column window be
    rebuilt for a different ``context`` without re-running the engine.
    """

    __slots__ = ("columns", "rows", "cell", "cell_rows", "types", "receptor_name",
                 "context", "n_untemplated", "_reference_index", "_resname_by_key",
                 "_interacting", "receptor", "native_runs", "graphics")

    def __init__(self, rows, cell, cell_rows, types, receptor_name,
                 reference_index, resname_by_key, interacting, context,
                 n_untemplated=0, *, receptor=None, native_runs=None):
        self.rows = rows
        self.cell = cell
        self.cell_rows = cell_rows
        self.types = types
        self.receptor_name = receptor_name
        self.n_untemplated = n_untemplated
        self._reference_index = reference_index
        self._resname_by_key = resname_by_key
        self._interacting = interacting
        self.context = context
        self.receptor = receptor
        self.native_runs = dict(native_runs or {})
        self.graphics = []
        self.columns = build_columns(interacting, resname_by_key,
                                     reference_index, context)

    def rebuild_columns(self, context):
        """Recompute the column window for a new ± context (no engine re-run)."""
        self.context = context
        self.columns = build_columns(self._interacting, self._resname_by_key,
                                     self._reference_index, context)

    @property
    def source_models(self):
        models = [] if self.receptor is None else [self.receptor]
        for row in self.rows:
            if not any(row.pose is model for model in models):
                models.append(row.pose)
        return tuple(models)

    def index_for_pose(self, pose):
        return next((i for i, row in enumerate(self.rows) if row.pose is pose), None)

    def prune_models(self, models):
        """Remove closed poses without re-running chemistry; False invalidates all."""
        removed = tuple(models)
        if self.receptor is not None and any(self.receptor is model for model in removed):
            return False
        kept = [i for i, row in enumerate(self.rows)
                if not any(row.pose is model for model in removed)]
        if not kept:
            return False
        index_map = {old: new for new, old in enumerate(kept)}
        self.rows = [self.rows[i] for i in kept]
        self.cell = {(index_map[i], key): types for (i, key), types in self.cell.items()
                     if i in index_map}
        self.cell_rows = {(index_map[i], key): rows for (i, key), rows in self.cell_rows.items()
                          if i in index_map}
        self.native_runs = {pose: run for pose, run in self.native_runs.items()
                            if not any(pose is model for model in removed)}
        self._interacting = {key for _, key in self.cell}
        self.types = sorted({itype for types in self.cell.values() for itype in types})
        self.n_untemplated = sum(row.analysis_status == "unresolved" for row in self.rows)
        self.rebuild_columns(self.context)
        return True


# ----- pure assembly (unit-testable, no ChimeraX) --------------------------

def build_matrix(pose_metas, per_pose_hits):
    """Assemble the cell matrix from per-pose hits.

    ``pose_metas`` is a list of ``(pose, name, fully_mapped)``; ``per_pose_hits``
    a parallel list of ``[Hit, ...]``. Returns ``(rows, cell, cell_rows,
    resname_by_key, types, interacting)``.
    """
    rows = [meta if isinstance(meta, PoseRow) else PoseRow(*meta) for meta in pose_metas]
    cell = {}
    cell_rows = {}
    resname_by_key = {}
    interacting = set()
    types = set()
    for idx, hits in enumerate(per_pose_hits):
        if not rows[idx].eligible_for_consensus:
            continue
        for hit in hits:
            key = (hit.chain_id, str(hit.residue_id))
            cell.setdefault((idx, key), set()).add(hit.itype)
            if hit.row is not None:
                cell_rows.setdefault((idx, key), []).append(hit.row)
            resname_by_key.setdefault(key, hit.resname)
            interacting.add(key)
            types.add(hit.itype)
    return rows, cell, cell_rows, resname_by_key, sorted(types), interacting


def build_columns(interacting, resname_by_key, reference_index, context):
    """Build the contiguous sequence-window columns.

    ``interacting`` is a set of ``(chain_id, number)`` residues some pose touches;
    ``resname_by_key`` maps those keys to residue names; ``reference_index`` maps
    ``(chain_id, number) -> resname`` for the whole receptor (so flanking context
    residues can be filled and named). For each interacting residue ``n`` every
    residue in ``[n-context, n+context]`` that exists is added (marked context if
    not itself interacting); the result is ordered by ``(chain_id, number)``.
    """
    interacting = {(chain, str(rid)) for chain, rid in interacting}
    resname_by_key = {(chain, str(rid)): name for (chain, rid), name in resname_by_key.items()}
    reference_index = {(chain, str(rid)): name for (chain, rid), name in reference_index.items()}
    display = set(interacting)
    if context:
        for chain, rid in interacting:
            number = _resnum(rid)
            if number is None:
                continue
            for key in reference_index:
                candidate = _resnum(key[1])
                if key[0] == chain and candidate is not None and abs(candidate - number) <= context:
                    display.add(key)

    def order(key):
        number = _resnum(key[1])
        return (key[0], number is None, number or 0, key[1])

    columns = []
    for chain, residue_id in sorted(display, key=order):
        key = (chain, residue_id)
        resname = resname_by_key.get(key) or reference_index.get(key) or "UNK"
        columns.append(ResidueColumn(chain, residue_id, resname, is_context=key not in interacting))
    return columns


def compute_consensus(comparison, include_types=None, threshold=0.5):
    """Per-residue consensus interactions across the poses.

    For each residue column, counts the fraction of poses making each interaction
    type and keeps those at or above ``threshold`` (0–1). Returns
    ``{col_key: {itype: fraction}}`` (only residues/types meeting the threshold).
    Pure: depends only on ``comparison.cell`` and ``comparison.rows``.
    """
    eligible = {i for i, row in enumerate(comparison.rows) if row.eligible_for_consensus}
    n = len(eligible)
    if n == 0:
        return {}
    counts = {}  # col_key -> {itype: pose_count}
    for (pose_idx, col_key), types in comparison.cell.items():
        if pose_idx not in eligible:
            continue
        for t in types:
            if include_types is not None and t not in include_types:
                continue
            counts.setdefault(col_key, {})
            counts[col_key][t] = counts[col_key].get(t, 0) + 1
    consensus = {}
    for col_key, by_type in counts.items():
        keep = {t: c / n for t, c in by_type.items() if c / n >= threshold}
        if keep:
            consensus[col_key] = keep
    return consensus


# ----- engine orchestration (ChimeraX) -------------------------------------

def _reference_index(receptor):
    """Map ``(chain_id, number) -> residue_name`` for every receptor residue."""
    index = {}
    if receptor is None:
        return index
    for r in receptor.residues:
        insertion = str(getattr(r, "insertion_code", "") or "").strip()
        index.setdefault((r.chain_id or "_", "%s%s" % (r.number, insertion)), r.name)
    return index


def _hits_from_rows(rows):
    """Protein-residue hits from mapped InteractionRows (3D-selectable)."""
    hits = []
    for row in rows:
        eps = _protein_ligand_endpoints(row)
        if eps is None:
            continue
        ep = eps[0]
        hits.append(Hit(ep.chain_id or "_", _resnum(ep.residue_id), ep.residue_id,
                        ep.residue_name, row.interaction_type, row))
    return hits


def _hits_from_result(result, mapped_rows=(), pose_component_ids=None):
    """Native projected contacts determine every cell, irrespective of mapping.

    Mapped rows only add 3D selection handles; missing rows cannot remove native
    contacts. ``pose_component_ids`` scopes separate poses independently of
    receptor cofactors. Projection occurs before protein/ligand filtering so
    shell atoms never appear to interact merely by sharing an enclosure.
    """
    from chemur.interaction_projection import project_interactions_for_scoring

    def interaction_key(interaction):
        return (interaction.interaction_type, tuple(sorted(interaction.atom_ids)),
                getattr(interaction, "assignment_order", None))

    def touches_protein_residue(row, key):
        for endpoint in (row.endpoint1, row.endpoint2):
            if endpoint.molecule_type != "protein":
                continue
            if (endpoint.chain_id or "_", str(endpoint.residue_id)) == key:
                return True
            # A peptide amide midpoint can span two residues. Both native
            # residue cells must retain the same selectable mapped contact.
            for atom in getattr(endpoint, "center_atoms", None) or ():
                residue = atom.residue
                insertion = str(getattr(residue, "insertion_code", "") or "").strip()
                if (residue.chain_id or "_", "%s%s" % (residue.number, insertion)) == key:
                    return True
        return False

    by_interaction = {}
    for row in mapped_rows:
        interaction = getattr(row, "interaction", None)
        if interaction is not None:
            by_interaction.setdefault(interaction_key(interaction), []).append(row)
    allowed = None if pose_component_ids is None else set(pose_component_ids)
    rec_by_id = {rec.atom_id: rec for rec in result.atoms}
    hits = []
    for inter in project_interactions_for_scoring(result.interactions):
        recs = [rec_by_id[a] for a in inter.atom_ids if a in rec_by_id]
        prot = [r for r in recs if r.molecule_type == "protein"]
        has_ligand = any(r.molecule_type == "ligand"
                         and (allowed is None or r.component_id in allowed) for r in recs)
        if not prot or not has_ligand:
            continue
        seen = set()
        for r in prot:
            key = (r.chain_id or "_", str(r.residue_id))
            if key in seen:
                continue
            seen.add(key)
            backing = next((row for row in by_interaction.get(interaction_key(inter), ())
                            if touches_protein_residue(row, key)), None)
            hits.append(Hit(r.chain_id or "_", _resnum(r.residue_id), r.residue_id,
                            r.residue_name, inter.interaction_type, backing))
    return hits


def _pose_analysis_status(result, pose_component_ids):
    """Resolve scientific status from native pose provenance, never 3D mapping.

    An identified, typed ligand with no contacts is a valid negative result.
    Missing target provenance, absent ligands and untyped targets are unresolved.
    Receptor cofactors do not affect a separate pose's status.
    """
    if pose_component_ids is None:
        return "unresolved", "Could not identify the pose's native ligand components"
    targets = set(pose_component_ids)
    if not targets:
        return "unresolved", "No native ligand components were found for the pose"
    skipped = set((getattr(result, "metadata", {}) or {}).get("skipped_ligands") or {})
    skipped.update(component.component_id for component in getattr(result, "components", ())
                   if (getattr(component, "metadata", {}) or {}).get("ignored_by_template"))
    unresolved = targets.intersection(skipped)
    if unresolved:
        return "unresolved", "Pose ligand templates were unresolved: %s" % ", ".join(sorted(unresolved))
    present = {atom.component_id for atom in result.atoms
               if atom.molecule_type == "ligand" and atom.element != "H"
               and not (getattr(atom, "metadata", {}) or {}).get("ignored_by_template")}
    missing = targets.difference(present)
    if missing:
        return "unresolved", "Pose ligand components have no resolved heavy atoms: %s" % ", ".join(sorted(missing))
    return "success", ""


def compute_pose_comparison(session, receptor, poses, run_kwargs, *,
                            context=0, progress=None, cancel_check=None):
    """Run each pose against ``receptor`` and assemble a :class:`PoseComparison`.

    ``run_kwargs`` are the analysis options the GUI already builds for the other
    panels (protonate, protonate_ph, ligand_smiles, add_hydrogens, rule_overrides,
    skip_biopolymer_internal); ``selected_only`` is ignored for poses. ``context``
    adds that many flanking residues on each side of every interacting residue.
    ``progress(done, total)`` is called after each pose.
    """
    from .runner import run_pose_interactions, save_models_mmcif, _addh_and_clear
    from .engine_support import require_engine

    require_engine()
    kwargs = dict(run_kwargs)
    kwargs.pop("selected_only", None)
    cancel_check = cancel_check or kwargs.pop("cancel_check", None)
    kwargs.pop("cancel_check", None)
    def check():
        if cancel_check is not None and cancel_check():
            raise RunCancelled("Analysis cancelled")
    def report_progress(done, total):
        if progress is not None:
            progress(done, total)
        check()
    check()
    add_hydrogens = kwargs.get("add_hydrogens", True)

    # Save the receptor mmCIF once (add H once) and reuse it for every SDF-route
    # pose, rather than re-saving the (possibly large) receptor per pose.
    receptor_path = None
    if receptor is not None:
        try:
            _addh_and_clear(session, [receptor], add_hydrogens,
                            cancel_check=cancel_check)
            check()
            receptor_path = save_models_mmcif(session, [receptor])
        except RunCancelled:
            raise
        except Exception as e:
            session.logger.warning("Chemur poses: could not pre-save receptor (%s)" % e)

    pose_metas = []
    per_pose_hits = []
    n_untemplated = 0
    native_runs = {}
    total = len(poses)
    try:
        for i, pose in enumerate(poses):
            check()
            name = "#%s %s" % (pose.id_string, pose.name)
            try:
                prun = run_pose_interactions(
                    session, pose, receptor, draw=False,
                    receptor_path=receptor_path, cancel_check=cancel_check, **kwargs)
                check()
            except RunCancelled:
                raise
            except Exception as e:
                session.logger.warning("Chemur poses: skipped %s (%s)" % (name, e))
                pose_metas.append(PoseRow(pose, name, analysis_status="failed", diagnostic=str(e)))
                per_pose_hits.append([])
                report_progress(i + 1, total)
                continue

            native_runs[pose] = prun
            pose_component_ids = getattr(prun, "pose_component_ids", None)
            analysis_status, diagnostic = _pose_analysis_status(prun.result, pose_component_ids)
            if analysis_status == "unresolved":
                n_untemplated += 1
                session.logger.warning("Chemur poses: %s: %s." % (name, diagnostic))

            hits = (_hits_from_result(prun.result, prun.rows, pose_component_ids)
                    if analysis_status == "success" else [])
            pose_metas.append(PoseRow(
                pose, name, prun.fully_mapped,
                mapping_status=getattr(prun, "mapping_status", None),
                analysis_status=analysis_status, diagnostic=diagnostic,
            ))
            per_pose_hits.append(hits)
            report_progress(i + 1, total)
    finally:
        if receptor_path is not None:
            try:
                os.unlink(receptor_path)
            except OSError:
                pass

    check()
    rows, cell, cell_rows, resname_by_key, types, interacting = build_matrix(
        pose_metas, per_pose_hits)
    reference_index = _reference_index(receptor)
    receptor_name = "#%s %s" % (receptor.id_string, receptor.name) if receptor else "receptor"
    return PoseComparison(
        rows, cell, cell_rows, types, receptor_name,
        reference_index, resname_by_key, interacting, context,
        n_untemplated=n_untemplated, receptor=receptor, native_runs=native_runs)

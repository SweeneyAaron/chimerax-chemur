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

    __slots__ = ("chain_id", "number", "resname", "key", "label1", "label3",
                 "is_context")

    def __init__(self, chain_id, number, resname, is_context):
        self.chain_id = chain_id
        self.number = number
        self.resname = resname
        self.key = (chain_id, number)
        self.label1 = "%s%d" % (one_letter(resname), number)
        self.label3 = "%s%d" % ((resname or "?").title(), number)
        self.is_context = is_context


class PoseRow:
    """One row of the MSA grid: a docked pose."""

    __slots__ = ("pose", "name", "fully_mapped")

    def __init__(self, pose, name, fully_mapped):
        self.pose = pose
        self.name = name
        self.fully_mapped = fully_mapped


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
                 "_interacting")

    def __init__(self, rows, cell, cell_rows, types, receptor_name,
                 reference_index, resname_by_key, interacting, context,
                 n_untemplated=0):
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
        self.columns = build_columns(interacting, resname_by_key,
                                     reference_index, context)

    def rebuild_columns(self, context):
        """Recompute the column window for a new ± context (no engine re-run)."""
        self.context = context
        self.columns = build_columns(self._interacting, self._resname_by_key,
                                     self._reference_index, context)


# ----- pure assembly (unit-testable, no ChimeraX) --------------------------

def build_matrix(pose_metas, per_pose_hits):
    """Assemble the cell matrix from per-pose hits.

    ``pose_metas`` is a list of ``(pose, name, fully_mapped)``; ``per_pose_hits``
    a parallel list of ``[Hit, ...]``. Returns ``(rows, cell, cell_rows,
    resname_by_key, types, interacting)``.
    """
    rows = [PoseRow(pose, name, fully_mapped)
            for (pose, name, fully_mapped) in pose_metas]
    cell = {}
    cell_rows = {}
    resname_by_key = {}
    interacting = set()
    types = set()
    for idx, hits in enumerate(per_pose_hits):
        for hit in hits:
            if hit.number is None:
                continue  # cannot place a non-numeric residue on the sequence axis
            key = (hit.chain_id, hit.number)
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
    by_chain = {}
    for (chain, number) in interacting:
        by_chain.setdefault(chain, set()).add(number)

    columns = []
    for chain in sorted(by_chain):
        display = set()
        for n in by_chain[chain]:
            for d in range(n - context, n + context + 1):
                key = (chain, d)
                if d in by_chain[chain] or key in reference_index:
                    display.add(d)
        for number in sorted(display):
            key = (chain, number)
            resname = resname_by_key.get(key) or reference_index.get(key) or "UNK"
            columns.append(ResidueColumn(
                chain, number, resname, is_context=key not in interacting))
    return columns


def compute_consensus(comparison, include_types=None, threshold=0.5):
    """Per-residue consensus interactions across the poses.

    For each residue column, counts the fraction of poses making each interaction
    type and keeps those at or above ``threshold`` (0–1). Returns
    ``{col_key: {itype: fraction}}`` (only residues/types meeting the threshold).
    Pure: depends only on ``comparison.cell`` and ``comparison.rows``.
    """
    n = len(comparison.rows)
    if n == 0:
        return {}
    counts = {}  # col_key -> {itype: pose_count}
    for (_pose_idx, col_key), types in comparison.cell.items():
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
        index.setdefault((r.chain_id or "_", r.number), r.name)
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


def _hits_from_result(result):
    """Protein-residue hits read straight from the engine result (fallback).

    Used when a pose's atoms could not be mapped back to ChimeraX atoms: the
    figure still needs only residue identity + interaction type.
    """
    rec_by_id = {rec.atom_id: rec for rec in result.atoms}
    hits = []
    for inter in result.interactions:
        recs = [rec_by_id[a] for a in inter.atom_ids if a in rec_by_id]
        prot = [r for r in recs if r.molecule_type == "protein"]
        has_ligand = any(r.molecule_type == "ligand" for r in recs)
        if not prot or not has_ligand:
            continue
        seen = set()
        for r in prot:
            key = (r.chain_id or "_", _resnum(r.residue_id))
            if key in seen:
                continue
            seen.add(key)
            hits.append(Hit(r.chain_id or "_", _resnum(r.residue_id), r.residue_id,
                            r.residue_name, inter.interaction_type, None))
    return hits


def compute_pose_comparison(session, receptor, poses, run_kwargs, *,
                            context=0, progress=None):
    """Run each pose against ``receptor`` and assemble a :class:`PoseComparison`.

    ``run_kwargs`` are the analysis options the GUI already builds for the other
    panels (protonate, protonate_ph, ligand_smiles, add_hydrogens, rule_overrides,
    skip_biopolymer_internal); ``selected_only`` is ignored for poses. ``context``
    adds that many flanking residues on each side of every interacting residue.
    ``progress(done, total)`` is called after each pose.
    """
    from .runner import run_pose_interactions, save_models_mmcif, _addh_and_clear

    kwargs = dict(run_kwargs)
    kwargs.pop("selected_only", None)
    add_hydrogens = kwargs.get("add_hydrogens", True)

    # Save the receptor mmCIF once (add H once) and reuse it for every SDF-route
    # pose, rather than re-saving the (possibly large) receptor per pose.
    receptor_path = None
    if receptor is not None:
        try:
            _addh_and_clear(session, [receptor], add_hydrogens)
            receptor_path = save_models_mmcif(session, [receptor])
        except Exception as e:
            session.logger.warning("Chemur poses: could not pre-save receptor (%s)" % e)

    pose_metas = []
    per_pose_hits = []
    n_untemplated = 0
    total = len(poses)
    try:
        for i, pose in enumerate(poses):
            name = "#%s %s" % (pose.id_string, pose.name)
            try:
                prun = run_pose_interactions(
                    session, pose, receptor, draw=False,
                    receptor_path=receptor_path, **kwargs)
            except Exception as e:
                session.logger.warning("Chemur poses: skipped %s (%s)" % (name, e))
                pose_metas.append((pose, name, False))
                per_pose_hits.append([])
                if progress is not None:
                    progress(i + 1, total)
                continue

            skipped = prun.result.metadata.get("skipped_ligands") or {}
            if skipped:
                n_untemplated += 1
                session.logger.warning(
                    "Chemur poses: %s ligand(s) in %s could not be templated and "
                    "were skipped (%s)." % (len(skipped), name, ", ".join(sorted(skipped))))

            hits = _hits_from_rows(prun.rows) if prun.rows else _hits_from_result(prun.result)
            pose_metas.append((pose, name, prun.fully_mapped))
            per_pose_hits.append(hits)
            if progress is not None:
                progress(i + 1, total)
    finally:
        if receptor_path is not None:
            try:
                os.unlink(receptor_path)
            except OSError:
                pass

    rows, cell, cell_rows, resname_by_key, types, interacting = build_matrix(
        pose_metas, per_pose_hits)
    reference_index = _reference_index(receptor)
    receptor_name = "#%s %s" % (receptor.id_string, receptor.name) if receptor else "receptor"
    return PoseComparison(
        rows, cell, cell_rows, types, receptor_name,
        reference_index, resname_by_key, interacting, context,
        n_untemplated=n_untemplated)

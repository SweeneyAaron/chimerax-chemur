"""Compare predicted interactions between two ChimeraX structures.

Two pieces:

* :func:`residue_correspondence` builds a residue-to-residue map between two
  models using ChimeraX MatchMaker (sequence alignment of the polymer chains,
  with a name/number fallback for non-polymer residues such as ligands and
  ions). MatchMaker also superimposes model B onto model A in 3D as a side
  effect, which is convenient for inspecting the two models side by side.

* :func:`compute_difference` runs the ChemeleonX engine on both models and buckets
  the resulting interactions — restricted to *common* residues — into shared,
  only-in-A and only-in-B. Interactions are compared at residue+type level
  (atom identity is unreliable across models after protonation), keyed by the
  canonical residue labels taken from model A.
"""

from __future__ import annotations


def _res_label(residue):
    """Canonical, model-independent residue label, e.g. ``A:LYS:31``."""
    chain = residue.chain_id or "_"
    return "%s:%s:%s" % (chain, residue.name, residue.number)


class DifferenceResult:
    """Outcome of comparing two models' interactions.

    ``entries`` drives the diff heatmap: one dict per distinct interaction
    signature with keys ``label`` (residue-pair text), ``itype``,
    ``category`` ("shared" | "only_a" | "only_b") and ``row_a``/``row_b`` (a
    representative :class:`~.runner.InteractionRow` in each model, or ``None``).
    ``name_a``/``name_b`` are display names for the two models; the residue
    counts summarise the correspondence.
    """

    __slots__ = ("entries", "name_a", "name_b",
                 "n_common", "n_only_a", "n_only_b")

    def __init__(self, entries, name_a, name_b, n_common, n_only_a, n_only_b):
        self.entries = entries
        self.name_a = name_a
        self.name_b = name_b
        self.n_common = n_common
        self.n_only_a = n_only_a
        self.n_only_b = n_only_b

    @property
    def n_changed(self):
        return sum(1 for e in self.entries if e["category"] != "shared")


def _chain_pairs(struct_a, struct_b):
    """Pair chains sharing a chain id between the two structures."""
    b_by_id = {c.chain_id: c for c in struct_b.chains}
    return [(c, b_by_id[c.chain_id]) for c in struct_a.chains
            if c.chain_id in b_by_id]


def residue_correspondence(session, struct_a, struct_b):
    """Map residues of ``struct_a`` to ``struct_b``.

    Returns ``(canon, matched_a, matched_b)`` where ``canon`` maps a ChimeraX
    :class:`Residue` (from either model) to a canonical label string (always the
    model-A label of its matched pair), and ``matched_a``/``matched_b`` are the
    sets of matched residues in each model.

    Polymer residues are paired by MatchMaker sequence alignment; non-polymer
    residues (ligands/ions/solvent) by ``(name, number)`` equality.
    """
    from chimerax.match_maker.match import (
        match, defaults, CP_SPECIFIC_SPECIFIC, CP_BEST_BEST)

    canon = {}
    matched_a = set()
    matched_b = set()

    def pair(ra, rb):
        lab = _res_label(ra)
        canon[ra] = lab
        canon[rb] = lab
        matched_a.add(ra)
        matched_b.add(rb)

    # --- polymer chains via sequence alignment ---------------------------
    pairs = _chain_pairs(struct_a, struct_b)
    kw = dict(show_alignment=False, verbose=None, ss_fraction=False)
    try:
        if pairs:
            results = match(session, CP_SPECIFIC_SPECIFIC, pairs,
                            defaults["matrix"], defaults["alignment_algorithm"],
                            defaults["gap_open"], defaults["gap_extend"], **kw)
        else:
            results = match(session, CP_BEST_BEST, (struct_a, [struct_b]),
                            defaults["matrix"], defaults["alignment_algorithm"],
                            defaults["gap_open"], defaults["gap_extend"], **kw)
    except Exception as e:
        # No polymer chains, or alignment failed: rely on the name/number
        # fallback below (covers e.g. a bare ligand or two ion sites).
        session.logger.info("ChemeleonX compare: sequence alignment skipped (%s)" % e)
        results = []

    for r in results:
        s1 = r["aligned ref seq"]
        s2 = r["aligned match seq"]
        for ra, rb in zip(s1.residues, s2.residues):
            if ra is not None and rb is not None:
                pair(ra, rb)

    # --- non-polymer residues by (name, number) --------------------------
    b_index = {}
    for rb in struct_b.residues:
        if rb in matched_b:
            continue
        b_index.setdefault((rb.name, rb.number), []).append(rb)
    for ra in struct_a.residues:
        if ra in matched_a:
            continue
        bucket = b_index.get((ra.name, ra.number))
        if bucket:
            pair(ra, bucket.pop(0))

    return canon, matched_a, matched_b


def compute_difference(session, struct_a, struct_b, run_kwargs,
                       name_a=None, name_b=None):
    """Run ChemeleonX on both models and classify interactions over common residues."""
    from .runner import run_interactions

    name_a = name_a or ("#%s" % struct_a.id_string)
    name_b = name_b or ("#%s" % struct_b.id_string)

    canon, matched_a, matched_b = residue_correspondence(session, struct_a, struct_b)

    run_a = run_interactions(session, struct_a, **run_kwargs)
    run_b = run_interactions(session, struct_b, **run_kwargs)

    def signature(row):
        c1 = canon.get(row.endpoint1.atom.residue)
        c2 = canon.get(row.endpoint2.atom.residue)
        if c1 is None or c2 is None:
            return None  # touches a residue with no counterpart -> not comparable
        return (row.interaction_type, frozenset((c1, c2)))

    def index(rows):
        out = {}
        for row in rows:
            sig = signature(row)
            if sig is not None:
                out.setdefault(sig, row)
        return out

    a_rows = index(run_a.rows)
    b_rows = index(run_b.rows)

    _ORDER = {"only_a": 0, "only_b": 1, "shared": 2}
    entries = []
    for sig in set(a_rows) | set(b_rows):
        in_a = sig in a_rows
        in_b = sig in b_rows
        category = "shared" if (in_a and in_b) else ("only_a" if in_a else "only_b")
        labels = sorted(sig[1])
        entries.append({
            "sig": sig,
            "itype": sig[0],
            "label": "%s — %s" % (" ⇄ ".join(labels), sig[0]),
            "category": category,
            "row_a": a_rows.get(sig),
            "row_b": b_rows.get(sig),
        })
    # Cluster changed rows (only-A, then only-B) above shared ones.
    entries.sort(key=lambda e: (_ORDER[e["category"]], e["label"]))

    return DifferenceResult(
        entries, name_a, name_b,
        n_common=len(matched_a),
        n_only_a=struct_a.num_residues - len(matched_a),
        n_only_b=struct_b.num_residues - len(matched_b),
    )

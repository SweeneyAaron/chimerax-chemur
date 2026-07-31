"""The ``chemur interactions`` ChimeraX command."""

from __future__ import annotations

from chimerax.core.commands import CmdDesc, register, BoolArg, FloatArg, IntArg, StringArg
from chimerax.atomic import AtomicStructureArg


def chemur_interactions(session, structure, protonate=False, pH=7.4,
                         profile="default", ligandSmiles=None, addHydrogens=True,
                         selectedOnly=False, skipProteinNucleic=False):
    from .runner import run_interactions
    return run_interactions(
        session,
        structure,
        protonate=protonate,
        protonate_ph=pH,
        profile=profile,
        ligand_smiles=parse_ligand_smiles(ligandSmiles),
        add_hydrogens=addHydrogens,
        selected_only=selectedOnly,
        skip_biopolymer_internal=skipProteinNucleic,
    ).rows


def parse_ligand_smiles(spec):
    """Parse "NAME:SMILES, NAME2:SMILES2" (or NAME=SMILES) into a dict."""
    if not spec:
        return None
    out = {}
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, smiles = part.split(":", 1)
        elif "=" in part:
            name, smiles = part.split("=", 1)
        else:
            continue
        name = name.strip()
        smiles = smiles.strip()
        if name and smiles:
            out[name.upper()] = smiles
    return out or None


def chemur_trajectory(session, structure, excludeSolvent=True, excludeIons=True,
                       start=0, stop=None, stride=1, processes=1,
                       profile="default", ligandSmiles=None, protonate=False, pH=7.4):
    """Analyse interactions per frame across a loaded trajectory; returns the run."""
    from .runner import run_trajectory

    def _progress(done, total):
        session.logger.status("Chemur trajectory: frame %d/%d" % (done, total))

    return run_trajectory(
        session,
        structure,
        exclude_solvent=excludeSolvent,
        exclude_ions=excludeIons,
        frame_start=start,
        frame_stop=stop,
        frame_stride=stride,
        processes=processes,
        profile=profile,
        ligand_smiles=parse_ligand_smiles(ligandSmiles),
        protonate=protonate,
        protonate_ph=pH,
        progress=_progress,
    )


_desc = CmdDesc(
    required=[("structure", AtomicStructureArg)],
    keyword=[
        ("protonate", BoolArg),
        ("pH", FloatArg),
        ("profile", StringArg),
        ("ligandSmiles", StringArg),
        ("addHydrogens", BoolArg),
        ("selectedOnly", BoolArg),
        ("skipProteinNucleic", BoolArg),
    ],
    synopsis="Predict Chemur interactions and draw them as pseudobonds",
)


_traj_desc = CmdDesc(
    required=[("structure", AtomicStructureArg)],
    keyword=[
        ("excludeSolvent", BoolArg),
        ("excludeIons", BoolArg),
        ("start", IntArg),
        ("stop", IntArg),
        ("stride", IntArg),
        ("processes", IntArg),
        ("profile", StringArg),
        ("ligandSmiles", StringArg),
        ("protonate", BoolArg),
        ("pH", FloatArg),
    ],
    synopsis="Analyse Chemur interactions per frame across an MD trajectory",
)


def register_command(name, logger):
    # ChimeraX calls this once per command declared in pyproject.toml; register the
    # matching handler by name.
    if name == "chemur trajectory":
        register("chemur trajectory", _traj_desc, chemur_trajectory, logger=logger)
    else:
        register("chemur interactions", _desc, chemur_interactions, logger=logger)

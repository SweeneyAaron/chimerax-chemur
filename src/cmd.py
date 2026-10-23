"""The ``chemur interactions`` and ``chemur trajectory`` ChimeraX commands."""

from __future__ import annotations

from chimerax.core.commands import CmdDesc, register, BoolArg, FloatArg, IntArg, StringArg
from chimerax.atomic import AtomicStructureArg


def chemur_interactions(session, structure, protonate=False, pH=7.4,
                         profile="default", ligandSmiles=None, addHydrogens=True,
                         selectedOnly=False, skipProteinNucleic=False,
                         hydrophobicEnclosure=None, assemblyOverrides=None):
    from .runner import run_interactions
    return run_interactions(
        session,
        structure,
        protonate=protonate,
        protonate_ph=pH,
        profile=profile,
        ligand_smiles=_command_ligand_smiles(ligandSmiles),
        add_hydrogens=addHydrogens,
        selected_only=selectedOnly,
        skip_biopolymer_internal=skipProteinNucleic,
        hydrophobic_enclosure=hydrophobicEnclosure,
        assembly_overrides=assemblyOverrides,
    ).rows


def parse_ligand_smiles(spec):
    """Parse NAME:SMILES or CHAIN:NAME:RESID=SMILES without splitting bonds."""
    import re
    if not spec:
        return None
    out = {}
    explicit = re.compile(r"^([A-Za-z0-9_.+-]+(?::[A-Za-z0-9_.+-]+:-?\d+[A-Za-z]?)?)\s*=(.+)$")
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        match = explicit.match(part)
        if match:
            name, smiles = match.groups()
        elif ":" in part:
            if re.match(r"^[A-Za-z0-9_.+-]+:[A-Za-z0-9_.+-]+:-?\d+[A-Za-z]?(?:$|\s|=)", part):
                raise ValueError("Full component identifiers require CHAIN:NAME:RESID=SMILES")
            name, smiles = part.split(":", 1)
            if not re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
                raise ValueError("Use NAME:SMILES or CHAIN:NAME:RESID=SMILES")
        else:
            raise ValueError("Use NAME:SMILES or CHAIN:NAME:RESID=SMILES")
        name, smiles = name.strip(), smiles.strip()
        if not name or not smiles:
            raise ValueError("A ligand identifier and nonempty SMILES are required")
        out[name.upper()] = smiles
    if not out:
        raise ValueError("A ligand identifier and nonempty SMILES are required")
    return out


def _command_ligand_smiles(spec):
    try:
        return parse_ligand_smiles(spec)
    except ValueError as error:
        from chimerax.core.errors import UserError
        raise UserError(str(error)) from error


def chemur_trajectory(session, structure, excludeSolvent=True, excludeIons=True,
                       start=0, stop=None, stride=1, processes=1,
                       profile="default", ligandSmiles=None, protonate=False, pH=7.4,
                       hydrophobicEnclosure=None, assemblyOverrides=None):
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
        ligand_smiles=_command_ligand_smiles(ligandSmiles),
        protonate=protonate,
        protonate_ph=pH,
        progress=_progress,
        hydrophobic_enclosure=hydrophobicEnclosure,
        assembly_overrides=assemblyOverrides,
    )


_desc = CmdDesc(
    required=[("structure", AtomicStructureArg)],
    keyword=[
        ("protonate", BoolArg),
        ("pH", FloatArg),
        ("profile", StringArg),
        ("ligandSmiles", StringArg),
        ("hydrophobicEnclosure", BoolArg),
        ("assemblyOverrides", StringArg),
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
        ("hydrophobicEnclosure", BoolArg),
        ("assemblyOverrides", StringArg),
        ("protonate", BoolArg),
        ("pH", FloatArg),
    ],
    synopsis="Analyse Chemur interactions per frame across an MD trajectory",
)


_COMMANDS = {
    "chemur interactions": (_desc, chemur_interactions),
    "chemur trajectory": (_traj_desc, chemur_trajectory),
}


def register_command(name, logger):
    # ChimeraX calls this once per command declared in pyproject.toml; register the
    # matching handler by name. An unknown name means a command was declared in
    # pyproject.toml with no handler here, so fail loudly rather than quietly
    # registering the wrong one under that name.
    try:
        desc, func = _COMMANDS[name]
    except KeyError:
        raise ValueError("ChimeraX-Chemur has no handler for command %r" % (name,))
    register(name, desc, func, logger=logger)

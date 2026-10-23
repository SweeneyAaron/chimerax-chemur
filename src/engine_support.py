"""Small, lazy compatibility boundary for the required Chemur engine."""

from __future__ import annotations

from copy import deepcopy
import inspect
from types import SimpleNamespace


def require_engine():
    """Validate the engine before a caller changes atoms or graphics."""
    try:
        from chemur import analyze
        from chemur.trajectory import ArrayFrameSource, analyze_trajectory
        from chemur.interaction_projection import project_interactions_for_scoring
        from chemur.ligand_selection import ligand_selections
        from chemur.preparation import prepare_ligand_chemistry
        from chemur.chemistry import rdkit_atom_annotations
        from chemur.parser import parse_structure, parse_embedded_ligand_smiles
        from chemur.features import perceive_features
        from chemur.profile import load_profile, apply_rule_overrides
        from chemur.interactions import HYDROPHOBIC_ENDPOINT_HYDROGEN_EXEMPTION
        from chemur.geometry_catalogue import GEOMETRY_REVISION, geometry_parameters, get_geometry_catalogue
        from chemur.preparation import MODIFIED_POLYMER_CHEMISTRY

        for function in (analyze, analyze_trajectory):
            if "assembly_overrides" not in inspect.signature(function).parameters:
                raise ImportError("missing ligand assembly support")
        if "hydrophobic_enclosure" not in load_profile("default").get("rules", {}):
            raise ImportError("missing hydrophobic enclosure support")
        if not GEOMETRY_REVISION or not callable(geometry_parameters) or not callable(get_geometry_catalogue):
            raise ImportError("missing versioned geometry definitions")
        if MODIFIED_POLYMER_CHEMISTRY is not True:
            raise ImportError("missing modified-polymer chemistry support")
        if HYDROPHOBIC_ENDPOINT_HYDROGEN_EXEMPTION is not True:
            raise ImportError("missing hydrophobic endpoint-hydrogen occlusion exemption")
    except (ImportError, AttributeError, ValueError) as exc:
        try:
            from chimerax.core.errors import UserError
        except ImportError:
            UserError = RuntimeError
        raise UserError(
            "Chemur requires an updated engine with hydrophobic enclosures, "
            "ligand assemblies, modified-polymer chemistry, versioned geometry definitions "
            "and the hydrophobic endpoint-hydrogen occlusion "
            "exemption (%s). Install the matching updated Chemur build "
            "in ChimeraX, then restart ChimeraX. No analysis was started." % exc
        ) from exc
    return SimpleNamespace(
        analyze=analyze, analyze_trajectory=analyze_trajectory,
        ArrayFrameSource=ArrayFrameSource, parse_structure=parse_structure,
        parse_embedded_ligand_smiles=parse_embedded_ligand_smiles,
        prepare_ligand_chemistry=prepare_ligand_chemistry,
        perceive_features=perceive_features, ligand_selections=ligand_selections,
        rdkit_atom_annotations=rdkit_atom_annotations,
        project_interactions_for_scoring=project_interactions_for_scoring,
        load_profile=load_profile, apply_rule_overrides=apply_rule_overrides,
        geometry_revision=GEOMETRY_REVISION, geometry_parameters=geometry_parameters,
    )


def check_cancel(callback=None):
    """Callbacks may raise RunCancelled or return True to request cancellation."""
    if callback is not None and callback():
        from .lifecycle import RunCancelled
        raise RunCancelled("Chemur analysis was cancelled.")


def analysis_options(engine, *, profile="default", rule_overrides=None,
                     hydrophobic_enclosure=None, assembly_overrides=None):
    """Copy and validate options; None preserves the profile's enclosure setting."""
    overrides = deepcopy(rule_overrides or {})
    if hydrophobic_enclosure is not None:
        overrides.setdefault("hydrophobic_enclosure", {})["enabled"] = bool(hydrophobic_enclosure)
    engine.apply_rule_overrides(engine.load_profile(profile), overrides)
    return dict(profile=profile, rule_overrides=overrides or None,
                assembly_overrides=assembly_overrides, include_raw=False)


def projected_interactions(interactions):
    """Unique actual display legs, with native group provenance retained."""
    from chemur.interaction_projection import project_interactions_for_scoring
    interactions = list(interactions)
    # Already projected records can arrive from a comparison or a redraw.
    if not any(i.interaction_type == "hydrophobic_enclosure"
               and "center_atom_id" in i.metadata for i in interactions):
        return interactions
    return project_interactions_for_scoring(interactions)


def capture_analysis_options(engine, options, **extra):
    """Freeze both requested settings and the effective rule profile on a run."""
    snapshot = deepcopy({**options, **extra})
    snapshot["geometry_catalogue_revision"] = engine.geometry_revision
    snapshot["effective_profile"] = engine.apply_rule_overrides(
        engine.load_profile(options.get("profile", "default")), options.get("rule_overrides"))
    return snapshot

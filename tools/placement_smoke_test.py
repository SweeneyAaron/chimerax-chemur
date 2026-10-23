"""Check separate model frames, both pose input routes and cross-model markers.

ChimeraX --nogui --notools --exit --cmd "runscript tools/placement_smoke_test.py /path/4W53.cif --engine-source /path/chemur/src"
Uses the local 4W53 toluene geometry; all outputs are temporary.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("structure", type=Path)
parser.add_argument("--engine-source", type=Path)
parser.add_argument("--sdf-only", action="store_true", help="isolate SDF and drawing checks")
args = parser.parse_args(sys.argv[1:])
if args.engine_source:
    sys.path.insert(0, str(args.engine_source.resolve()))
root = Path(__file__).resolve().parents[1]
package = types.ModuleType("_chemur_placement_smoke")
package.__path__ = [str(root / "src")]
sys.modules[package.__name__] = package
runner = importlib.import_module(package.__name__ + ".runner")
sdf_io = importlib.import_module(package.__name__ + ".sdf_io")
support = importlib.import_module(package.__name__ + ".engine_support")
engine = support.require_engine()

from chimerax.atomic import AtomicStructure, Atoms, Bond
from chimerax.core.commands import run
from chimerax.geometry import rotation, translation
from chimerax.markers import MarkerSet
from rdkit import Chem


def compensated_frame(model, transform):
    world = np.array(model.atoms.scene_coords, dtype=np.float64)
    model.position = transform
    model.atoms.coords = model.scene_position.inverse().transform_points(world)
    assert np.allclose(model.atoms.scene_coords, world, atol=2e-5, rtol=0)


def check_pose(analyzed, pose, receptor, route):
    assert analyzed.mapping_status == "complete", (route, analyzed.mapping_status)
    assert analyzed.pose_component_ids and len(analyzed.pose_component_ids) == 1
    targets = set(analyzed.pose_component_ids)
    atoms = {atom.atom_id: atom for atom in analyzed.result.atoms}
    ligand = [atom for atom in atoms.values() if atom.component_id in targets and atom.element != "H"]
    assert len(ligand) == 7, (route, "target heavy atoms", len(ligand))
    assert all(analyzed.id_to_atom[atom.atom_id].structure is pose for atom in ligand)
    for atom in atoms.values():
        live = analyzed.id_to_atom.get(atom.atom_id)
        if live is not None:
            assert np.allclose(atom.coord, live.scene_coord, atol=0.003, rtol=0), (route, atom.label)
    methyl = next(atom for atom in pose.atoms if atom.name == "C")
    groups = [inter for inter in analyzed.result.interactions
              if inter.interaction_type == "hydrophobic_enclosure"
              and analyzed.id_to_atom.get(inter.metadata["center_atom_id"]) is methyl]
    assert len(groups) == 1, (route, "methyl enclosure", len(groups))
    group = groups[0]
    neighbors = [analyzed.id_to_atom[aid] for aid in group.metadata["neighbor_atom_ids"]]
    assert {(atom.residue.number, atom.name) for atom in neighbors} == {
        (111, "CG1"), (118, "CD2"), (121, "CD1"), (153, "CZ")}, route
    assert all(atom.structure is receptor for atom in neighbors)
    rows = [row for row in analyzed.rows if any(
        member["center_atom_id"] == group.metadata["center_atom_id"]
        for member in row.extra.get("enclosure_groups", ()))]
    assert len(rows) == 4, (route, "mapped enclosure legs", len(rows))
    print(route, "PASS: complete pose mapping and four native enclosure legs")
    return rows


before = set(session.models.list())
run(session, "open " + json.dumps(str(args.structure.resolve())))
receptor = next(model for model in session.models.list()
                if model not in before and isinstance(model, AtomicStructure))
pose = receptor.copy(name="placement_pose")
session.models.add([pose])
Atoms([atom for atom in pose.atoms
       if (atom.residue.name, atom.residue.number) != ("MBN", 200)]).delete()
amino_acids = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
Atoms([atom for atom in receptor.atoms if atom.residue.name not in amino_acids]).delete()
assert pose.num_atoms == 7 and receptor.num_atoms > 1000
compensated_frame(receptor, translation((23, -13, 8)) * rotation((1, 2, 1), 37))
compensated_frame(pose, translation((-19, 17, -11)) * rotation((2, -1, 3), -53))
assert not np.allclose(pose.atoms.coords, pose.atoms.scene_coords)

with tempfile.TemporaryDirectory(prefix="chemur-placement-") as temporary:
    temporary = Path(temporary)
    if not args.sdf_only:
        saved = Path(runner.save_models_mmcif(session, [receptor, pose]))
        try:
            records, _components = engine.parse_structure(saved)
            mapping = runner._map_atoms(types.SimpleNamespace(atoms=records), [receptor, pose])
            assert len(mapping) == len(records), "Scene mmCIF did not map every source atom"
            for record in records:
                assert np.allclose(record.coord, mapping[record.atom_id].scene_coord, atol=0.003, rtol=0)
        finally:
            saved.unlink()

        assert not sdf_io.has_bond_orders(pose), "First analysis must exercise merged mmCIF"
        merged = runner.run_pose_interactions(session, pose, receptor,
            add_hydrogens=False, ligand_smiles={"MBN": "Cc1ccccc1"}, draw=False)
        check_pose(merged, pose, receptor, "merged mmCIF")

    # The validated fixture is toluene: six aromatic bonds and one methyl bond.
    Bond.register_attr(session, "order", "Chemur placement smoke", attr_type=float)
    for bond in pose.bonds:
        bond.order = 1.0 if any(atom.name == "C" for atom in bond.atoms) else 4.0
    assert sdf_io.has_bond_orders(pose) and pose.num_bonds == 7
    source_heavy = np.array(pose.atoms.scene_coords, dtype=np.float64)
    sdf_path = temporary / "pose_with_hydrogens.sdf"
    sdf_io.write_ligand_sdf(pose, str(sdf_path), add_hydrogens=True)
    molecule = next(iter(Chem.SDMolSupplier(str(sdf_path), removeHs=False)))
    assert molecule is not None and molecule.GetNumHeavyAtoms() == 7
    assert sum(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()) == 8
    positions = molecule.GetConformer().GetPositions()
    assert np.allclose(positions[:7], source_heavy, atol=0.0001, rtol=0)
    assert np.allclose(pose.atoms.scene_coords, source_heavy, atol=1e-7, rtol=0)
    assert np.isfinite(positions).all()

    sdf = runner.run_pose_interactions(session, pose, receptor, add_hydrogens=False, draw=False)
    legs = check_pose(sdf, pose, receptor, "SDF")

    protein_ring = [atom for atom in receptor.atoms if atom.residue.number == 153
                    and atom.name in {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}]
    ligand_ring = [atom for atom in pose.atoms if atom.name != "C"]
    assert len(protein_ring) == len(ligand_ring) == 6
    first = runner.Endpoint(protein_ring[0], "protein", "A", "153", "PHE", protein_ring)
    second = runner.Endpoint(ligand_ring[0], "ligand", "A", "200", "MBN", ligand_ring)
    ring_row = runner.InteractionRow("pipi_stack", first, second, 4.0, None, None, {}, None)
    graphics = []
    group = runner.draw_pose_rows(session, [ring_row, legs[0]], "Placement smoke", graphics=graphics)
    assert group.num_pseudobonds == 2, "Cross-model ring/contact pseudobonds were not created"
    markers = [model for model in graphics if isinstance(model, MarkerSet)]
    assert len(markers) == 2 and {model.parent for model in markers} == {receptor, pose}
    expected = {receptor: np.mean([atom.scene_coord for atom in protein_ring], axis=0),
                pose: np.mean([atom.scene_coord for atom in ligand_ring], axis=0)}
    for marker_set in markers:
        assert marker_set.num_atoms == 1
        assert np.allclose(marker_set.atoms[0].scene_coord, expected[marker_set.parent], atol=2e-5, rtol=0)
    # Child markers must continue to follow independently moved source models.
    pose.position = translation((2, 3, -1)) * pose.position
    ligand_marker = next(model.atoms[0] for model in markers if model.parent is pose)
    assert np.allclose(ligand_marker.scene_coord,
                       np.mean([atom.scene_coord for atom in ligand_ring], axis=0), atol=2e-5, rtol=0)
    assert all(marker.element.name == "He" for model in markers for marker in model.atoms)
    handles = list(graphics)
    session.models.close([pose])
    assert all(handle.deleted for handle in handles), "Pose source close leaked cross-model graphics"
    assert not receptor.deleted, "Pose source close deleted the unrelated receptor"
    session.models.close([receptor])
    assert set(session.models.list()) == before, "Placement analysis leaked a model"

routes = "SDF" if args.sdf_only else "scene mmCIF/SDF"
print("CHEMUR PLACEMENT SMOKE PASS: %s, H coordinates, independent model frames, ring markers, cleanup" % routes)

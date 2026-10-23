"""Native chemistry, whole assemblies and covalent boundaries in 2D diagrams."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from rdkit import Chem

from chemur.ligand_selection import ligand_selections
from chemur.models import AnalysisResult, AtomRecord, ComponentRecord


def _load_diagram():
    package = "chemur_diagram_under_test"
    root = Path(__file__).resolve().parents[1] / "src"
    if package not in sys.modules:
        module = ModuleType(package)
        module.__path__ = [str(root)]
        sys.modules[package] = module
    name = package + ".diagram"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "diagram.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


diagram = _load_diagram()


class Residue:
    def __init__(self, name, number):
        self.name, self.number = name, number
        self.insertion_code = ""
        self.atoms = []


class Atom:
    def __init__(self, record, residue):
        self.name, self.residue = record.name, residue
        self.element = SimpleNamespace(name=record.element, number={"C": 6, "N": 7, "O": 8}[record.element])
        self.scene_coord = record.coord
        self.bonds, self.neighbors = [], []
        residue.atoms.append(self)


class Bond:
    def __init__(self, first, second):
        self.atoms = first, second
        for atom, other in ((first, second), (second, first)):
            atom.bonds.append(self)
            atom.neighbors.append(other)

    def other_atom(self, atom):
        return self.atoms[1] if atom is self.atoms[0] else self.atoms[0]


def _assembly(attached=False):
    if attached:
        specs = [(10, "C1", "C", "A:L1:1", "ligand"), (20, "C2", "C", "A:L1:1", "ligand"),
                 (30, "O1", "O", "B:L2:2", "ligand"), (90, "NZ", "N", "P:LYS:9", "protein"),
                 (91, "CE", "C", "P:LYS:9", "protein")]
        smiles = "CC(=O)NC"
    else:
        specs = [(10, "O1", "O", "A:L1:1", "ligand"), (20, "C1", "C", "A:L1:1", "ligand"),
                 (30, "N1", "N", "B:L2:2", "ligand"), (90, "O", "O", "P:ALA:9", "protein")]
        smiles = "OC[NH3+]"
    atoms, components, live = [], [], {}
    residues = {}
    for index, (aid, name, element, cid, kind) in enumerate(specs):
        chain, resname, number = cid.split(":")
        atom = AtomRecord(aid, name, element, (float(index), 0., 0.), resname, number, chain, cid, kind)
        atoms.append(atom)
        residue = residues.setdefault(cid, Residue(resname, int(number)))
        live[aid] = Atom(atom, residue)
    for cid, residue in residues.items():
        chain, name, number = cid.split(":")
        ids = tuple(atom.atom_id for atom in atoms if atom.component_id == cid)
        components.append(ComponentRecord(cid, name, chain, number, atoms[next(i for i, a in enumerate(atoms) if a.component_id == cid)].molecule_type, ids))
    molecule = Chem.MolFromSmiles(smiles)
    mapping = {str(index): aid for index, aid in enumerate([10, 20, 30, 90, 91] if attached else [10, 20, 30])}
    orders = [{"atom_id_1": mapping[str(b.GetBeginAtomIdx())], "atom_id_2": mapping[str(b.GetEndAtomIdx())],
               "order": b.GetBondTypeAsDouble()} for b in molecule.GetBonds()]
    chemistry = {"assembly_id": "assembly:A:L1:1", "status": "validated", "smiles": smiles,
                 "template_mapping": mapping, "component_ids": ["A:L1:1", "B:L2:2"],
                 "atom_ids": [10, 20, 30], "bond_orders": orders}
    components[0].metadata.update(ligand_assembly_id=chemistry["assembly_id"], ligand_assembly=chemistry)
    components[1].metadata["ligand_assembly_id"] = chemistry["assembly_id"]
    result = AnalysisResult(atoms, components, [], [], metadata={"ligand_assemblies": {chemistry["assembly_id"]: chemistry}})
    return result, live, residues, chemistry


def test_whole_ligand_assembly_uses_template_charges_and_all_member_atoms():
    result, live, residues, chemistry = _assembly()
    unit = diagram._build_ligand_unit(ligand_selections(result)[0], result, live, np, Chem)
    assert unit.ok
    assert set(unit.residues) == {residues["A:L1:1"], residues["B:L2:2"]}
    assert set(unit.atom_to_local) == {live[10], live[20], live[30]}
    assert unit.mol.GetAtomWithIdx(unit.atom_to_local[live[30]]).GetFormalCharge() == 1
    assert unit.mol.GetNumBonds() == 2


def test_attached_product_keeps_carbonyl_and_marks_receptor_boundary_as_dummy():
    result, live, _residues, chemistry = _assembly(attached=True)
    unit = diagram._build_ligand_unit(ligand_selections(result)[0], result, live, np, Chem)
    assert unit.ok
    assert unit.mol.GetNumAtoms() == 4
    dummy = next(atom for atom in unit.mol.GetAtoms() if atom.GetAtomicNum() == 0)
    assert dummy.GetProp("atomLabel") == "LYS 9:NZ"
    assert "covalent attachment" in unit.label
    assert unit.mol.GetBondBetweenAtoms(unit.atom_to_local[live[20]], unit.atom_to_local[live[30]]).GetBondTypeAsDouble() == 2
    assert live[90] not in unit.atom_to_local


def test_missing_template_mapping_has_explicit_safe_fallback():
    result, live, _residues, chemistry = _assembly()
    del chemistry["template_mapping"]["2"]
    unit = diagram._build_ligand_unit(ligand_selections(result)[0], result, live, np, Chem)
    assert not unit.ok
    assert unit.mol is None
    assert "unresolved chemistry" in unit.label
    assert len(unit.atom_to_local) == 3


def test_partial_live_mapping_still_draws_full_chemical_assembly():
    result, live, _residues, chemistry = _assembly()
    live.pop(30)
    unit = diagram._build_ligand_unit(ligand_selections(result)[0], result, live, np, Chem)
    assert unit.ok and unit.mol.GetNumAtoms() == 3
    assert "L2 2" in unit.label


def test_selecting_one_member_includes_whole_assembly_contacts(monkeypatch):
    result, live, residues, chemistry = _assembly()
    def row(aid):
        return SimpleNamespace(endpoint1=SimpleNamespace(atom=live[aid], molecule_type="ligand"),
                               endpoint2=SimpleNamespace(atom=live[90], molecule_type="protein"),
                               interaction_type="hbond", distance=3., pseudobond=None)
    monkeypatch.setattr(diagram, "_mol2_bond_orders", lambda *args: {})
    def render(unit, *args, **kwargs):
        unit.atom_px = {atom: unit.local_xy[index] for atom, index in unit.atom_to_local.items()}
        unit.centroid_px = np.mean(list(unit.atom_px.values()), axis=0)
    monkeypatch.setattr(diagram, "_render_unit", render)
    scene = diagram.build_scene(None, None, [row(10), row(30), row(30)], {residues["A:L1:1"]},
                                native_result=result, id_to_atom=live)
    assert len(scene.units) == 2
    assert len(scene.edges) == 2
    assert scene.atom_unit[live[10]] is scene.atom_unit[live[30]]


def test_receptor_unit_uses_native_updated_charge_and_bond_order():
    result, live, residues, chemistry = _assembly(attached=True)
    first, second = next(a for a in result.atoms if a.atom_id == 90), next(a for a in result.atoms if a.atom_id == 91)
    first.formal_charge = 0
    first.hydrogen_count = 1
    first.metadata["chemistry_source"] = "product_override"
    Bond(live[90], live[91])
    unit = diagram._build_unit(residues["P:LYS:9"], {}, np, Chem,
                               native_by_atom={live[90]: first, live[91]: second},
                               native_orders={frozenset({90, 91}): 2.0})
    assert unit.ok
    nitrogen = unit.mol.GetAtomWithIdx(unit.atom_to_local[live[90]])
    assert nitrogen.GetFormalCharge() == 0  # Do not restore legacy LYS +1.
    assert nitrogen.GetNumExplicitHs() == 1
    assert unit.mol.GetBondWithIdx(0).GetBondTypeAsDouble() == 2


def test_diagram_fonts_apply_to_chemical_labels_at_every_size():
    from rdkit.Chem.Draw import rdMolDraw2D
    result, live, _residues, _chemistry = _assembly(attached=True)
    unit = diagram._build_ligand_unit(ligand_selections(result)[0], result, live, np, Chem)
    for size in (6, 10, 24):
        options = rdMolDraw2D.MolDrawOptions()
        diagram._apply_draw_style(options, 'bold', size)
        assert options.fixedFontSize == round(size * 96 / 72)
        diagram.restyle_unit(unit, 'bold', size)
        assert unit.png and set(unit.atom_px) == set(unit.atom_to_local)
        assert all(0 <= xy[0] <= unit.img_w and 0 <= xy[1] <= unit.img_h
                   for xy in unit.atom_px.values())
    assert diagram._STRUCTURE_STYLES == ('default', 'bw', 'bold')


def test_modified_polymer_product_uses_validated_depiction_and_report():
    from test_runner_backend import runner
    result, live, residues, chemistry = _assembly()
    chemistry['product_kind'] = 'modified_polymer'
    for component in result.components[:2]:
        component.molecule_type = 'nucleotide'
        component.metadata.clear()
    result.components[0].metadata['chemical_product'] = chemistry
    result.metadata = {'chemical_products': {chemistry['assembly_id']: chemistry}}
    selections, by_residue = diagram._ligand_context(result, live)
    assert len(selections) == 1
    assert by_residue[residues['A:L1:1']] is by_residue[residues['B:L2:2']]
    unit = diagram._build_ligand_unit(selections[0], result, live, np, Chem)
    assert unit.ok and len(unit.residues) == 2
    report = runner._build_ligand_report(result)
    assert len(report) == 1 and not report[0].editable
    assert report[0].smiles_used == chemistry['smiles']


def test_serialized_boundary_dummy_uses_omitted_atom_identity_and_never_contact_mapping():
    result, live, _residues, chemistry = _assembly(attached=True)
    chemistry['smiles'] = 'CC(=O)*'
    chemistry['template_mapping'] = {'0': 10, '1': 20, '2': 30}
    chemistry['boundary_template_mapping'] = {'3': 90}
    unit = diagram._build_ligand_unit(ligand_selections(result)[0], result, live, np, Chem)
    assert unit.ok
    dummy = next(atom for atom in unit.mol.GetAtoms() if atom.GetAtomicNum() == 0)
    assert dummy.GetProp('atomLabel') == 'LYS 9:NZ'
    assert set(unit.atom_to_local) == {live[10], live[20], live[30]}
    assert dummy.GetIdx() not in unit.atom_to_local.values()
    assert unit.mol.GetBondBetweenAtoms(unit.atom_to_local[live[20]], unit.atom_to_local[live[30]]).GetBondTypeAsDouble() == 2


def _presentation_scene():
    atoms = [object(), object(), object()]
    units = []
    for index, x in enumerate((100., 200., 9000.)):
        unit = diagram._Unit(SimpleNamespace(name='UNIT', number=index + 1, insertion_code=''))
        unit.ok = True
        unit.image = np.ones((20, 20, 4))
        unit.img_w = unit.img_h = 20
        unit.global_origin = np.array([x, 200.])
        unit.atom_to_local = {atoms[index]: 0}
        units.append(unit)
    positions = {atom: unit.global_origin + [10., 10.] for atom, unit in zip(atoms, units)}
    edge = diagram._Edge(units[0], atoms[0], None, units[1], atoms[1], None, 'hydrophobic', 3.5)
    hidden_edge = diagram._Edge(units[1], atoms[1], None, units[2], atoms[2], None, 'hbond', 3.0)
    scene = diagram.DiagramScene(units, [edge, hidden_edge], positions,
                                dict(zip(atoms, units)), (9999., 9999.), {}, True)
    scene.presentation.units[id(units[2])] = False
    return scene, atoms


def test_visible_bounds_follow_current_positions_and_exclude_hidden_units_and_legs():
    scene, atoms = _presentation_scene()
    units, positions, routes, (minimum, maximum) = diagram._visible_scene_geometry(scene)
    assert units == scene.units[:2]
    assert atoms[2] not in positions and [edge for edge, _ in routes] == scene.edges[:1]
    np.testing.assert_allclose(minimum, (80, 180))
    np.testing.assert_allclose(maximum, (240, 240))
    scene.units[0].global_origin += [-300, 50]
    scene.atom_pos[atoms[0]] += [-300, 50]
    _, positions, _, (minimum, maximum) = diagram._visible_scene_geometry(scene)
    assert minimum[0] == -220 and maximum[1] == 290
    np.testing.assert_allclose(positions[atoms[0]], (-190, 260))
    for unit in scene.units:
        scene.presentation.units[id(unit)] = False
    units, positions, routes, bounds = diagram._visible_scene_geometry(scene)
    assert units == [] and positions == {} and routes == []
    assert np.all(np.isfinite(bounds))


def test_headless_canvas_contains_long_boundary_labels_and_shared_presentation(tmp_path):
    import pytest
    pytest.importorskip('matplotlib')
    scene, _ = _presentation_scene()
    scene.units[0].label = 'C36 427\ncovalent attachment: ' + ', '.join(['DC 426:O3prime', 'DG 428:P', 'CYS 81:SG'] * 4)
    scene.presentation.font_size = 18
    scene.presentation.colors['hydrophobic'] = (1., 0., 0., 1.)
    scene.presentation.legend_visible = False
    figure = diagram.render_scene_matplotlib(scene)
    axis = figure.axes[0]
    assert axis.get_xlim() == (80, 240)
    assert axis.get_ylim() == (240, 180)
    assert len(axis.lines) == 1 and axis.lines[0].get_color() == (1., 0., 0., 1.)
    assert axis.get_legend() is None
    assert all(text.get_fontsize() == 18 for text in axis.texts)
    renderer = figure.canvas.get_renderer()
    for text in axis.texts:
        extent = text.get_window_extent(renderer)
        assert figure.bbox.contains(extent.x0, extent.y0)
        assert figure.bbox.contains(extent.x1, extent.y1)
    # A plain save must include labels; callers need not know about tight bboxes.
    figure.savefig(tmp_path / 'current.png')
    from PIL import Image
    with Image.open(tmp_path / 'current.png') as saved:
        assert saved.width >= int(max(t.get_window_extent(renderer).width for t in axis.texts))

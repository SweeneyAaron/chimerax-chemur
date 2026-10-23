"""Native enclosure, scene mapping, chemistry and cancellation backend contracts."""
from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from test_runner_mapping import runner, _CxAtom, _Residue, _Structure, _Rec
from chemur.models import AtomRecord, AssignedInteraction, ComponentRecord, FeatureRecord

support = importlib.import_module(runner.__package__ + '.engine_support')
lifecycle = importlib.import_module(runner.__package__ + '.lifecycle')


def fixture_result(name):
    data = json.loads((Path(__file__).parent / 'fixtures' / (name + '_enclosure.json')).read_text())
    return NS(atoms=[AtomRecord(**a) for a in data['atoms']],
              components=[ComponentRecord(**c) for c in data['components']],
              features=[FeatureRecord(**f) for f in data['features']],
              interactions=[AssignedInteraction(**i) for i in data['interactions']],
              metadata=data.get('metadata', {}))


def live_atoms(result):
    mapping = {}
    for rec in result.atoms:
        residue = _Residue(rec.chain_id, int(rec.residue_id), rec.residue_name)
        atom = _CxAtom(rec.name, tuple(rec.coord), residue)
        atom.scene_coord = tuple(rec.coord)
        atom.selected = rec.molecule_type == 'ligand'
        atom.element = NS(name=rec.element)
        mapping[rec.atom_id] = atom
    return mapping


@pytest.mark.parametrize('name', ['4W53', '1MHT'])
def test_real_native_enclosures_map_to_all_actual_ligand_legs(name):
    result = fixture_result(name)
    original = deepcopy(result.interactions)
    mapping = live_atoms(result)
    rows = runner._resolve_rows(result.interactions, {a.atom_id: a for a in result.atoms},
                                mapping, {f.feature_id: f for f in result.features},
                                skip_biopolymer_internal=True, selected_only=True)
    group = result.interactions[0]
    assert {tuple(sorted(r.atom_ids)) for r in rows} == {
        tuple(sorted(leg['atom_ids'])) for leg in group.metadata['legs']}
    assert all(group.metadata['center_atom_id'] in row.atom_ids for row in rows)
    assert {r.distance for r in rows} == {leg['distance'] for leg in group.metadata['legs']}
    assert all(row.extra['enclosure_groups'][0]['center_atom_id'] == group.metadata['center_atom_id']
               for row in rows)
    assert result.interactions == original
    run = runner.InteractionRun(rows, [], result, (), mapping)
    assert run.result.interactions == original


def test_overlap_leg_dedup_keeps_both_native_group_provenances():
    result = fixture_result('4W53')
    group = result.interactions[0]
    leg = group.metadata['legs'][0]
    second_center = next(a for a in leg['atom_ids'] if a != group.metadata['center_atom_id'])
    second = replace(group, assignment_order=900, metadata=dict(
        center_atom_id=second_center, neighbor_atom_ids=[group.metadata['center_atom_id']],
        interaction_subtype='hydrophobic_enclosure', legs=[deepcopy(leg)]))
    rows = runner._resolve_rows([group, second], {a.atom_id: a for a in result.atoms},
                               live_atoms(result), {})
    shared = [r for r in rows if set(r.atom_ids) == set(leg['atom_ids'])]
    assert len(shared) == 1
    assert len(shared[0].extra['enclosure_groups']) == 2
    assert len(rows) == len(group.metadata['legs'])


def test_mapping_uses_scene_frame_and_rejects_ambiguous_copies():
    residue = _Residue('A', 1, 'LIG')
    first = _CxAtom('C1', (0, 0, 0), residue)
    second = _CxAtom('C1', (0, 0, 0), residue)
    first.scene_coord = (10, 2, 3)
    second.scene_coord = (20, 2, 3)
    coords, names = runner._chimerax_atom_lookup(_Structure([first, second]))
    assert runner._resolve_atom(_Rec('C1', (20, 2, 3), 'A', '1', 'LIG'), coords, names) is second
    second.scene_coord = first.scene_coord
    coords, names = runner._chimerax_atom_lookup(_Structure([first, second]))
    assert runner._resolve_atom(_Rec('C1', (10, 2, 3), 'A', '1', 'LIG'), coords, names) is None


def test_mapping_rejects_one_live_atom_for_two_native_records():
    result = fixture_result('4W53')
    atom = result.atoms[0]
    result.atoms = [atom, replace(atom, atom_id=99999)]
    live = live_atoms(NS(atoms=[atom]))[atom.atom_id]
    assert runner._map_atoms(result, [_Structure([live])]) == {}


def test_enclosure_option_preserves_profile_and_copies_overrides():
    engine = support.require_engine()
    overrides = {'hydrophobic': {'distance': 3.8}}
    options = support.analysis_options(engine, rule_overrides=overrides, hydrophobic_enclosure=None)
    assert 'hydrophobic_enclosure' not in options['rule_overrides']
    off = support.analysis_options(engine, rule_overrides=overrides, hydrophobic_enclosure=False)
    assert off['rule_overrides']['hydrophobic_enclosure']['enabled'] is False
    assert overrides == {'hydrophobic': {'distance': 3.8}}
    assert off['include_raw'] is False


def install_errors(monkeypatch):
    errors = ModuleType('chimerax.core.errors')
    errors.UserError = RuntimeError
    monkeypatch.setitem(sys.modules, 'chimerax.core.errors', errors)


def test_old_engine_gate_fails_before_mutations(monkeypatch):
    import chemur.profile
    install_errors(monkeypatch)
    monkeypatch.setattr(chemur.profile, 'load_profile', lambda p: {'rules': {}})
    mutations = []
    monkeypatch.setattr(runner, '_addh_and_clear', lambda *a, **kw: mutations.append('addh'))
    with pytest.raises(RuntimeError, match='updated engine'):
        runner.run_interactions(NS(), NS())
    assert not mutations


@pytest.mark.parametrize('capability', [None, False])
def test_engine_gate_requires_endpoint_hydrogen_exemption_before_mutations(monkeypatch, capability):
    import chemur.interactions
    install_errors(monkeypatch)
    name = 'HYDROPHOBIC_ENDPOINT_HYDROGEN_EXEMPTION'
    if capability is None:
        monkeypatch.delattr(chemur.interactions, name, raising=False)
    else:
        monkeypatch.setattr(chemur.interactions, name, capability, raising=False)
    mutations = []
    monkeypatch.setattr(runner, '_addh_and_clear', lambda *a, **kw: mutations.append('addh'))
    monkeypatch.setattr(runner, '_draw_interactions', lambda *a, **kw: mutations.append('draw'))
    with pytest.raises(RuntimeError, match='hydrophobic endpoint-hydrogen occlusion exemption'):
        runner.run_interactions(NS(), NS())
    assert not mutations


def test_cancelled_engine_result_never_maps_or_draws(monkeypatch, tmp_path):
    install_errors(monkeypatch)
    engine = support.require_engine()
    path = tmp_path / 'source.cif'
    path.write_text('test')
    cancelled = []
    def analyze(*args, **kwargs):
        assert kwargs['include_raw'] is False
        cancelled.append(True)
        return fixture_result('4W53')
    engine.analyze = analyze
    monkeypatch.setattr(runner, 'require_engine', lambda: engine)
    monkeypatch.setattr(runner, '_addh_and_clear', lambda *a, **kw: None)
    monkeypatch.setattr(runner, 'save_models_mmcif', lambda *a: str(path))
    monkeypatch.setattr(runner, '_map_atoms', lambda *a: pytest.fail('mapped after cancellation'))
    monkeypatch.setattr(runner, '_draw_interactions', lambda *a, **k: pytest.fail('drew after cancellation'))
    with pytest.raises(lifecycle.RunCancelled):
        runner.run_interactions(NS(), NS(), cancel_check=lambda: bool(cancelled))
    assert not path.exists()


def test_explicit_model_writer_does_not_select_child_markers(monkeypatch):
    writer_module = ModuleType('chimerax.mmcif.mmcif_write')
    calls = []
    writer_module.write_mmcif = lambda session, path, **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, 'chimerax.mmcif.mmcif_write', writer_module)
    atom = _CxAtom('C', (1, 2, 3), _Residue('A', 1, 'LIG'))
    atom.element = NS(name='C')
    source, child = _Structure([atom]), object()
    snapshots = []
    chemistry = importlib.import_module(runner.__package__ + '.source_chemistry')
    preserved = []
    monkeypatch.setattr(chemistry, 'preserve_source_chemistry',
                        lambda path, sources, **kw: preserved.extend(sources))
    monkeypatch.setattr(runner, '_normalize_mmcif_snapshot', lambda path, atoms, n: snapshots.extend(atoms))
    path = runner.save_models_mmcif(NS(logger=NS()), [source])
    try:
        assert calls == [{'models': [source], 'rel_model': None, 'all_coordsets': False}]
        assert snapshots[0]['coord'] == (1, 2, 3)
        assert preserved == [source]
        assert child not in calls[0]['models']
    finally:
        Path(path).unlink()


def test_assembly_report_uses_product_smiles_and_stable_ids():
    members = [ComponentRecord('A:X:1', 'X', 'A', '1', 'ligand', (1,), {}),
               ComponentRecord('A:X:2', 'X', 'A', '2', 'ligand', (2,), {})]
    atoms = [AtomRecord(i, 'C', 'C', (0, 0, 0), 'X', str(i), 'A', 'A:X:'+str(i),
                        'ligand', formal_charge=-1 if i == 1 else 0) for i in (1, 2)]
    chemistry = dict(assembly_id='assembly:A:X:1', component_ids=[c.component_id for c in members],
                     atom_ids=[1, 2], status='validated', smiles='C[O-]', chemistry_source='assembly_override')
    result = NS(atoms=atoms, components=members, metadata={'ligand_assemblies': {chemistry['assembly_id']: chemistry}})
    report = runner._build_ligand_report(result)
    assert len(report) == 1
    assert report[0].smiles_used == 'C[O-]'
    assert report[0].component_id == 'assembly:A:X:1'
    assert report[0].component_ids == ('A:X:1', 'A:X:2')
    assert report[0].net_charge == -1
    assert report[0].editable is False


def test_frame_shell_is_current_and_has_center_shell_endpoint_groups():
    result = fixture_result('4W53')
    group = result.interactions[0]
    center = group.metadata['center_atom_id']
    first_neighbor, second_neighbor = group.metadata['neighbor_atom_ids'][:2]
    def frame_group(neighbor):
        return replace(group, atom_ids=(center, neighbor), metadata={**group.metadata,
            'neighbor_atom_ids': [neighbor]})
    key = ('hydrophobic_enclosure', frozenset([center]))
    trajectory = NS(frames=[NS(frame_index=0, interactions=[frame_group(first_neighbor)]),
                            NS(frame_index=1, interactions=[frame_group(second_neighbor)])],
                    interaction_key=lambda interaction: key)
    mapping = live_atoms(result)
    run = runner.TrajectoryRun(trajectory, {}, {}, NS(), [1, 2],
                               {a.atom_id: a for a in result.atoms}, {}, mapping)
    assert run.frame_participants(1, key) == [mapping[center], mapping[second_neighbor]]
    assert run.frame_endpoints(1, key) == [[mapping[center]], [mapping[second_neighbor]]]
    assert mapping[first_neighbor] not in run.frame_participants(1, key)
    assert run.frame_participants(None, key) == [mapping[center]]
    assert run.frame_endpoints(None, key) == [[mapping[center]]]


def test_acceptor_snapshot_uses_engine_features_and_scene_cutoff():
    calls = []
    engine = NS(parse_structure=lambda path, **kw: (calls.append(('parse', kw)) or ([], [])),
                prepare_ligand_chemistry=lambda *args, **kw: calls.append(('prepare', kw)),
                perceive_features=lambda *args: [NS(feature_type='acceptor', center=(10, 0, 0)),
                                               NS(feature_type='donor', center=(1, 0, 0))])
    targets = runner.prepare_receptor_acceptors('prepared.cif', engine=engine)
    assert targets == ((10, 0, 0),)
    assert calls[0][1]['include_ligands'] is False
    assert calls[1][1]['sdf_mode'] is True
    pose = NS(atoms=[NS(coord=(0, 0, 0), scene_coord=(11, 0, 0))])
    assert runner._nearby_acceptors(targets, pose) == [(10.0, 0.0, 0.0)]


def test_trajectory_uses_scene_stack_and_keeps_filtered_topology_identity(monkeypatch, tmp_path):
    install_errors(monkeypatch)
    engine = support.require_engine()
    structure = NS(active_coordset_id=1, num_coordsets=2, coordset_ids=[1, 2], id_string='7')
    class FrameAtom:
        def __init__(self, index, residue):
            self.index = index
            self.residue = residue
            self.name = 'C' if index != 1 else 'O'
            self.element = NS(name=self.name)
        @property
        def coord(self):
            return (self.index * 10 + structure.active_coordset_id - 1, 0, 0)
        @property
        def scene_coord(self):
            return (self.coord[0] + 100, 3, 4)
    # Same-name protein copies flank a removable water: name-only remapping
    # would collapse atom 2 onto atom 0 after the engine filters topology.
    repeated = _Residue('A', 1, 'ALA')
    structure.atoms = [FrameAtom(0, repeated), FrameAtom(1, _Residue('W', 1, 'HOH')),
                       FrameAtom(2, repeated)]
    topology = [AtomRecord(i, atom.name, atom.element.name, atom.scene_coord,
                          atom.residue.name, '1', atom.residue.chain_id,
                          atom.residue.chain_id + ':' + atom.residue.name + ':1',
                          'solvent' if i == 1 else 'protein')
                for i, atom in enumerate(structure.atoms)]
    topo_path = tmp_path / 'topology.cif'
    topo_path.write_text('topology')
    monkeypatch.setattr(runner, 'save_models_mmcif', lambda *a: str(topo_path))
    engine.parse_structure = lambda path: (deepcopy(topology), [])
    engine.perceive_features = lambda *args: []
    paths = []
    def source(path, coords):
        paths.append(coords)
        return np.load(coords).copy()
    engine.ArrayFrameSource = source
    def analyze(stack, **kwargs):
        assert kwargs['exclude_solvent'] is True
        assert stack[:, 0, :].tolist() == [[100, 3, 4], [101, 3, 4]]
        assert stack[:, 2, :].tolist() == [[120, 3, 4], [121, 3, 4]]
        assert 'include_raw' not in kwargs
        kwargs['progress'](2, 2)
        return NS(atoms=[replace(topology[i], coord=tuple(stack[-1, i])) for i in (0, 2)],
                  components=[], frames=[NS(frame_index=0, interactions=[]), NS(frame_index=1, interactions=[])],
                  frame_indices=[0, 1], n_frames=2, occupancy=lambda: {})
    engine.analyze_trajectory = analyze
    monkeypatch.setattr(runner, 'require_engine', lambda: engine)
    run = runner.run_trajectory(NS(logger=NS(info=lambda message: None)), structure, exclude_solvent=True)
    assert run.id_to_atom == {0: structure.atoms[0], 2: structure.atoms[2]}
    assert structure.active_coordset_id == 1
    assert run.frame_coordset_ids == [1, 2]
    assert run.analysis_options["exclude_solvent"] is True
    assert run.analysis_options["include_raw"] is False
    assert not topo_path.exists()
    assert all(not Path(path).exists() for path in paths)


def test_successful_run_retains_native_group_and_graphics_ownership(monkeypatch, tmp_path):
    install_errors(monkeypatch)
    engine = support.require_engine()
    result = fixture_result('4W53')
    original = deepcopy(result.interactions)
    mapping = live_atoms(result)
    structure = _Structure(list(mapping.values()))
    path = tmp_path / 'source.cif'
    path.write_text('test')
    engine.analyze = lambda *a, **kw: result
    monkeypatch.setattr(runner, 'require_engine', lambda: engine)
    monkeypatch.setattr(runner, '_addh_and_clear', lambda *a, **kw: None)
    monkeypatch.setattr(runner, 'save_models_mmcif', lambda *a: str(path))
    handle = object()
    def draw(session, source, interactions, records, mapped, features, **kwargs):
        kwargs['graphics'].append(handle)
        return runner._resolve_rows(interactions, records, mapped, features,
                                    selected_only=True, skip_biopolymer_internal=True)
    monkeypatch.setattr(runner, '_draw_interactions', draw)
    tracked = []
    monkeypatch.setattr(runner, '_track_graphics', lambda session, sources, handles: tracked.append((sources, handles)))
    run = runner.run_interactions(NS(logger=NS(warning=lambda message: None)), structure,
                                  selected_only=True, skip_biopolymer_internal=True)
    assert len(run.rows) == 4
    assert run.result is result and run.result.interactions == original
    assert run.source_models == (structure,)
    assert run.graphics == [handle]
    assert tracked[0][1] is run.graphics
    assert run.analysis_options["selected_only"] is True
    assert run.analysis_options["skip_biopolymer_internal"] is True
    assert "hydrophobic_enclosure" in run.analysis_options["effective_profile"]["rules"]
    assert not path.exists()


def test_cancelled_partial_draw_cleans_owned_handles(monkeypatch):
    result = fixture_result('4W53')
    class Group:
        def __init__(self):
            self.num_pseudobonds = 0
            self.id = None
            self.deleted = False
        def delete(self):
            self.deleted = True
    pbg, pi = Group(), Group()
    session = NS(pb_manager=NS(get_group=lambda *a, **kw: pi))
    structure = NS(pseudobond_group=lambda name: pbg, id_string='1')
    monkeypatch.setattr(runner, '_clear_pi_markers', lambda *a: None)
    cancelled = []
    monkeypatch.setattr(runner, '_draw_interaction', lambda *a: cancelled.append(True) or object())
    graphics = []
    with pytest.raises(lifecycle.RunCancelled):
        runner._draw_interactions(session, structure, result.interactions,
                                  {a.atom_id: a for a in result.atoms}, live_atoms(result), {},
                                  graphics=graphics, cancel_check=lambda: bool(cancelled))
    assert pbg.deleted and pi.deleted
    assert graphics == []


@pytest.mark.parametrize('changes', [
    {'protonate': True},
    {'ligand_smiles': {'MBN': 'Cc1ccccc1'}},
    {'assembly_overrides': {'products': []}},
])
def test_sdf_orientation_skips_chemistry_that_engine_will_change(monkeypatch, changes):
    engine = support.require_engine()
    result = fixture_result('4W53')
    def analyze(*args, **kwargs):
        for atom in result.atoms:
            if atom.molecule_type == 'ligand':
                atom.metadata.update(source='sdf', source_path=kwargs['ligand_sdf'][0])
        return result
    engine.analyze = analyze
    monkeypatch.setattr(runner, 'require_engine', lambda: engine)
    monkeypatch.setattr(runner, '_has_protein', lambda pose: False)
    mapping = live_atoms(result)
    monkeypatch.setattr(runner, '_map_atoms', lambda *a: mapping)
    monkeypatch.setattr(runner, 'prepare_receptor_acceptors',
                        lambda *a, **kw: pytest.fail('typed targets for changing chemical state'))
    sdf = importlib.import_module(runner.__package__ + '.sdf_io')
    monkeypatch.setattr(sdf, 'has_bond_orders', lambda pose: True)
    written = []
    monkeypatch.setattr(sdf, 'write_ligand_sdf', lambda pose, path, **kw: written.append(kw) or path)
    center = result.interactions[0].metadata['center_atom_id']
    pose = NS(name='pose', atoms=[mapping[center]])
    receptor = NS()
    run = runner.run_pose_interactions(NS(), pose, receptor,
                                      receptor_path='already_prepared.cif',
                                      receptor_acceptors=[mapping[center].scene_coord], **changes)
    assert written[0]['acceptor_coords'] is None
    assert run.mapping_status == 'complete'
    assert run.fully_mapped is True
    assert run.result is result


def test_pose_component_provenance_excludes_cofactor_without_using_live_mapping():
    result = fixture_result('4W53')
    ligand = next(a for a in result.atoms if a.molecule_type == 'ligand')
    cofactor = replace(ligand, atom_id=99999, component_id='B:HEM:9', residue_name='HEM',
                       chain_id='B', residue_id='9', coord=(100, 100, 100))
    result.atoms.append(cofactor)
    mapping = live_atoms(result)
    pose = _Structure([mapping[ligand.atom_id]])
    receptor = _Structure([atom for aid, atom in mapping.items() if aid != ligand.atom_id])
    snapshot = runner._source_atom_snapshot([receptor, pose], pose)
    # Origin was frozen before analysis; current atoms can now be missing/moved.
    pose.atoms.clear()
    assert runner._pose_component_ids(result, snapshot=snapshot) == (ligand.component_id,)
    assert set(runner._pose_component_ids(result, whole_complex=True)) == {ligand.component_id, cofactor.component_id}
    assert runner._pose_component_ids(result, snapshot=tuple((False, *item[1:]) for item in snapshot)) == ()


def test_pose_component_provenance_refuses_cross_source_component_collision():
    result = fixture_result('4W53')
    ligand = next(a for a in result.atoms if a.molecule_type == 'ligand')
    duplicate = replace(ligand, atom_id=99999, coord=(100, 100, 100))
    result.atoms.append(duplicate)
    mapping = live_atoms(result)
    pose = _Structure([mapping[ligand.atom_id]])
    receptor = _Structure([atom for aid, atom in mapping.items() if aid != ligand.atom_id])
    snapshot = runner._source_atom_snapshot([receptor, pose], pose)
    assert runner._pose_component_ids(result, snapshot=snapshot) is None


def test_sdf_pose_provenance_requires_its_exact_source_file_and_expands_assembly():
    result = fixture_result('4W53')
    ligand = next(a for a in result.atoms if a.molecule_type == 'ligand')
    ligand.metadata.update(source='sdf', source_path='/tmp/pose.sdf')
    result.metadata['ligand_assemblies'] = {'assembly': {'component_ids': [ligand.component_id, 'S:L2:2']}}
    assert runner._pose_component_ids(result, sdf_path='/tmp/pose.sdf') == tuple(sorted([ligand.component_id, 'S:L2:2']))
    assert runner._pose_component_ids(result, sdf_path='/tmp/other.sdf') == ()


def test_run_payload_options_are_copied_not_live_editor_state():
    settings = {'ligand_smiles': {'SAH': 'original'}, 'rule_overrides': {'hydrophobic': {'distance': 3.8}}}
    result = fixture_result('1MHT')
    runs = [runner.InteractionRun([], [], analysis_options=settings),
            runner.PoseInteractionRun([], result, analysis_options=settings),
            runner.TrajectoryRun(NS(frames=[]), {}, {}, NS(), [], {}, {}, {}, analysis_options=settings)]
    settings['ligand_smiles']['SAH'] = 'edited'
    settings['rule_overrides']['hydrophobic']['distance'] = 2.0
    for run in runs:
        assert run.analysis_options['ligand_smiles']['SAH'] == 'original'
        assert run.analysis_options['rule_overrides']['hydrophobic']['distance'] == 3.8


def test_summary_distinguishes_enclosure_groups_from_display_contacts():
    messages = []
    session = NS(logger=NS(info=messages.append))
    runner._log_summary(session, NS(id_string='1'), {'hydrophobic_enclosure': 4}, 4, 0, 0,
                        enclosure_groups=1)
    assert '4 contacts' in messages[0]
    assert '1 hydrophobic enclosure group(s)' in messages[0]


def test_absent_enclosure_in_analyzed_frame_selects_only_center():
    result = fixture_result('4W53')
    group = result.interactions[0]
    center = group.metadata['center_atom_id']
    key = ('hydrophobic_enclosure', frozenset([center]))
    trajectory = NS(frames=[NS(frame_index=0, interactions=[group]),
                            NS(frame_index=1, interactions=[])],
                    interaction_key=lambda interaction: key)
    mapping = live_atoms(result)
    run = runner.TrajectoryRun(trajectory, {}, {}, NS(), [1, 2],
                               {a.atom_id: a for a in result.atoms}, {}, mapping)
    assert run.frame_participants(1, key) == [mapping[center]]
    assert run.frame_participants(1) == {}
    assert run.frame_endpoints(1, key) == []
    assert run.frame_endpoints(1) == {}
    assert run.frame_participants(None, key) == [mapping[center]]
    assert trajectory.frames[1].interactions == []


def test_mmcif_snapshot_merges_models_keeps_active_altloc_and_named_chemistry(tmp_path):
    import gemmi
    from chemur.parser import parse_structure
    document = gemmi.cif.Document()
    columns = ['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id',
               'label_comp_id', 'label_asym_id', 'label_entity_id', 'label_seq_id',
               'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv',
               'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'pdbx_PDB_model_num']
    def add_block(name, rows, residue_name):
        block = document.add_new_block(name)
        block.set_mmcif_category('_atom_site.', {key: [False if v == '.' else v for v in values] for key, values in zip(columns, zip(*rows))})
        block.set_mmcif_category('_struct_asym.', {'id': ['A'], 'entity_id': ['1']})
        block.set_mmcif_category('_entity.', {'id': ['1'], 'type': ['non-polymer']})
        block.set_mmcif_category('_chem_comp.', {'id': [residue_name], 'type': ['non-polymer']})
        return block
    receptor = add_block('receptor', [
        ['ATOM', '1', 'C', 'CA', 'A', 'MET', 'A', '1', '1', '1', '2', '3', '.5', '10', '1', 'MET', 'A', 'CA', '1'],
        ['ATOM', '2', 'C', 'CA', 'B', 'MET', 'A', '1', '1', '1', '2', '3', '.5', '10', '1', 'MET', 'A', 'CA', '1'],
        ['ATOM', '3', 'C', 'CA', 'B', 'MET', 'A', '1', '1', '9', '2', '3', '.5', '10', '1', 'MET', 'A', 'CA', '2'],
    ], 'MET')
    pose = add_block('pose', [
        ['HETATM', '1', 'C', 'C', '.', 'MBN', 'A', '1', '.', '7', '8', '9', '1', '10', '200', 'MBN', 'A', 'C', '1'],
    ], 'MBN')
    pose.set_mmcif_category('_chem_comp_atom.', {'comp_id': ['MBN'], 'atom_id': ['C'],
                                               'type_symbol': ['C'], 'charge': ['0']})
    path = tmp_path / 'separate_blocks.cif'
    document.write_file(str(path))
    snapshots = [dict(id=(0, 0), model=0, element='C', name=('A', 'MET', '1', 'CA'),
                      coord=(1., 2., 3.), altloc='B'),
                 dict(id=(1, 0), model=1, element='C', name=('A', 'MBN', '200', 'C'),
                      coord=(7., 8., 9.), altloc='')]
    original = deepcopy(snapshots)
    runner._normalize_mmcif_snapshot(path, snapshots, 2)
    normalized = gemmi.cif.read_file(str(path))
    assert len(normalized) == 1
    block = normalized.sole_block()
    atom_table = block.get_mmcif_category('_atom_site.')
    assert atom_table['label_alt_id'] == ['B', False]
    assert atom_table['pdbx_PDB_model_num'] == ['1', '1']
    assert atom_table['label_asym_id'] == ['s1_A', 's2_A']
    assert block.get_mmcif_category('_chem_comp_atom.')['atom_id'] == ['C']
    atoms, components = parse_structure(path)
    assert [(a.component_id, a.name, a.coord) for a in atoms] == [
        ('A:MET:1', 'CA', (1., 2., 3.)), ('A:MBN:200', 'C', (7., 8., 9.))]
    assert snapshots == original


def test_combined_models_reject_ambiguous_component_identifiers(monkeypatch):
    writer_module = ModuleType('chimerax.mmcif.mmcif_write')
    called = []
    writer_module.write_mmcif = lambda *a, **kw: called.append(True)
    monkeypatch.setitem(sys.modules, 'chimerax.mmcif.mmcif_write', writer_module)
    first = _Structure([_CxAtom('C', (0, 0, 0), _Residue('A', 1, 'LIG'))])
    second = _Structure([_CxAtom('C', (5, 0, 0), _Residue('A', 1, 'LIG'))])
    with pytest.raises(ValueError, match='distinct chain identifiers'):
        runner.save_models_mmcif(NS(), [first, second])
    assert called == []

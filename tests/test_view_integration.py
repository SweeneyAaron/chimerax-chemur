"""Exercise view membership and frame filters without starting a Qt event loop."""
import ast
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_runner_backend import runner, fixture_result, live_atoms

view = importlib.import_module(runner.__package__ + '.view_state')
export = importlib.import_module(runner.__package__ + '.export')
ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, *, class_name=None, globals=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    namespace = dict(globals or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('code', ['4W53', '1MHT'])
def test_native_group_displayed_legs_filter_and_export_roundtrip(code, tmp_path):
    native = fixture_result(code)
    mapping = live_atoms(native)
    rows = runner._resolve_rows(native.interactions, {a.atom_id: a for a in native.atoms}, mapping, {})
    state = view.ContactView(rows)
    center = native.interactions[0].metadata['center_atom_id']
    assert state.native_count == 1 and len(rows) == 4
    state.set_visible([0], False, center)
    displayed = [r for i, r in enumerate(rows) if state.row_visible(i)]
    export.write_json(displayed, tmp_path / 'displayed.json')
    import json
    payload = json.loads((tmp_path / 'displayed.json').read_text())
    assert len(payload) == 3
    assert all(row['enclosure_groups'][0]['center_atom_id'] == center for row in payload)
    assert len(native.interactions[0].metadata['legs']) == 4
    state.set_group_visible(center, False)
    assert not any(state.row_visible(i) for i in range(4))
    state.set_group_visible(center, True)
    assert all(state.row_visible(i) for i in range(4))


def test_shared_contact_visibility_is_union_of_group_memberships():
    first = {'center_atom_id': 1, 'neighbor_atom_ids': [2, 3]}
    second = {'center_atom_id': 2, 'neighbor_atom_ids': [1, 4]}
    rows = [NS(interaction_type='hydrophobic_enclosure', extra={'enclosure_groups': groups})
            for groups in ([first, second], [first], [second])]
    state = view.ContactView(rows)
    assert state.native_count == 2
    state.set_group_visible(1, False)
    assert [state.row_visible(i) for i in range(3)] == [True, False, True]
    state.set_group_visible(2, False)
    assert not any(state.row_visible(i) for i in range(3))
    state.set_visible([0], True, center=1)
    assert state.member_visible(0, 1) and not state.member_visible(0, 2)
    assert state.row_visible(0)


parse = load_functions(ROOT / 'src/cmd.py', ['parse_ligand_smiles'])['parse_ligand_smiles']


def test_full_identifiers_do_not_split_smiles_bonds_or_aromatic_colons():
    assert parse('A:LIG:133A=O=C(O)c1ccccc1;B:LIG:133=C[C:1](O)C') == {
        'A:LIG:133A': 'O=C(O)c1ccccc1', 'B:LIG:133': 'C[C:1](O)C'}
    assert parse('LIG:O=C(O)c1ccccc1') == {'LIG': 'O=C(O)c1ccccc1'}
    assert parse('LIG:c1:c:c:c:c:c:1') == {'LIG': 'c1:c:c:c:c:c:1'}


@pytest.mark.parametrize('spec', ['LIG', 'A:LIG:133', 'LIG:', 'LIG=', ';', '=CC'])
def test_malformed_ligand_corrections_fail_clearly(spec):
    with pytest.raises(ValueError):
        parse(spec)


def test_late_neighbor_selection_changes_only_its_frame_counts_and_fingerprint():
    center, early, late = (NS(selected=False, deleted=False) for _ in range(3))
    late.selected = True
    key = ('hydrophobic_enclosure', frozenset({1}))
    inter = NS(interaction_type=key[0])
    result = NS(frames=[NS(frame_index=i, interactions=[inter]) for i in (0, 1)],
                frame_indices=[0, 1], n_frames=2, interaction_key=lambda _: key,
                _ordered_keys=lambda: [key], interaction_labels=lambda: {key: 'center'})
    funcs = load_functions(ROOT / 'src/trajectory_plots.py', ['_passes_scope', '_compute_counts'],
                           class_name='TrajectoryPlotPanel', globals={'passes_selection': view.passes_selection})
    panel = NS(_result=result, _all_types=[key[0]], _type_included={},
               _frame_endpoints=lambda frame, k: [[center], [early if frame == 0 else late]],
               _scope=lambda: 'any')
    panel._passes_scope = lambda *args: funcs['_passes_scope'](panel, *args)
    assert funcs['_compute_counts'](panel)[1] == [0, 1]
    assert panel._passes_scope(key, 'any')
    assert not panel._passes_scope(key, 'both', 1)
    center.selected = True
    assert panel._passes_scope(key, 'both', 1)
    assert not panel._passes_scope(key, 'both', 0)
    draw = load_functions(ROOT / 'src/trajectory_plots.py', ['draw_fingerprint'])['draw_fingerprint']
    grids = []
    ax = NS(imshow=lambda grid, **kw: grids.append(deepcopy(grid)))
    for name in ('set_yticks', 'set_yticklabels', 'set_xticks', 'set_xticklabels', 'set_xlabel', 'set_title'):
        setattr(ax, name, lambda *a, **kw: None)
    fig = NS(clear=lambda: None, add_subplot=lambda *a: ax, tight_layout=lambda: None)
    draw(fig, result, frame_filter=lambda frame, inter: panel._passes_scope(key, 'both', frame.frame_index))
    np.testing.assert_array_equal(grids[0], [[0, 1]])


def test_selected_hidden_membership_is_not_exported_through_another_group():
    row = NS(interaction_type='hydrophobic_enclosure', extra={'enclosure_groups': [
        {'center_atom_id': 1}, {'center_atom_id': 2}]})
    state = view.ContactView([row])
    state.set_group_visible(1, False)
    hidden = NS(data=lambda *a: ('leg', 0, 1), childCount=lambda: 0)
    parent = NS(data=lambda *a: ('group', 1), childCount=lambda: 1, child=lambda _: hidden)
    methods = load_functions(ROOT / 'src/tool.py', ['_collect_row_indices', '_rows_for_scope'],
                             class_name='ChemurTool', globals={'_USER_ROLE': 1})
    tool = NS(_view=state, _rows=[row], _row_items={0: []}, _row_visible=state.row_visible,
              _export_scope=NS(currentData=lambda: 'rows'),
              _tree=NS(selectedItems=lambda: [parent]))
    tool._collect_row_indices = lambda *a, **kw: methods['_collect_row_indices'](tool, *a, **kw)
    assert state.row_visible(0)
    assert methods['_rows_for_scope'](tool) == []
    tool._tree.selectedItems = lambda: [NS(data=lambda *a: ('leg', 0, 2), childCount=lambda: 0)]
    assert methods['_rows_for_scope'](tool) == [row]


def test_inactive_trajectory_slider_preserves_static_display_owner():
    method = load_functions(ROOT / 'src/tool.py', ['_on_frame_slider'], class_name='ChemurTool')['_on_frame_slider']
    applied = []
    tool = NS(_lifecycle=NS(latest={'trajectory': NS(payload=NS(structure=object()))}, active_run=None),
              _traj_live_cb=NS(isChecked=lambda: False),
              _activate_result=lambda _: pytest.fail('inactive trajectory claimed shared view'),
              _install_coordset_handler=lambda _: pytest.fail('installed inactive playback handler'),
              _apply_frame=lambda *a, **kw: applied.append((a, kw)))
    method(tool, 3)
    assert applied == [((3,), {'set_coordset': True})]


def test_replacing_display_owner_closes_dependent_dialogs_and_hides_old_graphics():
    method = load_functions(ROOT / 'src/tool.py', ['_activate_result'], class_name='ChemurTool')['_activate_result']
    old = NS(mode='trajectory', graphics=[NS(deleted=False, display=True)])
    new = NS(mode='static', graphics=[NS(deleted=False, display=False)],
             payload=NS(result=object(), id_to_atom={}), source_models=[NS(id_string='2', name='new')])
    closed = []
    tool = NS(_lifecycle=NS(disposed=False, display_owner=old, latest={'trajectory': old, 'static': new}),
              _close_dialog=closed.append, _owner_label=NS(setText=lambda text: None),
              _export_assigned_btn=NS(setEnabled=lambda on: None), _remove_coordset_handler=lambda: None)
    tool._lifecycle.activate = lambda state: setattr(tool._lifecycle, 'display_owner', state)
    method(tool, new)
    assert closed == ['_preview_dialog', '_protonation_dialog']
    assert not old.graphics[0].display and new.graphics[0].display
    assert tool._active_result is new.payload.result


def test_graphics_cleanup_survives_reentrant_removal_of_its_input_handles():
    method = load_functions(ROOT / 'src/tool.py', ['_close_graphics'], class_name='ChemurTool')['_close_graphics']
    graphics = [NS(id=(i,), deleted=False) for i in range(3)]
    original = list(graphics)
    def close(models):
        for model in models:
            model.deleted = True
            graphics.remove(model)
    tool = NS(session=NS(models=NS(close=close)))
    method(tool, graphics)
    assert all(model.deleted for model in original) and not graphics


def test_polymer_intrachain_requires_live_chain_and_structure_identity():
    structure, other_structure, chain, other_chain = object(), object(), object(), object()
    def endpoint(kind='protein', source=structure, polymer_chain=chain):
        return NS(molecule_type=kind, chain_id='A',
                  atom=NS(structure=source, residue=NS(chain=polymer_chain)))
    def intrachain(first, second):
        return view.is_polymer_intrachain(NS(endpoint1=first, endpoint2=second))
    assert intrachain(endpoint(), endpoint())
    assert intrachain(endpoint('nucleotide'), endpoint('nucleotide'))
    assert not intrachain(endpoint('ligand'), endpoint())
    assert not intrachain(endpoint('ligand'), endpoint('nucleotide'))
    assert not intrachain(endpoint(source=other_structure), endpoint())
    assert not intrachain(endpoint(polymer_chain=other_chain), endpoint())
    assert not intrachain(endpoint(polymer_chain=None), endpoint(polymer_chain=None))


def test_diagram_unit_visibility_preserves_individual_contact_choices():
    first, second, third = object(), object(), object()
    edges = [NS(unitA=first, unitB=second, interaction_type='hbond'),
             NS(unitA=second, unitB=third, interaction_type='hydrophobic')]
    state = view.DiagramPresentation()
    state.edges[id(edges[0])] = False
    state.units[id(second)] = False
    assert not any(state.edge_visible(edge) for edge in edges)
    state.units[id(second)] = True
    assert [state.edge_visible(edge) for edge in edges] == [False, True]
    state.types['hydrophobic'] = False
    assert not state.edge_visible(edges[1])


def test_zoom_uses_fractional_trackpad_motion_and_limits():
    assert view.zoom_scale(1, 120) == pytest.approx(1.08)
    assert view.zoom_scale(1, -120) == pytest.approx(1 / 1.08)
    assert view.zoom_scale(1, 12) ** 10 == pytest.approx(1.08)
    assert view.zoom_scale(1, pixel_delta=4) ** 10 == pytest.approx(1.08)
    assert view.zoom_scale(1, 0, 0) == 1
    assert view.zoom_scale(8, 120) == 8
    assert view.zoom_scale(.25, -120) == .25


def test_geometry_cutoffs_come_from_versioned_engine_catalogue():
    from chemur.geometry_catalogue import geometry_parameters, GEOMETRY_REVISION
    funcs = load_functions(ROOT / 'src/tool.py', ['_geometry_parameters'], class_name='ChemurTool')
    parameters = funcs['_geometry_parameters'](None)
    assert parameters == geometry_parameters('default')
    values = {(parameter['rule'], parameter['key']): parameter['value'] for parameter in parameters}
    assert values['weak_hbond', 'distance'] == 3.6
    assert values['hbond_pi', 'hydrogen_distance'] == 3.5
    assert values['hbond_pi', 'donor_angle'] == 100
    assert all(parameter.get('label') and parameter.get('description') for parameter in parameters)
    from test_runner_backend import support
    engine = support.require_engine()
    options = support.capture_analysis_options(engine, support.analysis_options(engine))
    assert options['geometry_catalogue_revision'] == GEOMETRY_REVISION
    assert options['include_raw'] is False


@pytest.mark.parametrize('mode', ['static', 'trajectory'])
def test_chemistry_correction_reruns_its_displayed_owner(mode):
    funcs = load_functions(ROOT / 'src/tool.py', ['_rerun_with_ligand_smiles'], class_name='ChemurTool')
    source = object()
    calls = []
    owner = NS(mode=mode, source_models=[source])
    tool = NS(_lifecycle=NS(display_owner=owner, disposed=False), _source_overrides={},
              _model_button=NS(value=None), _input_model_changed=lambda: None,
              _analyze=lambda: calls.append('static'), _analyze_trajectory=lambda: calls.append('trajectory'))
    funcs['_rerun_with_ligand_smiles'](tool, {'A:LIG:1': 'CC'})
    assert calls == [mode] and tool._source_overrides[source] == {'A:LIG:1': 'CC'}

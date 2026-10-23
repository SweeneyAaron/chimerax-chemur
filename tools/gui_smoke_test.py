"""Real Qt checks in a separate GUI ChimeraX session (not --nogui).

ChimeraX --notools --exit --cmd "runscript tools/gui_smoke_test.py /path/1MHT.cif --engine-source /path/chemur/src"
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('structure', type=Path)
parser.add_argument('--engine-source', type=Path)
parser.add_argument('--add-hydrogens', action='store_true',
                    help='exercise the GUI with Add hydrogens enabled')
args = parser.parse_args(sys.argv[1:])
if args.engine_source:
    sys.path.insert(0, str(args.engine_source.resolve()))
root = Path(__file__).resolve().parents[1]
package = types.ModuleType('_chemur_gui_smoke')
package.__path__ = [str(root / 'src')]
sys.modules[package.__name__] = package
module = importlib.import_module(package.__name__ + '.tool')
exports = importlib.import_module(package.__name__ + '.export')
runner = importlib.import_module(package.__name__ + '.runner')
from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
from Qt.QtCore import QTimer

assert session.ui.is_gui, 'This test needs a GUI ChimeraX session'


def open_structure():
    before = set(session.models.list())
    run(session, 'open ' + json.dumps(str(args.structure.resolve())))
    return next(m for m in session.models.list() if m not in before and isinstance(m, AtomicStructure))


def publish(tool, structure, *, add_hydrogens=None):
    tool._model_button.value = structure
    if add_hydrogens is None:
        add_hydrogens = args.add_hydrogens
    tool._addh_cb.setChecked(add_hydrogens)
    tool._skip_bio_cb.setChecked(True)
    heavy_atoms = [(atom, np.asarray(atom.coord).copy()) for atom in structure.atoms
                   if atom.element.number > 1]
    tool._analyze()
    assert tool._static_run is not None, tool._status.text()
    assert tool._lifecycle.active_run is None
    assert all(np.array_equal(atom.coord, original) for atom, original in heavy_atoms), (
        'hydrogen preparation moved heavy atoms')
    native = tool._static_run.result
    atoms = {a.atom_id: a for a in native.atoms}
    group = next(i for i in native.interactions if i.interaction_type == 'hydrophobic_enclosure'
                 and atoms[i.metadata['center_atom_id']].residue_name ==
                 {'4W53': 'MBN', '1MHT': 'SAH'}[args.structure.stem.upper()])
    return group.metadata['center_atom_id']


tool = module.ChemurTool(session, 'Chemur integration test')
structure = open_structure()
old_id = structure.id
center = publish(tool, structure)
indices = tool._view.groups[center]['indices']
assert len(indices) == 4
parent = next(tool._tree.topLevelItem(i) for i in range(tool._tree.topLevelItemCount())
              if tool._tree.topLevelItem(i).data(0, module._USER_ROLE) == ('group', center))
assert parent.childCount() == 4
# Group selection includes all five participants.
tool._tree.setCurrentItem(parent)
tool._tree_selection_changed()
assert sum(a.selected for a in structure.atoms) == 5
parent.child(0).setCheckState(0, module._UNCHECKED)
assert parent.checkState(0) == module.Qt.CheckState.PartiallyChecked
assert not tool._view.member_visible(indices[0], center)
parent.setCheckState(0, module._CHECKED)
assert all(tool._view.member_visible(i, center) for i in indices)
with tempfile.TemporaryDirectory(prefix='chemur-gui-test-') as tmp:
    path, kind = exports.resolve_output_path(str(Path(tmp) / 'assigned'), 'JSON (*.json)')
    assert kind == 'json' and path.suffix == '.json'
    exports.write_native_json(tool._active_result, path)
    assert len(json.loads(path.read_text())['interactions']) == len(tool._active_result.interactions)
    exports.write_json([tool._rows[i] for i in indices], Path(tmp) / 'contacts.json')
    assert len(json.loads((Path(tmp) / 'contacts.json').read_text())) == 4

# Use the same validated scene in the actual interactive diagram renderer.
tool._export_diagram()
assert tool._preview_dialog is not None
diagram_view = tool._preview_dialog._view
assert module.DiagramPreviewDialog._STYLE_OPTIONS == [
    ('Colour', 'default'), ('Black & white', 'bw'), ('Bold', 'bold')]
centers = {id(unit): np.asarray([diagram_view._unit_items[id(unit)].pos().x(),
                               diagram_view._unit_items[id(unit)].pos().y()]) + unit.centroid_px
           for unit in diagram_view._model.units if unit.centroid_px is not None}
diagram_view.set_font_size(24)
for unit in diagram_view._model.units:
    item = diagram_view._unit_items[id(unit)]
    assert item.label_item.font().pointSizeF() == 24
    if id(unit) in centers:
        np.testing.assert_allclose(np.asarray([item.pos().x(), item.pos().y()]) + unit.centroid_px,
                                   centers[id(unit)], atol=1e-6)
for record in diagram_view._edge_recs:
    if record.label is not None:
        assert record.label.font().pointSizeF() == 24
unit = diagram_view._model.units[0]
diagram_view.set_unit_visible(unit, False)
assert not diagram_view._unit_items[id(unit)].isVisible()
assert all(not rec.path.isVisible() for rec in diagram_view._edges_by_unit[id(unit)])
assert len(tool._static_run.result.interactions) == len(tool._active_result.interactions)
diagram_view.set_unit_visible(unit, True)
diagram_view.set_structure_style('bw')
diagram_view.set_font_size(10)
class Wheel:
    def __init__(self, angle=0, pixels=0):
        self.angle, self.pixels, self.accepted = angle, pixels, False
    def angleDelta(self):
        return types.SimpleNamespace(y=lambda: self.angle)
    def pixelDelta(self):
        return types.SimpleNamespace(y=lambda: self.pixels)
    def accept(self):
        self.accepted = True
    def ignore(self):
        self.accepted = False

diagram_view.fit()
for _ in range(10):
    diagram_view.wheelEvent(Wheel(angle=12))
assert abs(diagram_view._relative_zoom - 1.08) < 1e-10
unchanged = diagram_view.transform().m11()
diagram_view.wheelEvent(Wheel())
assert diagram_view.transform().m11() == unchanged
with tempfile.TemporaryDirectory(prefix='chemur-diagram-export-') as tmp:
    for kind, signature in [('png', b'\x89PNG'), ('svg', b'<?xml'), ('pdf', b'%PDF')]:
        path = Path(tmp) / ('diagram.' + kind)
        diagram_view.render_to(path)
        assert path.read_bytes().startswith(signature)
# Midpoint endpoints may involve two different residue depictions. Dragging or
# hiding either contributor must update the same amide-to-ring contact.
tool._close_dialog('_preview_dialog')
amide_rows = [row for row in tool._rows if row.endpoint1.center_atoms]
if args.structure.stem.upper() == '1MHT':
    assert amide_rows, 'Expected an amide-to-ring contact in 1MHT'
if amide_rows:
    row = amide_rows[0]
    tool._select_atoms([*row.endpoint1.center_atoms, row.endpoint2.atom])
    tool._export_diagram()
    assert tool._preview_dialog is not None
    diagram_view = tool._preview_dialog._view
amide_contacts = [rec for rec in diagram_view._edge_recs if rec.edge.centerA] if amide_rows else []
if amide_rows:
    assert amide_contacts, 'Selected amide contact is missing from the diagram'
for rec in amide_contacts:
    def check_midpoint():
        midpoint = np.mean([diagram_view._atom_pos[atom] for atom in rec.edge.centerA], axis=0)
        start = rec.path.path().elementAt(0)
        np.testing.assert_allclose((start.x, start.y), midpoint, atol=1e-6)
    check_midpoint()
    nitrogen_unit = diagram_view._atom_unit[rec.edge.centerA[1]]
    item = diagram_view._unit_items[id(nitrogen_unit)]
    previous_position = item.pos()
    item.setPos(previous_position.x() + 35., previous_position.y() - 20.)
    check_midpoint()
    diagram_view.set_unit_visible(nitrogen_unit, False)
    assert not rec.path.isVisible()
    diagram_view.set_unit_visible(nitrogen_unit, True)
    assert rec.path.isVisible()
    item.setPos(previous_position)
    check_midpoint()
tool._close_dialog('_preview_dialog')

# Exercise actual frame analysis, plotting and ownership when static replaces it.
structure.add_coordset(2, np.asarray(structure.atoms.coords, dtype=np.float64))
tool._traj_live_cb.setChecked(True)
tool._analyze_trajectory()
assert tool._traj_run is not None, tool._traj_status.text()
assert tool._lifecycle.display_owner.mode == 'trajectory'
assert tool._traj_run.result.n_frames == 2
assert tool._rows and tool._traj_plots is not None
with tempfile.TemporaryDirectory(prefix='chemur-trajectory-export-') as tmp:
    target, kind = exports.resolve_output_path(str(Path(tmp) / 'trajectory'), 'JSON (*.json)')
    exports.write_native_json(tool._traj_run.result, target)
    assert json.loads(target.read_text()) == json.loads(tool._traj_run.result.to_json())
# Hydrogen preparation already ran before the extra coordinate set was added.
# ChimeraX addh does not support a structure containing multiple coordinate sets.
publish(tool, structure, add_hydrogens=False)
new_static = tool._active_result
assert tool._lifecycle.display_owner.mode == 'static'
structure.active_coordset_id = 2
session.ui.processEvents()
assert tool._active_result is new_static and tool._lifecycle.display_owner.mode == 'static'
assert tool._coordset_handler is None
tool._traj_live_cb.setChecked(False)
tool._on_frame_slider(0)
assert tool._active_result is new_static and tool._lifecycle.display_owner.mode == "static"

# Failure must preserve the previous result and its drawings.
previous = tool._static_run
handles = list(previous.graphics)
original = runner.run_interactions

def fail(*args, **kwargs):
    raise RuntimeError('intentional smoke-test analysis failure')

runner.run_interactions = fail
try:
    tool._analyze()
finally:
    runner.run_interactions = original
assert tool._static_run is previous
assert all(not g.deleted for g in handles)
assert tool._lifecycle.active_run is None
session.models.close([structure])
assert tool._static_run is None and tool._active_result is None and not tool._rows
assert not tool._lifecycle.latest and all(g.deleted for g in handles)
assert not tool._export_assigned_btn.isEnabled()
reopened = open_structure()
assert reopened is not structure and reopened.id == old_id
assert tool._active_result is None
publish(tool, reopened)

# Close a model while the progress callback pumps GUI events.
token = tool._begin_run('trajectory', [reopened])
QTimer.singleShot(0, lambda: session.models.close([reopened]))
try:
    tool._traj_progress(1, 2)
except module.RunCancelled:
    pass
else:
    raise AssertionError('Closing a model during progress did not cancel')
tool._finish_run(token)
assert tool._active_result is None and tool._lifecycle.active_run is None

# Delete the tool during progress; finally paths must not access dead widgets.
last = open_structure()
token = tool._begin_run('trajectory', [last])
QTimer.singleShot(0, tool.delete)
try:
    tool._traj_progress(1, 2)
except module.RunCancelled:
    pass
else:
    raise AssertionError('Deleting the tool during progress did not cancel')
tool._finish_run(token)
assert tool._lifecycle.disposed and not tool._session_handlers
assert not tool._view.rows and not tool._row_items and tool._traj_plots is None
session.models.close([last])
print('CHEMUR GUI SMOKE PASS')

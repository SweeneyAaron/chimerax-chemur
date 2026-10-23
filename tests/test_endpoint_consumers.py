"""Amide midpoint contacts retain every contributing atom in UI consumers."""

from types import SimpleNamespace as NS

from test_export import export, _Endpoint, _Row
from test_pose_compare import pc, _Inter, _Rec, _Result
from test_view_integration import ROOT, load_functions


def amide_contact():
    def atom(name, number, insertion=""):
        return NS(name=name, selected=False, deleted=False,
                  residue=NS(chain_id="A", number=number, insertion_code=insertion))

    carbon, nitrogen = atom("C", 10), atom("N", 11, "A")
    ligand, ring_neighbor = atom("C1", 99), atom("C2", 99)
    amide = NS(atom=carbon, center_atoms=[carbon, nitrogen], molecule_type="protein",
               chain_id="A", residue_id="10", residue_name="GLY")
    ring = NS(atom=ligand, ring_atoms=[ligand, ring_neighbor], molecule_type="ligand",
              chain_id="A", residue_id="99", residue_name="LIG")
    row = NS(endpoint1=amide, endpoint2=ring, pseudobond=None)
    return row, carbon, nitrogen, ligand, ring_neighbor


def methods(*names):
    return load_functions(ROOT / "src/tool.py", names, class_name="ChemurTool",
                          globals={"_USER_ROLE": 1})


def test_tree_selection_includes_both_amide_atoms_and_ligand_representative():
    row, carbon, nitrogen, ligand, _ = amide_contact()
    item = NS(data=lambda *args: 0, childCount=lambda: 0)
    selected = []
    tool = NS(_lifecycle=NS(disposed=False), _rows=[row],
              _tree=NS(selectedItems=lambda: [item]), _select_atoms=selected.extend)
    callbacks = methods("_tree_selection_changed", "_collect_row_indices")
    tool._collect_row_indices = lambda *args: callbacks["_collect_row_indices"](tool, *args)
    callbacks["_tree_selection_changed"](tool)
    assert [id(atom) for atom in selected] == [id(carbon), id(nitrogen), id(ligand)]


def test_pose_cell_selection_includes_both_amide_atoms():
    row, carbon, nitrogen, ligand, _ = amide_contact()
    selected = []
    column = ("A", "11A")
    comparison = NS(index_for_pose=lambda pose: 0, cell_rows={(0, column): [row]})
    tool = NS(_lifecycle=NS(disposed=False), _pose_comparison=comparison,
              _select_atoms=selected.extend)
    methods("_on_pose_cell_selected")["_on_pose_cell_selected"](tool, object(), column)
    assert [id(atom) for atom in selected] == [id(carbon), id(nitrogen), id(ligand)]


def test_displayed_export_scope_recognizes_cross_residue_amide_nitrogen():
    row, _, nitrogen, _, ring_neighbor = amide_contact()
    tool = NS(_rows=[row], _row_items={0: []}, _row_visible=lambda _: True,
              _export_scope=NS(currentData=lambda: "atoms"))
    export_rows = methods("_rows_for_scope")["_rows_for_scope"]
    assert export_rows(tool) == []
    nitrogen.selected = True
    assert export_rows(tool) == [row]
    nitrogen.selected = False
    ring_neighbor.selected = True
    assert export_rows(tool) == []  # Preserve existing ring representative selection.


def test_deleted_amide_nitrogen_hides_stale_row():
    row, _, nitrogen, _, _ = amide_contact()
    tool = NS(_rows=[row], _view=NS(row_visible=lambda _: True), _interchain_only=False)
    visible = methods("_row_visible")["_row_visible"]
    assert visible(tool, 0)
    nitrogen.deleted = True
    assert not visible(tool, 0)


def test_export_dedup_uses_complete_midpoint_group():
    carbon = _Endpoint("protein", "A", "GLY", "10", "C")
    nitrogen = _Endpoint("protein", "A", "ALA", "11A", "N")
    other_nitrogen = _Endpoint("protein", "A", "SER", "11B", "N")
    ligand = _Endpoint("ligand", "A", "LIG", "99", "C1")
    carbon.center_atoms = [carbon.atom, nitrogen.atom]
    second = _Endpoint("protein", "A", "GLY", "10", "C")
    second.atom = carbon.atom
    second.center_atoms = [carbon.atom, other_nitrogen.atom]
    nitrogen.center_atoms = [nitrogen.atom, carbon.atom]
    def row(endpoint):
        return _Row("amide_pi", endpoint, ligand, 3.4, None, None, {"amide_center": [1, 2, 3]})
    records = export.rows_to_records([row(carbon), row(second), row(nitrogen)])
    assert len(records) == 2
    assert all(record["metadata"]["amide_center"] == [1, 2, 3] for record in records)


def test_cross_residue_amide_pose_cells_share_mapped_row():
    row, _, _, _, _ = amide_contact()
    atoms = [_Rec(1, "protein", "A", "10", "GLY"),
             _Rec(2, "protein", "A", "11A", "ALA"),
             _Rec(3, "ligand", "A", "99", "LIG")]
    interaction = _Inter("amide_pi", [1, 2, 3])
    row.interaction = interaction
    hits = pc._hits_from_result(_Result(atoms, [interaction]), [row])
    assert [hit.residue_id for hit in hits] == ["10", "11A"]
    assert all(hit.row is row for hit in hits)
    _, _, cells, _, _, _ = pc.build_matrix([(object(), "pose", True)], [hits])
    assert cells == {(0, ("A", "10")): [row], (0, ("A", "11A")): [row]}

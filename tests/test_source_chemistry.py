"""Named source dictionaries survive live export without replacing live structure."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import gemmi
import pytest

spec = importlib.util.spec_from_file_location('_chemur_source_chemistry',
    Path(__file__).parents[1] / 'src' / 'source_chemistry.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
preserve = module.preserve_source_chemistry


def dictionary(charge='0', comp='LIG'):
    return {
        '_chem_comp_atom.': {
            'comp_id': [comp] * 3, 'atom_id': ['C1', 'O1', 'H1'],
            'type_symbol': ['C', 'O', 'H'], 'charge': ['0', charge, '0'],
            'pdbx_leaving_atom_flag': ['N', 'Y', 'N']},
        '_chem_comp_bond.': {'comp_id': [comp] * 2,
            'atom_id_1': ['C1', 'O1'], 'atom_id_2': ['O1', 'H1'],
            'value_order': ['SING', 'SING']},
        '_pdbx_chem_comp_descriptor.': {'comp_id': [comp], 'type': ['SMILES'],
            'descriptor': ['CO' if charge == '0' else 'C[O-]']},
        '_chem_comp.': {'id': [comp], 'type': ['non-polymer'],
            'name': ['SOURCE NAME'], 'pdbx_smiles': ['CO']},
    }


def write_source(path, tables=None):
    document = gemmi.cif.Document()
    block = document.add_new_block('deposited')
    for category, values in (dictionary() if tables is None else tables).items():
        block.set_mmcif_category(category, values)
    # Source coordinates and connectivity must never overwrite live exports.
    block.set_mmcif_category('_atom_site.', {'id': ['1'], 'Cartn_x': ['-999']})
    block.set_mmcif_category('_struct_conn.', {'id': ['SOURCE LINK']})
    document.write_file(str(path))
    return path


def snapshot(path, count=1, tables=None):
    document = gemmi.cif.Document()
    for number in range(count):
        block = document.add_new_block('live%d' % number)
        block.set_mmcif_category('_atom_site.', {'id': ['1'], 'Cartn_x': [str(10 + number)]})
        block.set_mmcif_category('_struct_conn.', {'id': ['LIVE LINK %d' % number]})
        block.set_mmcif_category('_chem_comp.', {'id': ['LIG'], 'type': ['non-polymer'],
                                                'name': ['LIVE NAME']})
        for category, values in (tables or {}).items():
            block.set_mmcif_category(category, values)
    document.write_file(str(path))
    return path


def model(source=None, metadata=None, name='model'):
    residue = NS(name='LIG', atoms=[])
    carbon = NS(name='C1', element=NS(name='C'), residue=residue, neighbors=[])
    oxygen = NS(name='O1', element=NS(name='O'), residue=residue, neighbors=[])
    carbon.neighbors = [oxygen]
    oxygen.neighbors = [carbon]
    residue.atoms = [carbon, oxygen]
    return NS(name=name, residues=[residue], filename=source, is_mmcif=True,
              metadata=metadata or {})


def metadata(tables):
    result = {}
    for category, values in tables.items():
        key = category.strip('_.')
        result[key] = [key, *values]
        result[key + ' data'] = [value for row in zip(*values.values()) for value in row]
    return result


class Logger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


def category(path, name, block=0):
    return gemmi.cif.read_file(str(path))[block].get_mmcif_category(name)


def test_local_source_restores_full_named_chemistry_without_touching_live_data(tmp_path):
    source = write_source(tmp_path / 'source.cif')
    output = snapshot(tmp_path / 'live.cif')
    structure = model(source)
    original_metadata = deepcopy(structure.metadata)
    preserve(output, [structure])
    assert category(output, '_chem_comp_atom.') == dictionary()['_chem_comp_atom.']
    assert category(output, '_chem_comp_bond.') == dictionary()['_chem_comp_bond.']
    assert category(output, '_pdbx_chem_comp_descriptor.') == dictionary()['_pdbx_chem_comp_descriptor.']
    assert category(output, '_atom_site.')['Cartn_x'] == ['10']
    assert category(output, '_struct_conn.')['id'] == ['LIVE LINK 0']
    assert category(output, '_chem_comp.')['name'] == ['LIVE NAME']
    assert category(output, '_chem_comp.')['pdbx_smiles'] == ['CO']
    assert structure.metadata == original_metadata


def test_metadata_dictionary_takes_precedence_over_local_source(tmp_path):
    source = write_source(tmp_path / 'source.cif', dictionary(charge='-1'))
    structure = model(source, metadata(dictionary()))
    output = snapshot(tmp_path / 'live.cif')
    preserve(output, [structure])
    assert category(output, '_chem_comp_atom.')['charge'] == ['0', '0', '0']
    assert category(output, '_pdbx_chem_comp_descriptor.')['descriptor'] == ['CO']


def test_live_export_dictionary_is_authoritative(tmp_path):
    source = write_source(tmp_path / 'source.cif')
    output = snapshot(tmp_path / 'live.cif', tables=dictionary(charge='-1'))
    before = output.read_bytes()
    preserve(output, [model(source)])
    assert output.read_bytes() == before


def test_added_hydrogens_do_not_prevent_named_heavy_chemistry(tmp_path):
    source = write_source(tmp_path / 'source.cif')
    structure = model(source)
    residue = structure.residues[0]
    carbon = residue.atoms[0]
    hydrogen = NS(name='NEW_H', element=NS(name='H'), residue=residue, neighbors=[carbon])
    carbon.neighbors.append(hydrogen)
    residue.atoms.append(hydrogen)
    output = snapshot(tmp_path / 'live.cif')
    preserve(output, [structure])
    assert category(output, '_chem_comp_atom.')['atom_id'] == ['C1', 'O1', 'H1']
    assert [atom.name for atom in residue.atoms] == ['C1', 'O1', 'NEW_H']


@pytest.mark.parametrize('mutation', ['rename', 'element', 'remove_bond', 'add_atom'])
def test_chemically_edited_ligands_do_not_receive_stale_dictionary(tmp_path, mutation):
    source = write_source(tmp_path / 'source.cif')
    structure = model(source)
    residue = structure.residues[0]
    carbon, oxygen = residue.atoms
    if mutation == 'rename':
        carbon.name = 'OTHER'
    elif mutation == 'element':
        oxygen.element.name = 'N'
    elif mutation == 'remove_bond':
        carbon.neighbors = []
        oxygen.neighbors = []
    else:
        residue.atoms.append(NS(name='EXTRA', element=NS(name='C'), residue=residue, neighbors=[]))
    output = snapshot(tmp_path / 'live.cif')
    logger = Logger()
    preserve(output, [structure], logger)
    assert not category(output, '_chem_comp_atom.')
    assert len(logger.messages) == 1
    assert 'did not restore LIG' in logger.messages[0]


def test_distinct_source_models_keep_conflicting_named_definitions(tmp_path):
    first = model(write_source(tmp_path / 'neutral.cif'))
    second = model(write_source(tmp_path / 'charged.cif', dictionary(charge='-1')))
    output = snapshot(tmp_path / 'live.cif', count=2)
    preserve(output, [first, second])
    assert category(output, '_chem_comp_atom.', 0)['charge'] == ['0', '0', '0']
    assert category(output, '_chem_comp_atom.', 1)['charge'] == ['0', '-1', '0']


def test_coalesced_export_uses_dictionary_blocks_without_guessing_ownership(tmp_path):
    first = model(write_source(tmp_path / 'neutral.cif'))
    second = model(write_source(tmp_path / 'charged.cif', dictionary(charge='-1')))
    output = snapshot(tmp_path / 'live.cif')
    preserve(output, [first, second])
    document = gemmi.cif.read_file(str(output))
    assert len(document) == 3
    assert document[0].get_mmcif_category('_atom_site.')['Cartn_x'] == ['10']
    assert document[1].get_mmcif_category('_chem_comp_atom.')['charge'] == ['0', '0', '0']
    assert document[2].get_mmcif_category('_chem_comp_atom.')['charge'] == ['0', '-1', '0']
    assert not document[1].get_mmcif_category('_atom_site.')


def test_missing_source_is_reported_and_snapshot_remains_usable(tmp_path):
    output = snapshot(tmp_path / 'live.cif')
    before = output.read_bytes()
    logger = Logger()
    preserve(output, [model(tmp_path / 'missing.cif')], logger)
    assert output.read_bytes() == before
    assert len(logger.messages) == 1
    assert 'Could not retain source chemical dictionaries' in logger.messages[0]


def test_unrelated_component_dictionary_is_not_copied(tmp_path):
    source = write_source(tmp_path / 'source.cif', dictionary(comp='OTHER'))
    output = snapshot(tmp_path / 'live.cif')
    preserve(output, [model(source)])
    assert not category(output, '_chem_comp_atom.')


def test_conflicting_multiblock_source_is_preserved_for_engine_validation(tmp_path):
    source = write_source(tmp_path / 'source.cif')
    document = gemmi.cif.read_file(str(source))
    block = document.add_new_block('alternative')
    for name, data in dictionary(charge='-1').items():
        block.set_mmcif_category(name, data)
    document.write_file(str(source))
    output = snapshot(tmp_path / 'live.cif')
    preserve(output, [model(source)])
    assert category(output, '_chem_comp_atom.')['charge'] == ['0', '0', '0', '-1']


def test_absent_leaving_atom_keeps_full_dictionary_for_engine_validation(tmp_path):
    source = write_source(tmp_path / 'source.cif')
    structure = model(source)
    residue = structure.residues[0]
    residue.atoms = residue.atoms[:1]
    residue.atoms[0].neighbors = []
    output = snapshot(tmp_path / 'live.cif')
    preserve(output, [structure])
    assert category(output, '_chem_comp_atom.')['atom_id'] == ['C1', 'O1', 'H1']
    assert category(output, '_chem_comp_atom.')['pdbx_leaving_atom_flag'] == ['N', 'Y', 'N']


def test_conflicting_smiles_definitions_have_explicit_diagnostic(tmp_path):
    first = model(write_source(tmp_path / 'first.cif'))
    alternative = dictionary()
    alternative['_pdbx_chem_comp_descriptor.']['descriptor'] = ['C[O-]']
    second = model(write_source(tmp_path / 'second.cif', alternative))
    output = snapshot(tmp_path / 'live.cif', count=2)
    logger = Logger()
    preserve(output, [first, second], logger)
    assert len(logger.messages) == 1
    assert 'conflicting deposited dictionaries for LIG' in logger.messages[0]
    assert category(output, '_pdbx_chem_comp_descriptor.', 0)['descriptor'] == ['CO']
    assert category(output, '_pdbx_chem_comp_descriptor.', 1)['descriptor'] == ['C[O-]']


@pytest.mark.parametrize('reverse', [False, True])
def test_source_dictionary_cannot_reach_an_incompatible_same_name_model(tmp_path, reverse):
    """The normalizer merges named dictionaries globally across coordinate blocks."""
    original = model(write_source(tmp_path / 'source.cif'), name='unchanged')
    edited = model(name='edited without source')
    for atom in edited.residues[0].atoms:
        atom.neighbors = []
    structures = [original, edited]
    if reverse:
        structures.reverse()
    output = snapshot(tmp_path / 'live.cif', count=2)
    logger = Logger()
    preserve(output, structures, logger)
    assert not category(output, '_chem_comp_atom.', 0)
    assert not category(output, '_chem_comp_atom.', 1)
    assert len(logger.messages) == 1
    assert 'live heavy bonds differ' in logger.messages[0]


def test_preserve_polymer_role_sequence_and_scoped_source_entry(tmp_path):
    source = tmp_path / 'deposited.cif'
    deposited = gemmi.cif.Document()
    block = deposited.add_new_block('deposited')
    block.set_pair('_entry.id', '1MHT')
    block.set_mmcif_category('_pdbx_struct_mod_residue.', {
        'id': ['1'], 'label_asym_id': ['A'], 'label_comp_id': ['C36'], 'label_seq_id': ['6'],
        'auth_asym_id': ['C'], 'auth_comp_id': ['C36'], 'auth_seq_id': ['427'],
        'PDB_ins_code': [False], 'parent_comp_id': ['DC'], 'details': ['modified cytidine']})
    block.set_mmcif_category('_entity_poly.', {'entity_id': ['1'], 'type': ['polydeoxyribonucleotide']})
    block.set_mmcif_category('_entity_poly_seq.', {'entity_id': ['1', '1'], 'num': ['5', '6'],
        'mon_id': ['DC', 'C36'], 'hetero': ['n', 'n']})
    rows = dict(group_PDB=['ATOM', 'HETATM'], id=['1', '2'], type_symbol=['O', 'P'],
        label_atom_id=["O3'", 'P'], label_alt_id=['.', '.'], label_comp_id=['DC', 'C36'],
        label_asym_id=['A', 'A'], label_entity_id=['1', '1'], label_seq_id=['5', '6'],
        auth_atom_id=["O3'", 'P'], auth_comp_id=['DC', 'C36'], auth_asym_id=['C', 'C'],
        auth_seq_id=['426', '427'], pdbx_PDB_ins_code=[False, False],
        Cartn_x=['0', '1'], Cartn_y=['0', '0'], Cartn_z=['0', '0'], occupancy=['1', '1'],
        B_iso_or_equiv=['10', '10'], pdbx_PDB_model_num=['1', '1'])
    block.set_mmcif_category('_atom_site.', rows)
    deposited.write_file(str(source))
    output = tmp_path / 'snapshot.cif'
    live = gemmi.cif.Document()
    target = live.add_new_block('live')
    target.set_mmcif_category('_atom_site.', {**rows, 'Cartn_x': ['10', '11'],
        'label_entity_id': ['9', '10'], 'label_asym_id': ['X', 'Y'], 'label_seq_id': ['1', False]})
    live.write_file(str(output))
    residues = []
    for name, number, atom_name, element, x in [('DC', 426, "O3'", 'O', 10), ('C36', 427, 'P', 'P', 11)]:
        residue = NS(name=name, number=number, chain_id='C', insertion_code='', atoms=[])
        atom = NS(name=atom_name, element=NS(name=element), scene_coord=(x, 0, 0),
                  residue=residue, neighbors=[])
        residue.atoms = [atom]
        residues.append(residue)
    structure = NS(residues=residues, filename=str(source), metadata={}, name='live', is_mmcif=True)
    preserve(output, [structure])
    atom_data = category(output, '_atom_site.')
    assert atom_data['label_seq_id'] == ['5', '6']
    assert len(set(atom_data['label_entity_id'])) == 1
    assert len(set(atom_data['label_asym_id'])) == 1
    assert atom_data['Cartn_x'] == ['10', '11']
    provenance = category(output, '_chemur_source_residue.')
    assert provenance['source_entry_id'] == ['1MHT', '1MHT']
    assert provenance['auth_seq_id'] == ['426', '427']
    from chemur.parser import parse_structure
    atoms, components = parse_structure(output)
    modified = next(component for component in components if component.name == 'C36')
    assert modified.molecule_type == 'nucleotide'
    assert modified.metadata['source_entry_id'] == '1MHT'
    assert modified.metadata['source_polymer_sequence_index'] == 6
    assert modified.metadata['modified_polymer'] is True
    assert modified.metadata['parent_component_id'] == 'DC'
    assert category(output, '_pdbx_struct_mod_residue.')['label_asym_id'] == [atom_data['label_asym_id'][1]]

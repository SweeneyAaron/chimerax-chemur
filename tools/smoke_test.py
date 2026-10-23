"""Exercise the staged bridge in headless ChimeraX with two local structures.

ChimeraX --nogui --exit --cmd "runscript tools/smoke_test.py /path/4W53.cif /path/1MHT.cif"

Optional: --engine-source /path/chemur/src --output-dir /path/reports
Without --output-dir, exports use an automatically removed temporary directory.
The script raises on failure and prints CHEMUR SMOKE PASS only after every check.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('structures', nargs=2, type=Path, help='local 4W53 and 1MHT coordinate files')
parser.add_argument('--engine-source', type=Path)
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args(sys.argv[1:])
if args.engine_source:
    sys.path.insert(0, str(args.engine_source.resolve()))

root = Path(__file__).resolve().parents[1]
package_name = '_chemur_staged_smoke'
package = types.ModuleType(package_name)
package.__path__ = [str(root / 'src')]
sys.modules[package_name] = package
runner = importlib.import_module(package_name + '.runner')
exports = importlib.import_module(package_name + '.export')
support = importlib.import_module(package_name + '.engine_support')
support.require_engine()

from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
from chimerax.geometry import translation, rotation

CENTERS = {'4W53': ('A', 'MBN', '200', 'C'), '1MHT': ('A', 'SAH', '328', 'C4')}


def identity(atom):
    return (atom.chain_id, atom.residue_name, str(atom.residue_id), atom.name)


def expected_legs(code):
    payload = json.loads((root / 'tests' / 'fixtures' / (code + '_enclosure.json')).read_text())
    atoms = {a['atom_id']: types.SimpleNamespace(**a) for a in payload['atoms']}
    return {frozenset(identity(atoms[aid]) for aid in leg['atom_ids'])
            for leg in payload['interactions'][0]['metadata']['legs']}


def check_group(run_result, code):
    atoms = {a.atom_id: a for a in run_result.result.atoms}
    groups = [inter for inter in run_result.result.interactions
              if inter.interaction_type == 'hydrophobic_enclosure'
              and identity(atoms[inter.metadata['center_atom_id']]) == CENTERS[code]]
    assert len(groups) == 1, (code, 'expected one ligand-centered enclosure', len(groups))
    group = groups[0]
    expected = expected_legs(code)
    actual = {frozenset(identity(atoms[aid]) for aid in leg['atom_ids']) for leg in group.metadata['legs']}
    assert actual == expected and len(actual) == 4, (code, 'native leg identities', actual, expected)
    displayed = [row for row in run_result.rows if any(
        metadata['center_atom_id'] == group.metadata['center_atom_id']
        for metadata in row.extra.get('enclosure_groups', []))]
    assert len(displayed) == 4, (code, 'displayed leg count', len(displayed))
    assert {frozenset(identity(atoms[aid]) for aid in row.atom_ids) for row in displayed} == expected
    assert all(row.pseudobond is not None for row in displayed), (code, 'missing pseudobond')
    assert all(atoms[aid].molecule_type == 'ligand' or atoms[aid].molecule_type == 'protein'
               for row in displayed for aid in row.atom_ids)
    return displayed, group


def check_measured_endpoints(analyzed):
    """The drawn segment must use the geometry reported by the engine."""
    for row in analyzed.rows:
        if row.interaction_type not in {"salt_bridge", "amide_pi"}:
            continue
        assert row.pseudobond is not None, (row.interaction_type, "missing pseudobond")
        if row.interaction_type == "salt_bridge":
            expected = [analyzed.id_to_atom[aid] for aid in row.extra["distance_atom_ids"]]
            assert [row.endpoint1.atom, row.endpoint2.atom] == expected
        else:
            assert row.endpoint1.center_atoms and row.endpoint2.ring_atoms
        first, second = row.pseudobond.atoms
        distance = np.linalg.norm(np.asarray(first.scene_coord) - np.asarray(second.scene_coord))
        assert abs(distance - row.distance) < .005, (row.interaction_type, distance, row.distance)


def smoke_one(path, code, output, *, add_hydrogens=False):
    before = set(session.models.list())
    run(session, 'open ' + json.dumps(str(path.resolve())))
    opened = [m for m in session.models.list() if m not in before and isinstance(m, AtomicStructure)]
    assert len(opened) == 1, (code, 'expected one structure', opened)
    structure = opened[0]
    heavy_atoms = [(atom, np.asarray(atom.coord).copy()) for atom in structure.atoms
                   if atom.element.number > 1]
    hydrogen_count = sum(atom.element.number == 1 for atom in structure.atoms)
    kwargs = dict(add_hydrogens=add_hydrogens, skip_biopolymer_internal=True)
    analyzed = runner.run_interactions(session, structure, **kwargs)
    assert all(np.array_equal(atom.coord, original) for atom, original in heavy_atoms), (
        code, 'hydrogen preparation moved heavy atoms')
    if add_hydrogens:
        assert sum(atom.element.number == 1 for atom in structure.atoms) > hydrogen_count, (
            code, 'hydrogen-enabled analysis did not add hydrogens')
    assert analyzed.analysis_options['include_raw'] is False
    displayed, group = check_group(analyzed, code)
    check_measured_endpoints(analyzed)
    component = ":".join(CENTERS[code][:3])
    assert analyzed.result.metadata["ligand_assemblies"][component]["status"] == "validated"
    label = code + ('_addh' if add_hydrogens else '_noaddh')
    paths = {name: output / (label + '_' + name) for name in
             ('native.json', 'native.csv', 'displayed.json', 'displayed.csv')}
    exports.write_native_json(analyzed.result, paths['native.json'])
    exports.write_native_csv(analyzed.result, paths['native.csv'])
    exports.write_json(displayed, paths['displayed.json'])
    exports.write_csv(displayed, paths['displayed.csv'])
    native = json.loads(paths['native.json'].read_text())
    view = json.loads(paths['displayed.json'].read_text())
    assert len(native['interactions']) == len(analyzed.result.interactions)
    assert any(inter['metadata'].get('center_atom_id') == group.metadata['center_atom_id']
               for inter in native['interactions'] if inter['interaction_type'] == 'hydrophobic_enclosure')
    assert len(view) == 4 and all(item['enclosure_groups'] for item in view)
    with paths['displayed.csv'].open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 4 and all(json.loads(row['enclosure_groups']) for row in csv_rows)
    old_graphics = list(analyzed.graphics)
    structure.position = translation((12.0, -7.0, 4.0)) * rotation((1, 2, 3), 31) * structure.position
    transformed = runner.run_interactions(session, structure, **kwargs)
    check_group(transformed, code)
    check_measured_endpoints(transformed)
    assert all(getattr(g, 'deleted', False) for g in old_graphics), (code, 'command graphics were not replaced')
    for record in transformed.result.atoms:
        atom = transformed.id_to_atom.get(record.atom_id)
        if atom is not None:
            assert sum((float(a) - float(b)) ** 2 for a, b in zip(record.coord, atom.scene_coord)) < 0.05 ** 2
    handles = list(transformed.graphics)
    session.models.close([structure])
    assert all(getattr(g, 'deleted', False) for g in handles), (code, 'graphics survived source closure')
    print(label, 'PASS: 4 native/display legs, unchanged heavy coordinates, JSON+CSV exports, '
          'transformed mapping, source-close cleanup')


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    for path in args.structures:
        code = path.stem.upper()
        if code not in CENTERS:
            raise ValueError('Name the local files 4W53.cif and 1MHT.cif .')
        if not path.is_file():
            raise FileNotFoundError(path)
        smoke_one(path, code, output)
        if code == '1MHT':
            # Reopen the original file so the two hydrogen settings are independent.
            # SAH:C4 must retain the same four legs when PHE18:HD1 is added.
            smoke_one(path, code, output, add_hydrogens=True)
    assert {p.stem.upper() for p in args.structures} == set(CENTERS), 'Provide both different structures'
    print('CHEMUR SMOKE PASS')


if args.output_dir:
    main(args.output_dir.resolve())
else:
    with tempfile.TemporaryDirectory(prefix='chemur-smoke-') as temporary:
        main(Path(temporary))

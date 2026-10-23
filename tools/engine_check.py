"""Report the active Chemur build and required capabilities inside ChimeraX.

    ChimeraX --nogui --exit --cmd "runscript tools/engine_check.py"

For an explicit developer source check, append ``--engine-source /path/to/chemur/src``.
The default checks the engine ChimeraX actually imports; it never installs anything.
"""
import argparse
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine-source', type=Path)
args = parser.parse_args(sys.argv[1:])
if args.engine_source:
    sys.path.insert(0, str(args.engine_source.resolve()))

capabilities = {
    'static assembly overrides': ('chemur', 'analyze', 'assembly_overrides'),
    'trajectory assembly overrides': ('chemur.trajectory', 'analyze_trajectory', 'assembly_overrides'),
    'array trajectory source': ('chemur.trajectory', 'ArrayFrameSource', None),
    'enclosure leg projection': ('chemur.interaction_projection', 'project_interactions_for_scoring', None),
    'ligand assembly selection': ('chemur.ligand_selection', 'ligand_selections', None),
    'shared chemistry preparation': ('chemur.preparation', 'prepare_ligand_chemistry', None),
    'RDKit atom typing': ('chemur.chemistry', 'rdkit_atom_annotations', None),
    'structure parser': ('chemur.parser', 'parse_structure', None),
    'typed acceptor features': ('chemur.features', 'perceive_features', None),
    'geometry parameter catalogue': ('chemur.geometry_catalogue', 'geometry_parameters', None),
    'geometry definitions revision': ('chemur.geometry_catalogue', 'GEOMETRY_REVISION', None),
}
try:
    import chemur
    from chemur.core import USING_CPP_CORE
    print('chemur version:', getattr(chemur, '__version__', 'unknown'))
    print('chemur path:', chemur.__file__)
    print('backend:', 'C++' if USING_CPP_CORE else 'Python')
except ImportError as error:
    print('chemur import: FAILED:', error)

for label, (module_name, symbol, parameter) in capabilities.items():
    try:
        value = getattr(importlib.import_module(module_name), symbol)
        present = parameter is None or parameter in inspect.signature(value).parameters
    except (ImportError, AttributeError, ValueError):
        present = False
    print('%s: %s' % (label, 'OK' if present else 'MISSING'))
try:
    from chemur.profile import load_profile
    present = 'hydrophobic_enclosure' in load_profile('default')['rules']
except (ImportError, KeyError, AttributeError):
    present = False
print('hydrophobic enclosure rule:', 'OK' if present else 'MISSING')
try:
    from chemur.geometry_catalogue import GEOMETRY_REVISION
    print('geometry definitions:', GEOMETRY_REVISION)
    from chemur.preparation import MODIFIED_POLYMER_CHEMISTRY
    print('modified-polymer chemistry:', 'OK' if MODIFIED_POLYMER_CHEMISTRY else 'MISSING')
except ImportError:
    print('geometry definitions / modified-polymer chemistry: MISSING')
try:
    from chemur.interactions import HYDROPHOBIC_ENDPOINT_HYDROGEN_EXEMPTION
    present = HYDROPHOBIC_ENDPOINT_HYDROGEN_EXEMPTION is True
except ImportError:
    present = False
print('hydrophobic endpoint-hydrogen occlusion exemption:', 'OK' if present else 'MISSING')

# Load the shared gate directly from this checkout, even before bundle install.
path = Path(__file__).resolve().parents[1] / 'src' / 'engine_support.py'
spec = importlib.util.spec_from_file_location('_chemur_engine_check_support', path)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
support.require_engine()
print('Chemur capability gate: PASS')

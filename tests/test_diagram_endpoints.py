"""Midpoint endpoints retain both residues through layout and visibility."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from chemur.models import AtomRecord
from test_diagram_chemistry import Atom, Bond, Residue, diagram


def amide_scene(monkeypatch, *, split=True):
    first = Residue("ALA" if split else "ASN", 1)
    second = Residue("GLY", 2) if split else first
    ligand = Residue("LIG", 9)

    def atom(aid, name, element, coord, residue):
        return Atom(AtomRecord(aid, name, element, coord, residue.name,
                               str(residue.number), "A", "A:%s:%s" % (residue.name, residue.number),
                               "ligand" if residue is ligand else "protein"), residue)

    carbon = atom(0, "C", "C", (-.65, 0., 0.), first)
    nitrogen = atom(1, "N", "N", (.65, 0., 0.), second)
    oxygen = atom(2, "O", "O", (-.65, 1.2, 0.), first)
    Bond(carbon, nitrogen)
    Bond(carbon, oxygen)
    ring = [atom(i + 3, "C%d" % i, "C", (np.cos(i * np.pi / 3), np.sin(i * np.pi / 3), 4.), ligand)
            for i in range(6)]
    for index, member in enumerate(ring):
        Bond(member, ring[(index + 1) % len(ring)])
    row = NS(endpoint1=NS(atom=carbon, molecule_type="protein", center_atoms=(carbon, nitrogen)),
             endpoint2=NS(atom=ring[0], molecule_type="ligand", ring_atoms=ring),
             interaction_type="amide_pi", distance=4., pseudobond=None)
    orders = {frozenset((diagram._key(carbon), diagram._key(oxygen))): "2"}
    monkeypatch.setattr(diagram, "_mol2_bond_orders", lambda *args: orders)
    # Selecting the nitrogen residue must include this contact too.
    scene = diagram.build_scene(None, None, [row], {second})
    return scene, carbon, nitrogen, ring


@pytest.mark.parametrize("split", [False, True])
def test_amide_diagram_routes_midpoint_to_ring_and_tracks_all_units(monkeypatch, split):
    scene, carbon, nitrogen, ring = amide_scene(monkeypatch, split=split)
    edge, = scene.edges
    assert len(scene.units) == (3 if split else 2)
    assert set(edge.units) == set(scene.units)
    assert edge.centerA == (carbon, nitrogen)
    assert set(edge.ringB) == set(ring)

    def assert_endpoints():
        routes = diagram._visible_scene_geometry(scene)[2]
        assert len(routes) == 1
        points = routes[0][1]
        np.testing.assert_allclose(points[0], (scene.atom_pos[carbon] + scene.atom_pos[nitrogen]) / 2)
        np.testing.assert_allclose(points[-1], np.mean([scene.atom_pos[a] for a in ring], axis=0))

    assert_endpoints()
    # A drag of the second residue must move the midpoint, even though the
    # representative endpoint atom belongs to the first residue.
    unit = scene.atom_unit[nitrogen]
    for member in unit.atom_to_local:
        scene.atom_pos[member] += np.array([70., -35.])
    unit.global_origin += np.array([70., -35.])
    assert_endpoints()
    scene.presentation.units[id(unit)] = False
    assert not scene.presentation.edge_visible(edge)
    assert diagram._visible_scene_geometry(scene)[2] == []
    scene.presentation.units[id(unit)] = True
    assert_endpoints()


def test_incomplete_midpoint_never_falls_back_to_representative_atom(monkeypatch):
    scene, carbon, nitrogen, _ = amide_scene(monkeypatch)
    positions = dict(scene.atom_pos)
    positions.pop(nitrogen)
    assert diagram._endpoint_pos(carbon, None, positions, np, (carbon, nitrogen)) is None

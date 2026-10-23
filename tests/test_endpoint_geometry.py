"""Measured contact geometry survives Chemur-to-ChimeraX endpoint mapping."""
from dataclasses import replace
from math import cos, pi, sin
import sys
from types import SimpleNamespace as NS

import pytest

from chemur.interactions import DetectorContext, _amide_pi_candidates, _salt_bridge_candidates
from chemur.models import AtomRecord, FeatureRecord
from chemur.profile import load_profile
from test_runner_mapping import runner, _CxAtom, _Residue


def _detect(kind, atoms, features):
    grouped = {}
    for feature in features:
        grouped.setdefault(feature.feature_type, []).append(feature)
    profile = load_profile()
    context = DetectorContext({atom.atom_id: atom for atom in atoms}, grouped, profile)
    detector = _salt_bridge_candidates if kind == "salt_bridge" else _amide_pi_candidates
    hit, = detector(context, profile["rules"][kind])
    mapped = {}
    for atom in atoms:
        residue = _Residue(atom.chain_id, int(atom.residue_id), atom.residue_name)
        live = _CxAtom(atom.name, atom.coord, residue)
        live.scene_coord = atom.coord
        live.selected = False
        mapped[atom.atom_id] = live
    return hit, {atom.atom_id: atom for atom in atoms}, mapped, {
        feature.feature_id: feature for feature in features}


@pytest.fixture
def salt():
    # Guanidinium geometry puts NH1 closest to the carboxylate, while the CZ
    # atom appears first in the feature and native interaction atom IDs.
    atoms = [
        AtomRecord(0, "CZ", "C", (0., 0., 0.), "ARG", "1", "A", "A:ARG:1", "protein"),
        AtomRecord(1, "NH1", "N", (1.15, .665, 0.), "ARG", "1", "A", "A:ARG:1", "protein"),
        AtomRecord(2, "NH2", "N", (-1.15, .665, 0.), "ARG", "1", "A", "A:ARG:1", "protein"),
        AtomRecord(3, "NE", "N", (0., -1.33, 0.), "ARG", "1", "A", "A:ARG:1", "protein"),
        AtomRecord(4, "OD1", "O", (4.15, .665, 0.), "ASP", "2", "A", "A:ASP:2", "protein"),
    ]
    features = [
        FeatureRecord(0, "cation", "A:ARG:1", (0, 1, 2, 3), (0., 0., 0.),
                      metadata={"source": "protein_arg"}),
        FeatureRecord(1, "anion", "A:ASP:2", (4,), atoms[-1].coord),
    ]
    return _detect("salt_bridge", atoms, features)


@pytest.fixture
def amide():
    atoms = [
        AtomRecord(0, "C", "C", (0., 0., 0.), "ALA", "1", "A", "A:ALA:1", "protein"),
        AtomRecord(1, "O", "O", (0., 1.23, 0.), "ALA", "1", "A", "A:ALA:1", "protein"),
        AtomRecord(2, "N", "N", (1.15, -.665, 0.), "GLY", "2", "A", "A:GLY:2", "protein"),
    ]
    center = (.575, -.3325, 0.)
    ring_center = (*center[:2], 3.6)
    for index in range(6):
        coord = (center[0] + 1.4 * cos(index * pi / 3),
                 center[1] + 1.4 * sin(index * pi / 3), 3.6)
        atoms.append(AtomRecord(3 + index, "C%d" % (index + 1), "C", coord,
                                "LIG", "9", "B", "B:LIG:9", "ligand"))
    features = [
        FeatureRecord(0, "amide", "A:ALA:1", (0, 1, 2), center,
                      normal=(0., 0., 1.), metadata={
                          "carbon_atom_id": 0, "oxygen_atom_id": 1, "nitrogen_atom_id": 2,
                          "component_ids": ["A:ALA:1", "A:GLY:2"]}),
        FeatureRecord(1, "ring", "B:LIG:9", tuple(range(3, 9)), ring_center,
                      normal=(0., 0., 1.), metadata={"aromatic": True}),
    ]
    return _detect("amide_pi", atoms, features)


def _rows(data, **kwargs):
    hit, records, mapped, features = data
    return runner._resolve_rows([hit], records, mapped, features, **kwargs)


def test_salt_bridge_draws_the_actual_measured_charged_atoms(salt):
    hit, _, mapped, _ = salt
    row, = _rows(salt)
    assert hit.metadata["distance_atom_ids"] == [1, 4]
    assert row.endpoint1.atom is mapped[1]
    assert row.endpoint2.atom is mapped[4]
    assert runner._sq_dist(row.endpoint1.atom.coord, row.endpoint2.atom.coord) ** .5 == pytest.approx(hit.distance)
    assert hit.distance == pytest.approx(3.)


@pytest.mark.parametrize("missing", [1, 4])
def test_salt_bridge_never_substitutes_an_unmapped_measured_atom(salt, missing):
    del salt[2][missing]
    assert _rows(salt) == []


def test_salt_bridge_without_measured_pair_is_not_fabricated(salt):
    salt[0].metadata.clear()
    assert _rows(salt) == []


@pytest.mark.parametrize("reverse_features", [False, True])
def test_backbone_amide_is_one_partner_and_ligand_ring_is_the_other(amide, reverse_features):
    hit, records, mapped, features = amide
    if reverse_features:
        hit = replace(hit, feature_ids=tuple(reversed(hit.feature_ids)))
    row, = _rows((hit, records, mapped, features), skip_biopolymer_internal=True)
    assert row.endpoint1.atom is mapped[0]
    assert row.endpoint1.center_atoms == [mapped[0], mapped[2]]
    assert row.endpoint1.ring_atoms is None
    assert row.endpoint2.center_atoms is None
    assert row.endpoint2.ring_atoms == [mapped[aid] for aid in range(3, 9)]
    assert row.endpoint2.residue_name == "LIG"
    assert "midpoint" in row.endpoint1.atom_label
    assert "ALA 1 C" in row.endpoint1.atom_label and "GLY 2 N" in row.endpoint1.atom_label
    assert hit.distance == pytest.approx(3.6)


def test_selection_of_the_second_backbone_residue_retains_amide_contact(amide):
    amide[2][2].selected = True
    assert len(_rows(amide, selected_only=True, skip_biopolymer_internal=True)) == 1


@pytest.mark.parametrize("missing", [0, 2, 3])
def test_amide_contact_requires_complete_midpoint_and_ring_geometry(amide, missing):
    del amide[2][missing]
    assert _rows(amide) == []


def test_amide_contact_does_not_fall_back_to_residue_pairing_without_features(amide):
    amide[3].clear()
    assert _rows(amide) == []


def test_trajectory_scope_groups_backbone_amide_as_a_single_partner(amide):
    hit, records, mapped, features = amide
    key = (hit.interaction_type, hit.atom_ids)
    result = NS(frames=[NS(frame_index=0, interactions=[hit])],
                interaction_key=lambda inter: key, atoms=list(records.values()), components=[])
    run = runner.TrajectoryRun(result, {}, {}, NS(), [1], records, features, mapped)
    assert run.frame_endpoints(0, key) == [
        [mapped[0], mapped[1], mapped[2]], [mapped[aid] for aid in range(3, 9)]]


class _Position:
    """90-degree rotation around Z followed by translation."""
    def __init__(self, inverse=False):
        self._inverse = inverse

    def inverse(self):
        return _Position(not self._inverse)

    def __mul__(self, point):
        x, y, z = point
        return (y + 3., 10. - x, z - 2.) if self._inverse else (-y + 10., x - 3., z + 2.)


class _Model:
    scene_position = _Position()

    def __init__(self):
        self.children = []

    def add(self, children):
        self.children.extend(children)


class _Group:
    id = 1

    def __init__(self):
        self.bonds = []

    @property
    def num_pseudobonds(self):
        return len(self.bonds)

    def new_pseudobond(self, first, second):
        bond = NS(atoms=(first, second))
        self.bonds.append(bond)
        return bond


@pytest.mark.parametrize("pose", [False, True])
def test_midpoint_markers_use_transformed_model_frame_for_structure_and_pose(amide, monkeypatch, pose):
    class MarkerSet:
        def __init__(self, session, name):
            self.name = name

        def create_marker(self, coord, color, radius):
            return NS(coord=coord)

    monkeypatch.setitem(sys.modules, "chimerax.markers", NS(MarkerSet=MarkerSet))
    monkeypatch.setitem(sys.modules, "chimerax.atomic", NS(Element=NS(get_element=lambda name: name)))
    model = _Model()
    for atom in amide[2].values():
        atom.structure = model
        atom.coord = model.scene_position.inverse() * atom.scene_coord
    row, = _rows(amide)
    ordinary, global_group = _Group(), _Group()
    session = NS(pb_manager=NS(get_group=lambda *args, **kwargs: global_group))
    if pose:
        assert runner.draw_pose_rows(session, [row], "pose") is global_group
        bond, = global_group.bonds
    else:
        bond = runner._draw_interaction(session, model, row.endpoint1, row.endpoint2,
                                       {"color": (255, 255, 255, 255), "radius": .1},
                                       ordinary, global_group, {})
        assert global_group.bonds == [bond] and ordinary.bonds == []
    first, second = bond.atoms
    assert first is not row.endpoint1.atom and second is not row.endpoint2.atom
    center = model.scene_position * first.coord
    assert center == pytest.approx(amide[0].metadata["amide_center"])
    assert runner._sq_dist(center, model.scene_position * second.coord) ** .5 == pytest.approx(row.distance)
    assert first.element == second.element == "He"


def test_ring_only_endpoint_keeps_existing_ring_semantics(amide):
    row, = _rows(amide)
    endpoint = row.endpoint2
    assert endpoint.atom in endpoint.ring_atoms
    assert endpoint.center_atoms is None

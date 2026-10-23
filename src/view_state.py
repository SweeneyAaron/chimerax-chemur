"""Qt-independent visibility and membership for native enclosure groups."""
from __future__ import annotations


class ContactView:
    def __init__(self, rows):
        self.rows = list(rows)
        self.groups = {}
        self.memberships = {}
        self.visible = [True] * len(self.rows)
        for index, row in enumerate(self.rows):
            if row.interaction_type != "hydrophobic_enclosure":
                continue
            for metadata in row.extra.get("enclosure_groups", ()):
                center = int(metadata["center_atom_id"])
                group = self.groups.setdefault(center, {"metadata": metadata, "indices": []})
                if index not in group["indices"]:
                    group["indices"].append(index)
                self.memberships[index, center] = True

    def row_visible(self, index):
        values = [on for (idx, _), on in self.memberships.items() if idx == index]
        return any(values) if values else self.visible[index]

    def member_visible(self, index, center=None):
        return self.memberships.get((index, center), self.row_visible(index))

    def set_visible(self, indices, visible, center=None):
        for index in indices:
            self.visible[index] = bool(visible)
            for key in self.memberships:
                if key[0] == index and (center is None or key[1] == center):
                    self.memberships[key] = bool(visible)

    def set_group_visible(self, center, visible):
        self.set_visible(self.groups[center]["indices"], visible, center)

    @property
    def native_count(self):
        return len(self.groups) + sum(row.interaction_type != "hydrophobic_enclosure" for row in self.rows)


def passes_selection(groups, scope):
    """Two enclosure endpoints are the centre and the entire current shell."""
    if scope == "all":
        return True
    selected = [any(not getattr(atom, "deleted", False) and getattr(atom, "selected", False)
                    for atom in group) for group in groups]
    return any(selected) if scope == "any" else len(selected) >= 2 and all(selected)


def is_polymer_intrachain(row):
    """Author chain labels do not identify a polymer across source models."""
    first, second = row.endpoint1, row.endpoint2
    polymer_types = {"protein", "nucleotide"}
    if first.molecule_type not in polymer_types or second.molecule_type not in polymer_types:
        return False
    atoms = first.atom, second.atom
    structures = [getattr(atom, "structure", None) for atom in atoms]
    if structures[0] is None or structures[0] is not structures[1]:
        return False
    chains = [getattr(atom.residue, "chain", None) for atom in atoms]
    if all(chain is not None for chain in chains):
        return chains[0] is chains[1]
    # Missing live chain objects cannot establish polymer-chain identity.
    return False


class DiagramPresentation:
    """Presentation only: native interactions and chemistry are never filtered."""

    def __init__(self, *, style="default", font_size=10.0):
        self.style = style
        self.font_size = float(font_size)
        self.units = {}
        self.types = {}
        self.edges = {}
        self.colors = {}
        self.legend_visible = True

    def unit_visible(self, unit):
        return self.units.get(id(unit), True)

    def edge_visible(self, edge):
        return (all(self.unit_visible(unit) for unit in getattr(edge, "units", (edge.unitA, edge.unitB)))
                and self.types.get(edge.interaction_type, True)
                and self.edges.get(id(edge), True))


def zoom_scale(current, angle_delta=0.0, pixel_delta=0.0):
    """Wheel notch = 120 angle units or 40 trackpad pixels; return relative zoom."""
    delta = pixel_delta / 40.0 if pixel_delta else angle_delta / 120.0
    if not delta:
        return current
    # Limit exponents before evaluation as synthetic devices can emit huge deltas.
    return max(0.25, min(8.0, current * 1.08 ** max(-100.0, min(100.0, delta))))

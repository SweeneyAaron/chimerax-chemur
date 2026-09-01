"""Pseudobond styling per Chemur interaction type.

Colors are RGBA tuples (0-255); radius is in Angstroms. The keys are the
``interaction_type`` strings emitted by Chemur (the rule names in
``chemur/profiles/default.yaml``).
"""

from __future__ import annotations

_DEFAULT = {"color": (180, 180, 180, 255), "radius": 0.1, "dashes": 6}

INTERACTION_STYLE = {
    "hbond":              {"color": (70, 130, 255, 255), "radius": 0.12, "dashes": 6},
    "weak_hbond":         {"color": (140, 180, 255, 255), "radius": 0.10, "dashes": 8},
    "salt_bridge":        {"color": (255, 90, 90, 255),  "radius": 0.16, "dashes": 4},
    "metal_coordination": {"color": (200, 140, 255, 255), "radius": 0.16, "dashes": 4},
    "amide_bridge":       {"color": (90, 200, 200, 255),  "radius": 0.12, "dashes": 6},
    "solvent_bridge":     {"color": (120, 200, 255, 255), "radius": 0.10, "dashes": 8},
    "pipi_stack":         {"color": (90, 200, 90, 255),   "radius": 0.14, "dashes": 6},
    "cation_pi":          {"color": (255, 170, 60, 255),  "radius": 0.14, "dashes": 6},
    "anion_pi":           {"color": (60, 170, 255, 255),  "radius": 0.14, "dashes": 6},
    "hbond_pi":           {"color": (150, 220, 150, 255), "radius": 0.12, "dashes": 8},
    "amide_pi":           {"color": (120, 220, 200, 255), "radius": 0.12, "dashes": 8},
    "halogen_bond":       {"color": (0, 200, 160, 255),   "radius": 0.14, "dashes": 4},
    "halogen_pi":         {"color": (120, 230, 200, 255), "radius": 0.12, "dashes": 8},
    "ch_pi":              {"color": (200, 200, 120, 255), "radius": 0.10, "dashes": 8},
    "hydrophobic":        {"color": (210, 210, 210, 255), "radius": 0.08, "dashes": 10},
    # Types the chemur engine added after the initial release. Styled to the same
    # conventions as the rest of the table: sigma-hole donors get radius 0.14 /
    # 4 dashes like halogen_bond, their pi variants a lighter tint at 0.12 / 8,
    # and weak dispersive contacts a small radius with many dashes.
    "chalcogen_bond":     {"color": (230, 100, 200, 255), "radius": 0.14, "dashes": 4},
    "chalcogen_pi":       {"color": (240, 170, 225, 255), "radius": 0.12, "dashes": 8},
    "tetrel_bond":        {"color": (190, 150, 90, 255),  "radius": 0.14, "dashes": 4},
    "n_pi_star":          {"color": (120, 110, 210, 255), "radius": 0.10, "dashes": 8},
    "anion_aromatic_edge": {"color": (60, 210, 255, 255), "radius": 0.12, "dashes": 8},
    "aliphatic_pi_stack": {"color": (160, 205, 110, 255), "radius": 0.12, "dashes": 8},
    # Disabled in the engine's default profile, but styled so enabling it in a
    # custom profile does not fall through to grey.
    "aliphatic_stack":    {"color": (190, 215, 160, 255), "radius": 0.10, "dashes": 10},
}


def style_for(interaction_type: str) -> dict:
    return INTERACTION_STYLE.get(interaction_type, _DEFAULT)


def rgba_255(interaction_type: str) -> tuple:
    """RGBA 0-255 tuple for an interaction type (ChimeraX pseudobond color)."""
    return style_for(interaction_type)["color"]


def mpl_color(interaction_type: str) -> tuple:
    """RGBA 0-1 tuple for matplotlib."""
    r, g, b, a = style_for(interaction_type)["color"]
    return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

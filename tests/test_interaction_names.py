"""Fail if the bundle's pseudobond style table drifts from the engine's rule set.

``colors.INTERACTION_STYLE`` hardcodes the interaction-type strings the engine
emits. When they drift, ``style_for()`` quietly returns ``_DEFAULT`` and the new
interaction type draws as an unlabelled grey dashed line -- no exception, no log
message, just a wrong-looking figure.

This reads the *installed* engine's default profile, so it also catches an
engine release that adds or renames a type.
"""

import importlib.util
from pathlib import Path

import pytest

profile = pytest.importorskip(
    "chemur.profile",
    reason="engine not installed; `pip install chemur` to enable this guard",
)

_SRC = Path(__file__).resolve().parents[1] / "src"


def _colors():
    spec = importlib.util.spec_from_file_location(
        "chemur_chimerax_colors", _SRC / "colors.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_style_table_matches_engine_rules():
    engine = set(profile.load_profile("default")["rules"])
    styled = set(_colors().INTERACTION_STYLE)

    assert not (engine - styled), (
        "engine interaction types with no pseudobond style "
        f"(they would draw grey and unlabelled): {sorted(engine - styled)}"
    )
    assert not (styled - engine), (
        "styles for interaction types the engine no longer emits: "
        f"{sorted(styled - engine)}"
    )

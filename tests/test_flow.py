"""Tests for the flow-layout wrap arithmetic (src/flow.py).

The module deliberately has no Qt imports so it can be loaded straight from its
file path, like the other bundle modules under test.
"""

import importlib.util
from pathlib import Path

_FLOW_PATH = (Path(__file__).resolve().parent.parent
              / "src" / "flow.py")
_spec = importlib.util.spec_from_file_location("chemur_chimerax_flow", _FLOW_PATH)
flow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flow)


def _sizes(count, w=40, h=20):
    return [(w, h)] * count


def test_empty_input():
    rects, height = flow.flow_geometry([], 300)
    assert rects == []
    assert height == 0


def test_empty_input_still_spans_margins():
    _rects, height = flow.flow_geometry([], 300, margins=(3, 5, 3, 7))
    assert height == 12


def test_single_row_when_wide():
    rects, height = flow.flow_geometry(_sizes(3), 800, hspacing=6, vspacing=4)
    assert [r[1] for r in rects] == [0, 0, 0]           # all on one line
    assert [r[0] for r in rects] == [0, 46, 92]         # 40 wide + 6 spacing
    assert height == 20


def test_wraps_when_narrow():
    rects, height = flow.flow_geometry(_sizes(3), 100, hspacing=6, vspacing=4)
    assert [(r[0], r[1]) for r in rects] == [(0, 0), (46, 0), (0, 24)]
    assert height == 44                                 # 20 + 4 + 20


def test_wrap_boundary_exact_fit():
    # Two 40-wide items with 6 px spacing need exactly 86 px.
    rects, height = flow.flow_geometry(_sizes(2), 86, hspacing=6)
    assert [r[1] for r in rects] == [0, 0]
    assert height == 20


def test_wrap_boundary_one_pixel_short():
    rects, height = flow.flow_geometry(_sizes(2), 85, hspacing=6, vspacing=4)
    assert [r[1] for r in rects] == [0, 24]
    assert height == 44


def test_lone_overwide_item_does_not_wrap():
    # A single item wider than the row must overflow, not wrap onto an equally
    # narrow next line (which would leave an empty first line).
    rects, height = flow.flow_geometry([(500, 20)], 100)
    assert rects == [(0, 0, 500, 20)]
    assert height == 20


def test_overwide_item_wraps_off_a_populated_line():
    rects, _height = flow.flow_geometry([(40, 20), (500, 30)], 100, vspacing=4)
    assert [(r[0], r[1]) for r in rects] == [(0, 0), (0, 24)]


def test_line_height_follows_tallest_item_on_the_line():
    # 40x20 then 40x50 share line 0; the third wraps below the taller of the two.
    rects, height = flow.flow_geometry(
        [(40, 20), (40, 50), (40, 20)], 100, hspacing=6, vspacing=4)
    assert [(r[0], r[1]) for r in rects] == [(0, 0), (46, 0), (0, 54)]
    assert height == 74                                 # 50 + 4 + 20


def test_margins_offset_origin_and_shrink_usable_width():
    # 10 px left/right margins leave 80 of 100 px, so the second item wraps.
    rects, height = flow.flow_geometry(
        _sizes(2), 100, hspacing=6, vspacing=4, margins=(10, 5, 10, 5))
    assert [(r[0], r[1]) for r in rects] == [(10, 5), (10, 29)]
    assert height == 54                                 # 5 + 20 + 4 + 20 + 5


def test_narrower_width_is_never_shorter():
    sizes = [(40, 20), (70, 20), (55, 20), (90, 20)]
    heights = [flow.flow_height(sizes, w) for w in (600, 400, 300, 200, 120)]
    assert heights == sorted(heights)
    assert heights[0] == 20                             # one row when wide


def test_flow_height_matches_flow_geometry():
    sizes = [(40, 20), (70, 30), (55, 20)]
    assert flow.flow_height(sizes, 130) == flow.flow_geometry(sizes, 130)[1]


def test_items_never_resized():
    sizes = [(40, 20), (70, 30), (55, 25)]
    rects, _height = flow.flow_geometry(sizes, 90)
    assert [(r[2], r[3]) for r in rects] == sizes

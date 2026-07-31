"""Wrap arithmetic for the flow layout in widgets.py.

Kept free of Qt imports so it can be unit-tested headlessly (the bundle's Qt
modules only import inside ChimeraX). widgets.FlowLayout is a thin Qt shell
around flow_geometry().
"""

from __future__ import annotations


def flow_geometry(sizes, width, hspacing=6, vspacing=6, margins=(0, 0, 0, 0)):
    """Place ``sizes`` left-to-right, wrapping to a new line when out of width.

    ``sizes`` is a sequence of (w, h) item sizes; ``margins`` is
    (left, top, right, bottom). Items are placed at their requested size and
    never squashed, so a single item wider than ``width`` overflows its line
    rather than wrapping onto a line it still would not fit on.

    Returns (rects, total_height), where rects is a list of (x, y, w, h) in the
    same order as ``sizes`` and total_height spans both vertical margins.
    """
    left, top, right, bottom = margins
    # An item at x fits when its last pixel, x + w - 1, is still inside the
    # content box, whose last usable pixel is width - right - 1. Hence the test
    # below is `x + w > limit`, NOT the `nextX - spacing > right` form carried by
    # most ports of Qt's flowlayout example -- that one is off by one spacing and
    # wraps a row early.
    limit = width - right

    rects = []
    x, y, line_height = left, top, 0
    for w, h in sizes:
        # Only wrap once something is already on this line: a lone over-wide
        # item must overflow rather than wrap onto an equally narrow next line.
        if line_height > 0 and x + w > limit:
            x = left
            y += line_height + vspacing
            line_height = 0
        rects.append((x, y, w, h))
        x += w + hspacing
        line_height = max(line_height, h)

    return rects, y + line_height + bottom


def flow_height(sizes, width, hspacing=6, vspacing=6, margins=(0, 0, 0, 0)):
    """Height flow_geometry() would need for ``sizes`` at ``width``."""
    return flow_geometry(sizes, width, hspacing, vspacing, margins)[1]

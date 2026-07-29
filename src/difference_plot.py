"""Diff fingerprint heatmap comparing interactions between two models.

Rows are interaction signatures (residue-pair + type) restricted to residues
common to both models; the two columns are model A and model B. Each cell is
colour-coded by category:

* **shared**     — present in both models (green, both columns)
* **only A / lost**  — present only in model A (red, A column)
* **only B / gained** — present only in model B (blue, B column)

``draw_difference`` is backend-agnostic (operates on a ``matplotlib.figure``)
so it backs both the embedded Qt canvas and PNG/SVG export, mirroring
``trajectory_plots.draw_fingerprint``. ``DifferencePlotPanel`` wires it into a
widget with a "changed only" toggle and row-click selection;
``DifferenceDialog`` is the resizable pop-out window.
"""

from __future__ import annotations

from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

# Cell category -> integer code used in the heatmap grid.
_ABSENT, _SHARED, _ONLY_A, _ONLY_B = 0, 1, 2, 3
# Colours indexed by the codes above (white for absent).
_CELL_COLORS = ["white", "#2ca02c", "#d62728", "#1f77b4"]

# Listing every shared contact for a large complex makes an unreadably tall
# heatmap; cap the rows shown and report the cap in the title.
_MAX_ROWS = 60


def _visible_entries(entries, changed_only):
    rows = [e for e in entries if not (changed_only and e["category"] == "shared")]
    return rows[:_MAX_ROWS], len(rows)


def draw_difference(fig, result, *, changed_only=False):
    """Draw the diff heatmap onto ``fig``. Returns the entries actually plotted
    (top to bottom) so a caller can map a clicked row back to an interaction."""
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig.clear()
    ax = fig.add_subplot(111)

    entries, total = _visible_entries(result.entries, changed_only)
    if not entries:
        ax.text(0.5, 0.5, "No comparable interactions", ha="center", va="center")
        ax.set_axis_off()
        return []

    grid = np.zeros((len(entries), 2), dtype=float)
    for i, e in enumerate(entries):
        cat = e["category"]
        if cat == "shared":
            grid[i, 0] = grid[i, 1] = _SHARED
        elif cat == "only_a":
            grid[i, 0] = _ONLY_A
        else:  # only_b
            grid[i, 1] = _ONLY_B

    cmap = ListedColormap(_CELL_COLORS)
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3,
              interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([result.name_a, result.name_b], fontsize=7)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e["label"] for e in entries], fontsize=5)
    ax.set_xticks([0.5], minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.4)
    title = "Interaction difference"
    if len(entries) < total:
        title += " (top %d of %d rows)" % (len(entries), total)
    ax.set_title(title, fontsize=9)

    ax.legend(handles=[
        Patch(facecolor=_CELL_COLORS[_SHARED], label="shared"),
        Patch(facecolor=_CELL_COLORS[_ONLY_A], label="only %s" % result.name_a),
        Patch(facecolor=_CELL_COLORS[_ONLY_B], label="only %s" % result.name_b),
    ], fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.8)
    # Reserve the right ~22% for the external legend so it isn't clipped on
    # screen (export uses bbox_inches="tight" and is unaffected).
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    return entries


class DifferencePlotPanel(QWidget):
    """Embedded matplotlib panel for the diff heatmap.

    ``on_row_selected(entry)`` is invoked when the user clicks a heatmap row.
    """

    def __init__(self, parent=None, on_row_selected=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvas
        from matplotlib.figure import Figure

        self._on_row_selected = on_row_selected
        self._result = None
        self._plotted = []      # entries currently drawn (row order)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        bar = QHBoxLayout()
        self._summary = QLabel("")
        bar.addWidget(self._summary, 1)
        self._changed_cb = QCheckBox("Changed only")
        self._changed_cb.setToolTip("Hide shared interactions, leaving only gains/losses.")
        self._changed_cb.toggled.connect(lambda _c: self._redraw())
        bar.addWidget(self._changed_cb)
        layout.addLayout(bar)

        self._fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self._fig)
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def set_result(self, result):
        self._result = result
        self._summary.setText(
            "%d common residues · %d shared · %d changed"
            % (result.n_common,
               sum(1 for e in result.entries if e["category"] == "shared"),
               result.n_changed))
        self._redraw()

    def _redraw(self):
        if self._result is None:
            return
        self._plotted = draw_difference(
            self._fig, self._result, changed_only=self._changed_cb.isChecked())
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None or not self._plotted or self._on_row_selected is None:
            return
        row = int(round(event.ydata))
        if 0 <= row < len(self._plotted):
            self._on_row_selected(self._plotted[row])

    def save_current(self, path):
        self._fig.savefig(path, dpi=200, bbox_inches="tight")


class DifferenceDialog(QDialog):
    """Resizable pop-out window hosting the diff heatmap (non-modal)."""

    def __init__(self, parent, session, on_row_selected=None):
        super().__init__(parent)
        self.setWindowTitle("ChemeleonX model difference")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._session = session

        self.panel = DifferencePlotPanel(on_row_selected=on_row_selected)

        outer = QVBoxLayout(self)
        outer.addWidget(self.panel, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        save_btn = QPushButton("Save plot…")
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        buttons.addWidget(close_btn)
        outer.addLayout(buttons)

        self.resize(760, 680)

    def set_result(self, result):
        self.panel.set_result(result)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", "chemeleonx_difference.png",
            "Images (*.png *.svg *.pdf)")
        if path:
            self.panel.save_current(path)
            self._session.logger.info("ChemeleonX: saved difference plot to %s" % path)

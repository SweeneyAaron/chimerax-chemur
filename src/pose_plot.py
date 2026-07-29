"""MSA-like comparison of docked poses' protein-ligand interactions.

Columns are the protein residues the selected poses interact with, shown as a
contiguous sequence window (interacting residues plus optional greyed context
residues, labelled like a sequence); rows are the poses. Each cell shows which
interaction type(s) that pose makes with that residue, as up to K small colour
swatches; an optional consensus row summarises interactions conserved across the
poses. A legend maps colour -> type.

``draw_pose_msa`` is backend-agnostic (operates on a ``matplotlib.figure.Figure``)
so it backs both the embedded Qt canvas and PNG/SVG/PDF export. ``PoseMsaPanel``
wires it into a widget with a per-type include filter + colour pickers, a
consensus toggle/threshold, a 1-letter-label toggle, a context spinner, per-pose
"show in 3D" checkboxes and cell-click selection. ``PoseMsaDialog`` is the
resizable pop-out window.
"""

from __future__ import annotations

from Qt.QtCore import Qt
from Qt.QtGui import QColor, QIcon, QPixmap
from Qt.QtWidgets import (
    QCheckBox, QColorDialog, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .colors import mpl_color
from . import pose_compare

# Max interaction-type swatches drawn in one cell before collapsing to "+".
_MAX_SWATCHES = 4

# Publication-style accents.
_GRID = "#E6E9ED"
_SEGMENT = "#B7BEC8"
_BORDER = "#D9DEE4"
_CONTEXT = (0, 0, 0, 0.045)


def _segment_break_before(cols, c):
    """True if column ``c`` starts a new sequence segment (chain change or gap)."""
    prev = cols[c - 1]
    cur = cols[c]
    return cur.chain_id != prev.chain_id or cur.number != prev.number + 1


def _draw_cell(ax, Rectangle, cx, ry, types, cf, alpha_by_type=None):
    """Draw a cell's interaction-type swatches at column ``cx``, row ``ry``."""
    shown = sorted(types)
    overflow = len(shown) > _MAX_SWATCHES
    if overflow:
        shown = shown[:_MAX_SWATCHES - 1]
    slots = len(shown) + (1 if overflow else 0)
    slot_w = 0.82 / slots
    for i, t in enumerate(shown):
        r, g, b, a = cf(t)
        if alpha_by_type is not None:
            a = 0.4 + 0.6 * alpha_by_type.get(t, 1.0)
        x0 = cx - 0.41 + i * slot_w
        ax.add_patch(Rectangle((x0, ry - 0.41), slot_w * 0.9, 0.82,
                               facecolor=(r, g, b, a), edgecolor="white",
                               lw=0.5, zorder=2))
    if overflow:
        ax.text(cx + 0.41 - slot_w * 0.5, ry, "+", ha="center", va="center",
                fontsize=6, zorder=3)


def draw_pose_msa(fig, comparison, *, include_types=None, one_letter=True,
                  max_cols=None, color_for=None, show_consensus=False,
                  consensus_threshold=0.5):
    """Render the residue x pose interaction grid onto ``fig`` (publication style).

    ``include_types`` restricts which types are drawn/legended; ``one_letter``
    chooses 1- vs 3-letter column labels; ``color_for(type)`` overrides the swatch
    colour (default :func:`colors.mpl_color`); ``show_consensus`` appends a
    consensus row of interactions made by >= ``consensus_threshold`` of the poses.
    Returns a layout dict ``{"cols": [...], "rows": [...]}`` (rows = poses only) for
    click hit-testing.
    """
    import matplotlib
    import numpy as np
    from matplotlib.patches import Patch, Rectangle

    cf = color_for or mpl_color

    with matplotlib.rc_context({
        "font.size": 8,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        fig.clear()
        fig.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        cols = list(comparison.columns)
        rows = comparison.rows
        if max_cols and len(cols) > max_cols:
            interacting = [c for c in cols if not c.is_context]
            context = [c for c in cols if c.is_context]
            keep = set(id(c) for c in
                       interacting + context[:max(0, max_cols - len(interacting))])
            cols = [c for c in cols if id(c) in keep]

        n_rows = len(rows)
        n_cols = len(cols)
        if not n_rows or not n_cols:
            ax.text(0.5, 0.5, "No pose interactions", ha="center", va="center",
                    color="#666")
            ax.set_axis_off()
            return {"cols": [], "rows": []}

        consensus = (pose_compare.compute_consensus(
            comparison, include_types=include_types, threshold=consensus_threshold)
            if show_consensus else {})
        total_rows = n_rows + (1 if show_consensus else 0)

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, total_rows - 0.5)
        ax.set_aspect("equal")   # square cells; export crops with bbox_inches="tight"

        # Faint shading behind context (non-interacting) residues.
        for c, col in enumerate(cols):
            if col.is_context:
                ax.add_patch(Rectangle((c - 0.5, -0.5), 1, total_rows,
                                       facecolor=_CONTEXT, edgecolor="none", zorder=0))
        # Separators between non-contiguous sequence segments.
        for c in range(1, n_cols):
            if _segment_break_before(cols, c):
                ax.axvline(c - 0.5, color=_SEGMENT, lw=1.0, alpha=0.9, zorder=1)

        # Pose cells.
        for r in range(n_rows):
            for c, col in enumerate(cols):
                types = comparison.cell.get((r, col.key))
                if not types:
                    continue
                shown = [t for t in types
                         if include_types is None or t in include_types]
                if shown:
                    _draw_cell(ax, Rectangle, c, r, shown, cf)

        # Consensus row (drawn beneath the poses, alpha = fraction of poses).
        if show_consensus:
            ax.axhline(n_rows - 0.5, color=_SEGMENT, lw=1.2, zorder=1)
            for c, col in enumerate(cols):
                frac = consensus.get(col.key)
                if frac:
                    _draw_cell(ax, Rectangle, c, n_rows, set(frac), cf,
                               alpha_by_type=frac)

        labels = [c.label1 if one_letter else c.label3 for c in cols]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.xaxis.set_ticks_position("top")

        ylabels = [row.name for row in rows]
        if show_consensus:
            ylabels.append("Consensus")
        ax.set_yticks(range(total_rows))
        ytl = ax.set_yticklabels(ylabels, fontsize=7)
        if show_consensus:
            ytl[-1].set_fontweight("bold")
        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(total_rows + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=_GRID, lw=0.6)
        ax.tick_params(which="both", length=0)

        legend_types = [t for t in comparison.types
                        if include_types is None or t in include_types]
        if legend_types:
            ax.legend(
                handles=[Patch(facecolor=cf(t), edgecolor="none", label=t)
                         for t in legend_types],
                fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                frameon=True, framealpha=1.0, edgecolor=_BORDER, facecolor="white",
                title="Interaction", title_fontsize=8)

        n_interacting = len({c.key for c in cols if not c.is_context})
        # ax.set_title (unlike fig.suptitle) is honoured by tight_layout and is
        # centred over the grid rather than the whole figure; pad clears the
        # on-top residue labels.
        ax.set_title("Docked-pose interaction comparison",
                     fontsize=12, fontweight="bold", pad=26)
        fig.text(0.41, 0.012,
                 "%d poses · %d interacting residues · receptor %s"
                 % (n_rows, n_interacting, comparison.receptor_name),
                 ha="center", fontsize=7.5, color="#555")
        fig.tight_layout(rect=(0, 0.03, 0.82, 1))
    return {"cols": cols, "rows": rows}


def _type_swatch(rgba):
    r, g, b, a = rgba
    pix = QPixmap(12, 12)
    pix.fill(QColor.fromRgbF(r, g, b, a))
    return QIcon(pix)


class PoseMsaPanel(QWidget):
    """Embedded matplotlib panel for the pose-comparison grid.

    ``on_cell_selected(row_index, col_key)`` fires on a cell click;
    ``on_pose_show_3d(pose_index, show)`` fires when a pose's "show in 3D" checkbox
    is toggled.
    """

    def __init__(self, parent=None, on_cell_selected=None, on_pose_show_3d=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvas
        from matplotlib.figure import Figure

        self._on_cell_selected = on_cell_selected
        self._on_pose_show_3d = on_pose_show_3d
        self._comparison = None
        self._layout = {"cols": [], "rows": []}
        self._type_included = {}        # interaction_type -> bool
        self._type_colors = {}          # interaction_type -> matplotlib RGBA
        self._type_checkboxes = {}
        self._pose_checkboxes = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        bar = QHBoxLayout()
        self._summary = QLabel("")
        bar.addWidget(self._summary, 1)
        bar.addWidget(QLabel("Context ±"))
        self._context_spin = QSpinBox()
        self._context_spin.setRange(0, 3)
        self._context_spin.setToolTip(
            "Show this many flanking residues around each interacting residue.")
        self._context_spin.valueChanged.connect(self._on_context_changed)
        bar.addWidget(self._context_spin)
        self._one_letter_cb = QCheckBox("1-letter")
        self._one_letter_cb.setChecked(True)
        self._one_letter_cb.setToolTip("1-letter vs 3-letter residue labels.")
        self._one_letter_cb.toggled.connect(lambda _c: self._redraw())
        bar.addWidget(self._one_letter_cb)
        self._consensus_cb = QCheckBox("Consensus")
        self._consensus_cb.setToolTip(
            "Add a row of interactions made by at least the threshold fraction of poses.")
        self._consensus_cb.toggled.connect(lambda _c: self._redraw())
        bar.addWidget(self._consensus_cb)
        bar.addWidget(QLabel("≥"))
        self._consensus_spin = QSpinBox()
        self._consensus_spin.setRange(1, 100)
        self._consensus_spin.setValue(50)
        self._consensus_spin.setSuffix("%")
        self._consensus_spin.setToolTip("Consensus threshold (fraction of poses).")
        self._consensus_spin.valueChanged.connect(
            lambda _v: self._redraw() if self._consensus_cb.isChecked() else None)
        bar.addWidget(self._consensus_spin)
        layout.addLayout(bar)

        side = QHBoxLayout()
        # Per-type include filter + colour pickers.
        self._types_area = QScrollArea()
        self._types_area.setWidgetResizable(True)
        self._types_area.setFixedHeight(96)
        self._types_holder = QWidget()
        self._types_layout = QVBoxLayout(self._types_holder)
        self._types_layout.setContentsMargins(2, 2, 2, 2)
        self._types_area.setWidget(self._types_holder)
        side.addWidget(self._wrap_titled("Interactions", self._types_area), 1)

        # Per-pose "show in 3D" checkboxes.
        self._poses_area = QScrollArea()
        self._poses_area.setWidgetResizable(True)
        self._poses_area.setFixedHeight(96)
        self._poses_holder = QWidget()
        self._poses_layout = QVBoxLayout(self._poses_holder)
        self._poses_layout.setContentsMargins(2, 2, 2, 2)
        self._poses_area.setWidget(self._poses_holder)
        side.addWidget(self._wrap_titled("Show in 3D", self._poses_area), 1)
        layout.addLayout(side)

        self._fig = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self._fig)
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    @staticmethod
    def _wrap_titled(title, widget):
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold;")
        v.addWidget(lbl)
        v.addWidget(widget)
        return box

    # ----- data ------------------------------------------------------------
    def set_comparison(self, comparison):
        self._comparison = comparison
        self._type_included = {t: True for t in comparison.types}
        self._type_colors = {}
        self._update_summary()
        self._context_spin.blockSignals(True)
        self._context_spin.setValue(comparison.context)
        self._context_spin.blockSignals(False)
        self._build_type_filter()
        self._build_pose_list()
        self._redraw()

    def _update_summary(self):
        c = self._comparison
        self._summary.setText(
            "%d poses · %d residues · %d interaction type(s)"
            % (len(c.rows), len(c.columns), len(c.types)))

    # ----- colour helpers (mirrors trajectory_plots) -----------------------
    def _color_for(self, itype):
        return self._type_colors.get(itype) or mpl_color(itype)

    def _qcolor_for(self, itype):
        r, g, b, a = self._color_for(itype)
        return QColor.fromRgbF(r, g, b, a)

    def _pick_color(self, itype, checkbox):
        color = QColorDialog.getColor(self._qcolor_for(itype), self,
                                      "Colour for %s" % itype)
        if color.isValid():
            self._type_colors[itype] = (color.redF(), color.greenF(),
                                        color.blueF(), 1.0)
            checkbox.setIcon(_type_swatch(self._color_for(itype)))
            self._redraw()

    # ----- list builders ---------------------------------------------------
    def _build_type_filter(self):
        _clear_layout(self._types_layout)
        self._type_checkboxes = {}
        for t in self._comparison.types:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(t)
            cb.setChecked(self._type_included.get(t, True))
            cb.setIcon(_type_swatch(self._color_for(t)))
            cb.toggled.connect(lambda checked, it=t: self._on_type_toggled(it, checked))
            rl.addWidget(cb, 1)
            btn = QPushButton("Colour")
            btn.setFixedWidth(64)
            btn.clicked.connect(lambda _c=False, it=t, c=cb: self._pick_color(it, c))
            rl.addWidget(btn, 0)
            self._type_checkboxes[t] = cb
            self._types_layout.addWidget(row)
        self._types_layout.addStretch(1)

    def _build_pose_list(self):
        _clear_layout(self._poses_layout)
        self._pose_checkboxes = []
        for idx, prow in enumerate(self._comparison.rows):
            cb = QCheckBox(prow.name)
            if prow.fully_mapped:
                cb.setToolTip("Draw this pose's interactions as pseudobonds in 3D.")
                cb.toggled.connect(
                    lambda checked, i=idx: self._on_pose_3d_toggled(i, checked))
            else:
                cb.setEnabled(False)
                cb.setToolTip("This pose's atoms could not be mapped to the 3D models.")
            self._pose_checkboxes.append(cb)
            self._poses_layout.addWidget(cb)
        self._poses_layout.addStretch(1)

    # ----- events ----------------------------------------------------------
    def _on_type_toggled(self, itype, checked):
        self._type_included[itype] = checked
        self._redraw()

    def _on_pose_3d_toggled(self, idx, checked):
        if self._on_pose_show_3d is not None:
            self._on_pose_show_3d(idx, checked)

    def _on_context_changed(self, value):
        if self._comparison is None:
            return
        self._comparison.rebuild_columns(value)
        self._update_summary()
        self._redraw()

    def _included_types(self):
        inc = {t for t, on in self._type_included.items() if on}
        return inc if inc != set(self._comparison.types) else None

    def _redraw(self):
        if self._comparison is None:
            return
        self._layout = draw_pose_msa(
            self._fig, self._comparison,
            include_types=self._included_types(),
            one_letter=self._one_letter_cb.isChecked(),
            color_for=self._color_for,
            show_consensus=self._consensus_cb.isChecked(),
            consensus_threshold=self._consensus_spin.value() / 100.0)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if (event.inaxes is None or self._on_cell_selected is None
                or not self._layout["cols"]):
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        cols = self._layout["cols"]
        if 0 <= col < len(cols) and 0 <= row < len(self._layout["rows"]):
            self._on_cell_selected(row, cols[col].key)

    def uncheck_all_3d(self):
        """Clear every 'show in 3D' checkbox (without firing extra toggles twice)."""
        for cb in self._pose_checkboxes:
            if cb.isChecked():
                cb.setChecked(False)

    def save_current(self, path):
        self._fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")


class PoseMsaDialog(QDialog):
    """Resizable pop-out window hosting the pose-comparison grid (non-modal)."""

    def __init__(self, parent, session, on_cell_selected=None, on_pose_show_3d=None):
        super().__init__(parent)
        self.setWindowTitle("ChemeleonX pose comparison")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._session = session

        self.panel = PoseMsaPanel(on_cell_selected=on_cell_selected,
                                  on_pose_show_3d=on_pose_show_3d)

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

        self.resize(980, 640)

    def set_comparison(self, comparison):
        self.panel.set_comparison(comparison)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", "chemeleonx_pose_comparison.png",
            "Images (*.png *.svg *.pdf)")
        if path:
            self.panel.save_current(path)
            self._session.logger.info("ChemeleonX: saved pose comparison to %s" % path)


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()

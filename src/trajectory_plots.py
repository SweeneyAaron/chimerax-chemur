"""Matplotlib plots for trajectory interaction analysis.

Three figures, each embedded in a Qt tab:

* **Counts** — total (and per-type) interaction count vs frame. Interactive: recolour
  per-type lines, exclude types, and restrict counting to a selection scope
  (all / ≥1 end selected / both ends selected, using the 3D ChimeraX selection).
* **Fingerprint** — occupancy heatmap (interactions x frames, present/absent).
* **Distance** — distance-vs-frame for one selected interaction.

The drawing functions are backend-agnostic (operate on a ``matplotlib.figure.Figure``)
so they back both the embedded Qt canvases and PNG/SVG export. ``TrajectoryPlotPanel``
wires them into a ``QTabWidget`` and supports clicking a heatmap row to select an
interaction and a movable current-frame marker.
"""

from __future__ import annotations

from Qt.QtCore import Qt
from Qt.QtGui import QColor, QIcon, QPixmap
from Qt.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from .colors import mpl_color

# Selection-scope options for the counts/fingerprint plots: (label, key).
_SCOPES = [
    ("All interactions", "all"),
    ("≥ 1 end selected", "any"),
    ("Both ends selected", "both"),
]

# Heatmap colormaps offered for the fingerprint.
_FP_COLORMAPS = ["Greens", "Blues", "Reds", "Purples", "Oranges", "Greys",
                 "viridis", "magma", "cividis", "hot"]


# ----- pure drawing helpers ------------------------------------------------
def draw_counts(fig, frames, total, by_type, color_for, *, scope_label=None,
                show_total=True, show_legend=True):
    """Plot precomputed total + per-type counts vs frame.

    ``by_type`` is ``{interaction_type: [count per frame]}`` (already filtered to the
    included types and selection scope); ``color_for(itype)`` returns a matplotlib
    RGBA colour. ``show_total``/``show_legend`` toggle the total line and the legend.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    if show_total:
        ax.plot(frames, total, color="black", lw=1.6, label="total")
    for itype, series in by_type.items():
        ax.plot(frames, series, lw=1.0, color=color_for(itype), label=itype)
    ax.set_xlabel("frame")
    ax.set_ylabel("interactions")
    title = "Interaction count over trajectory"
    if scope_label:
        title += " — %s" % scope_label
    ax.set_title(title)
    ax.margins(x=0)
    if show_legend and (by_type or show_total):
        ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.6)
    fig.tight_layout()
    return ax


def draw_fingerprint(fig, result, *, max_rows=50, key_filter=None, cmap="Greens"):
    """Occupancy fingerprint heatmap. Returns the ordered keys (top to bottom).

    ``key_filter(key)`` (optional) restricts which interaction rows are shown (used
    for the selection-scope filter); ``cmap`` is the matplotlib colormap name.
    """
    import numpy as np

    fig.clear()
    ax = fig.add_subplot(111)
    keys = [k for k in result._ordered_keys() if key_filter is None or key_filter(k)]
    keys = keys[:max_rows]
    labels = result.interaction_labels()
    n_frames = result.n_frames
    if not keys or not n_frames:
        ax.text(0.5, 0.5, "No interactions", ha="center", va="center")
        ax.set_axis_off()
        return []

    row_of = {k: i for i, k in enumerate(keys)}
    grid = np.zeros((len(keys), n_frames), dtype=float)
    for j, frame in enumerate(result.frames):
        for inter in frame.interactions:
            k = result.interaction_key(inter)
            r = row_of.get(k)
            if r is not None:
                grid[r, j] = 1.0

    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(
        ["%s %s" % (k[0], labels.get(k, "")) for k in keys], fontsize=5
    )
    step = max(1, n_frames // 10)
    ax.set_xticks(range(0, n_frames, step))
    ax.set_xticklabels([str(result.frame_indices[i]) for i in range(0, n_frames, step)], fontsize=6)
    ax.set_xlabel("frame")
    ax.set_title("Interaction fingerprint (occupancy)")
    fig.tight_layout()
    return keys


def draw_timeseries(fig, result, key, label, *, metric="distance", color_for=None):
    """Distance/angle vs frame for a single interaction ``key``."""
    fig.clear()
    ax = fig.add_subplot(111)
    series = result.timeseries(key)
    xs = [pt["frame"] for pt in series]
    ys = [pt[metric] for pt in series]
    if color_for is not None and key:
        color = color_for(key[0])
    else:
        color = mpl_color(key[0]) if key else "black"
    ax.plot(xs, ys, color=color, marker="o", ms=2.5, lw=1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("%s (%s)" % (metric, "Å" if metric in ("distance", "offset") else "°"))
    ax.set_title("%s\n%s" % (label, key[0] if key else ""), fontsize=8)
    ax.margins(x=0)
    fig.tight_layout()
    return ax


# ----- embedded Qt panel ---------------------------------------------------
class TrajectoryPlotPanel(QWidget):
    """Tabbed matplotlib panel: counts, fingerprint, distance.

    ``on_key_selected(key)`` is invoked when the user clicks a fingerprint row.
    """

    def __init__(self, parent=None, on_key_selected=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvas
        from matplotlib.figure import Figure

        self._on_key_selected = on_key_selected
        self._result = None
        self._key_to_endpoints = {}
        self._all_types = []
        self._type_included = {}          # counts: interaction_type -> bool
        self._type_colors = {}            # interaction_type -> matplotlib RGBA tuple
        self._type_checkboxes = {}        # counts: interaction_type -> QCheckBox
        self._fp_type_included = {}       # fingerprint: interaction_type -> bool
        self._fp_type_checkboxes = {}     # fingerprint: interaction_type -> QCheckBox
        self._fp_keys = []
        self._labels = {}
        self._markers = {}                # figure -> current-frame vertical line
        self._current_frame = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._fig_counts = Figure(figsize=(5, 3))
        self._fig_fp = Figure(figsize=(5, 3))
        self._fig_dist = Figure(figsize=(5, 3))
        self.canvas_counts = FigureCanvas(self._fig_counts)
        self.canvas_fp = FigureCanvas(self._fig_fp)
        self.canvas_dist = FigureCanvas(self._fig_dist)

        self.tabs.addTab(self._build_counts_tab(), "Counts")
        self.tabs.addTab(self._build_fingerprint_tab(), "Fingerprint")
        self.tabs.addTab(self.canvas_dist, "Distance")

        self.canvas_fp.mpl_connect("button_press_event", self._on_fp_click)

    # ----- counts tab construction ----------------------------------------
    def _build_counts_tab(self):
        tab = QWidget()
        col = QVBoxLayout(tab)
        col.setContentsMargins(2, 2, 2, 2)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Show:"))
        self._scope_combo = QComboBox()
        for label, key in _SCOPES:
            self._scope_combo.addItem(label, key)
        self._scope_combo.setToolTip(
            "Count all interactions, or only those touching the 3D selection at one "
            "or both ends.")
        self._scope_combo.currentIndexChanged.connect(lambda _i: self._recompute_counts())
        bar.addWidget(self._scope_combo)
        refresh = QPushButton("Refresh from selection")
        refresh.setToolTip("Re-read the current 3D atom selection and recount.")
        refresh.clicked.connect(self._recompute_counts)
        bar.addWidget(refresh)
        bar.addStretch(1)
        self._total_cb = QCheckBox("Total")
        self._total_cb.setChecked(True)
        self._total_cb.setToolTip("Show the total-interactions line.")
        self._total_cb.toggled.connect(lambda _c: self._recompute_counts())
        bar.addWidget(self._total_cb)
        self._legend_cb = QCheckBox("Legend")
        self._legend_cb.setChecked(True)
        self._legend_cb.setToolTip("Show the plot legend.")
        self._legend_cb.toggled.connect(lambda _c: self._recompute_counts())
        bar.addWidget(self._legend_cb)
        col.addLayout(bar)

        # Per-type include checkboxes + colour buttons (rebuilt per analysis).
        self._types_area = QScrollArea()
        self._types_area.setWidgetResizable(True)
        self._types_area.setFixedHeight(120)
        self._types_holder = QWidget()
        self._types_layout = QVBoxLayout(self._types_holder)
        self._types_layout.setContentsMargins(2, 2, 2, 2)
        self._types_area.setWidget(self._types_holder)
        col.addWidget(self._types_area)

        col.addWidget(self.canvas_counts, 1)
        return tab

    def _build_fingerprint_tab(self):
        tab = QWidget()
        col = QVBoxLayout(tab)
        col.setContentsMargins(2, 2, 2, 2)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Show:"))
        self._fp_scope_combo = QComboBox()
        for label, key in _SCOPES:
            self._fp_scope_combo.addItem(label, key)
        self._fp_scope_combo.setToolTip(
            "Show all interactions, or only those touching the 3D selection at one "
            "or both ends.")
        self._fp_scope_combo.currentIndexChanged.connect(lambda _i: self._recompute_fingerprint())
        bar.addWidget(self._fp_scope_combo)
        refresh = QPushButton("Refresh from selection")
        refresh.setToolTip("Re-read the current 3D atom selection and redraw.")
        refresh.clicked.connect(self._recompute_fingerprint)
        bar.addWidget(refresh)
        bar.addStretch(1)
        bar.addWidget(QLabel("Colour:"))
        self._fp_cmap_combo = QComboBox()
        self._fp_cmap_combo.addItems(_FP_COLORMAPS)
        self._fp_cmap_combo.setToolTip("Heatmap colormap.")
        self._fp_cmap_combo.currentIndexChanged.connect(lambda _i: self._recompute_fingerprint())
        bar.addWidget(self._fp_cmap_combo)
        col.addLayout(bar)

        # Per-type include checkboxes for the fingerprint (independent of the
        # counts tab's list; rebuilt per analysis).
        self._fp_types_area = QScrollArea()
        self._fp_types_area.setWidgetResizable(True)
        self._fp_types_area.setFixedHeight(120)
        self._fp_types_holder = QWidget()
        self._fp_types_layout = QVBoxLayout(self._fp_types_holder)
        self._fp_types_layout.setContentsMargins(2, 2, 2, 2)
        self._fp_types_area.setWidget(self._fp_types_holder)
        col.addWidget(self._fp_types_area)

        col.addWidget(self.canvas_fp, 1)
        return tab

    def _build_type_rows(self):
        # Clear existing rows.
        while self._types_layout.count():
            item = self._types_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._type_checkboxes = {}
        for itype in self._all_types:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(itype)
            cb.setChecked(self._type_included.get(itype, True))
            cb.setIcon(self._swatch(self._qcolor_for(itype)))
            cb.toggled.connect(lambda checked, t=itype: self._on_type_toggled(t, checked))
            rl.addWidget(cb, 1)
            btn = QPushButton("Colour")
            btn.clicked.connect(lambda _c=False, t=itype, c=cb: self._pick_color(t, c))
            rl.addWidget(btn, 0)
            self._type_checkboxes[itype] = cb
            self._types_layout.addWidget(row)
        self._types_layout.addStretch(1)

    def _build_fp_type_rows(self):
        while self._fp_types_layout.count():
            item = self._fp_types_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._fp_type_checkboxes = {}
        for itype in self._all_types:
            cb = QCheckBox(itype)
            cb.setChecked(self._fp_type_included.get(itype, True))
            cb.setIcon(self._swatch(self._qcolor_for(itype)))
            cb.toggled.connect(lambda checked, t=itype: self._on_fp_type_toggled(t, checked))
            self._fp_type_checkboxes[itype] = cb
            self._fp_types_layout.addWidget(cb)
        self._fp_types_layout.addStretch(1)

    def _on_fp_type_toggled(self, itype, checked):
        self._fp_type_included[itype] = checked
        self._recompute_fingerprint()

    # ----- colour helpers --------------------------------------------------
    def _color_for(self, itype):
        """Matplotlib RGBA for a type (override, else the default palette)."""
        return self._type_colors.get(itype) or mpl_color(itype)

    def _qcolor_for(self, itype):
        r, g, b, a = self._color_for(itype)
        return QColor.fromRgbF(r, g, b, a)

    @staticmethod
    def _swatch(qcolor):
        pix = QPixmap(12, 12)
        pix.fill(qcolor)
        return QIcon(pix)

    def _pick_color(self, itype, checkbox):
        color = QColorDialog.getColor(self._qcolor_for(itype), self, "Colour for %s" % itype)
        if color.isValid():
            self._type_colors[itype] = (color.redF(), color.greenF(), color.blueF(), 1.0)
            checkbox.setIcon(self._swatch(color))
            self._recompute_counts()

    def _on_type_toggled(self, itype, checked):
        self._type_included[itype] = checked
        self._recompute_counts()

    # ----- data + recompute ------------------------------------------------
    def set_result(self, result, key_to_endpoints=None):
        self._result = result
        self._key_to_endpoints = key_to_endpoints or {}
        self._markers.clear()
        self._current_frame = None
        if result is None:
            for fig in (self._fig_counts, self._fig_fp, self._fig_dist):
                fig.clear()
            self._redraw_all()
            return
        self._labels = result.interaction_labels()
        self._all_types = sorted({i.interaction_type for f in result.frames for i in f.interactions})
        self._type_included = {t: True for t in self._all_types}
        self._fp_type_included = {t: True for t in self._all_types}
        self._type_colors = {}
        self._build_type_rows()
        self._build_fp_type_rows()
        self._recompute_counts()
        self._recompute_fingerprint()
        self._fig_dist.clear()
        self.canvas_dist.draw_idle()

    def _scope(self):
        return self._scope_combo.currentData() or "all"

    def _fp_scope(self):
        return self._fp_scope_combo.currentData() or "all"

    def _passes_scope(self, key, scope):
        if scope == "all":
            return True
        groups = self._key_to_endpoints.get(key) or []
        selected = [any(getattr(a, "selected", False) for a in grp) for grp in groups]
        if scope == "any":
            return any(selected)
        # "both": every endpoint group has a selected atom (needs >= 2 groups).
        return len(selected) >= 2 and all(selected)

    def _compute_counts(self):
        result = self._result
        frames = list(result.frame_indices)
        included = [t for t in self._all_types if self._type_included.get(t, True)]
        by_type = {t: [0] * len(result.frames) for t in included}
        total = [0] * len(result.frames)
        included_set = set(included)
        for j, frame in enumerate(result.frames):
            for inter in frame.interactions:
                t = inter.interaction_type
                if t not in included_set:
                    continue
                if not self._passes_scope(result.interaction_key(inter), self._scope()):
                    continue
                by_type[t][j] += 1
                total[j] += 1
        return frames, total, by_type

    def _recompute_counts(self):
        if self._result is None:
            return
        frames, total, by_type = self._compute_counts()
        scope_label = self._scope_combo.currentText() if self._scope() != "all" else None
        draw_counts(self._fig_counts, frames, total, by_type, self._color_for,
                    scope_label=scope_label,
                    show_total=self._total_cb.isChecked(),
                    show_legend=self._legend_cb.isChecked())
        self._markers.pop(self._fig_counts, None)
        self._reapply_marker(self._fig_counts, self.canvas_counts, is_heatmap=False)
        self.canvas_counts.draw_idle()

    def _recompute_fingerprint(self):
        if self._result is None:
            return
        scope = self._fp_scope()

        def key_filter(k):
            # Excluded types never appear, even if they pass the selection scope.
            if not self._fp_type_included.get(k[0], True):
                return False
            return self._passes_scope(k, scope)

        self._fp_keys = draw_fingerprint(
            self._fig_fp, self._result,
            key_filter=key_filter,
            cmap=self._fp_cmap_combo.currentText() or "Greens",
        )
        self._markers.pop(self._fig_fp, None)
        self._reapply_marker(self._fig_fp, self.canvas_fp, is_heatmap=True)
        self.canvas_fp.draw_idle()

    # ----- distance + markers ---------------------------------------------
    def show_key(self, key):
        if self._result is None or key is None:
            return
        label = self._labels.get(key, "")
        draw_timeseries(self._fig_dist, self._result, key, label, color_for=self._color_for)
        self._markers.pop(self._fig_dist, None)
        self._reapply_marker(self._fig_dist, self.canvas_dist, is_heatmap=False)
        self.canvas_dist.draw_idle()
        self.tabs.setCurrentWidget(self.canvas_dist)

    def mark_frame(self, frame_index):
        """Draw a vertical current-frame marker on every plot (by frame number)."""
        if self._result is None:
            return
        self._current_frame = frame_index
        for fig, canvas, is_heatmap in (
            (self._fig_counts, self.canvas_counts, False),
            (self._fig_fp, self.canvas_fp, True),
            (self._fig_dist, self.canvas_dist, False),
        ):
            self._reapply_marker(fig, canvas, is_heatmap=is_heatmap)
            canvas.draw_idle()

    def _reapply_marker(self, fig, canvas, *, is_heatmap):
        if self._current_frame is None:
            return
        axes = fig.get_axes()
        if not axes:
            return
        ax = axes[0]
        line = self._markers.get(fig)
        if line is not None:
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        if is_heatmap:
            try:
                x = self._result.frame_indices.index(self._current_frame)
            except ValueError:
                x = min(range(len(self._result.frame_indices)),
                        key=lambda i: abs(self._result.frame_indices[i] - self._current_frame))
        else:
            x = self._current_frame
        self._markers[fig] = ax.axvline(x, color="red", lw=0.8, alpha=0.7)

    def save_current(self, path):
        canvas = self.tabs.currentWidget()
        # The Counts tab is a container widget; find its figure canvas.
        fig = getattr(canvas, "figure", None)
        if fig is None:
            fig = self._fig_counts
        fig.savefig(path, dpi=200, bbox_inches="tight")

    def _on_fp_click(self, event):
        if event.inaxes is None or not self._fp_keys or self._on_key_selected is None:
            return
        row = int(round(event.ydata))
        if 0 <= row < len(self._fp_keys):
            self._on_key_selected(self._fp_keys[row])

    def _redraw_all(self):
        self.canvas_counts.draw_idle()
        self.canvas_fp.draw_idle()
        self.canvas_dist.draw_idle()

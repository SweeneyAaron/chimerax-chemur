"""Qt GUI tool for Chemur interaction prediction."""

from __future__ import annotations

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import CollapsiblePanel

from Qt.QtCore import Qt
from Qt.QtGui import QColor, QIcon, QPixmap
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QLabel,
    QLineEdit, QScrollArea, QWidget, QDoubleSpinBox, QFileDialog, QDialog,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QAbstractItemView,
    QComboBox, QColorDialog, QSpinBox, QSlider, QListWidget, QListWidgetItem,
    QSizePolicy,
)

from .colors import rgba_255
from .widgets import (
    ElidedLabel, NarrowScrollArea, ResizeNotifyWidget, flow_row, labelled,
    repin_collapsible,
)

# Per-rule cutoffs we let the user edit, with display labels and spin-box setup.
_CUTOFF_COLUMNS = [
    ("distance", "Distance (Å)", 0.0, 50.0, 2, 0.1),
    ("angle", "Angle (°)", 0.0, 180.0, 1, 1.0),
    ("offset", "Offset (Å)", 0.0, 50.0, 2, 0.1),
    ("donor_angle", "Donor angle (°)", 0.0, 180.0, 1, 1.0),
]

# Top-level result sections: (label, predicate on molecule_type, has chain level).
_SECTIONS = [
    ("Protein", lambda mt: mt == "protein", True),
    ("Nucleic acid", lambda mt: mt == "nucleotide", True),
    ("Ligands", lambda mt: mt == "ligand", False),
    ("Other (ions/solvent)", lambda mt: mt not in ("protein", "nucleotide", "ligand"), False),
]

_CHECKED = Qt.CheckState.Checked
_UNCHECKED = Qt.CheckState.Unchecked
_USER_ROLE = Qt.ItemDataRole.UserRole

#: Width the docked panel asks for on first show. Everything is laid out to fit
#: here without horizontal scrolling; the user can still drag it narrower (down
#: to the 180 px floor in _build_ui) or wider.
_PREFERRED_DOCK_WIDTH = 340


def _shrinkable(widget):
    """Let `widget` be narrower than its text so it cannot pin the panel width.

    Ignored is the only horizontal policy that zeroes a widget's contribution to
    the enclosing layout's minimum: qSmartMinSize() skips the width term for
    Ignored, but Preferred still uses minimumSizeHint().width() -- which for a
    QPushButton or QCheckBox is the full text width. Measured under ChimeraX's
    Qt: QPushButton("Export interactions…") is 131 px under both Minimum and
    Preferred, and 0 under Ignored.

    Never use this on a widget inside a FlowLayout -- see flow_row().
    """
    widget.setSizePolicy(QSizePolicy.Policy.Ignored,
                         widget.sizePolicy().verticalPolicy())
    return widget


def _mirror_text_to_tooltip(button):
    """Keep `button`'s tooltip showing its full text, which may be clipped.

    Used for ModelMenuButton, whose text is the model's "#id name". NEVER elide
    or setText() one of those: ItemMenuButton.get_value() looks the model up by
    button text (chimerax/ui/widgets/item_chooser.py), so a truncated string
    silently breaks .value and therefore every analysis entry point.
    """
    def update(*_args):
        button.setToolTip(button.text())
    update()
    button.value_changed.connect(update)
    return button


class ChemurTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    help = None

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "Chemur"
        self.tool_window = MainToolWindow(self)
        self._rows = []
        self._ligand_report = []       # [LigandProtonation] from the last analysis
        self._ligand_smiles_overrides = {}  # NAME(upper) -> SMILES, from the report dialog
        self._protonation_dialog = None
        self._cutoff_spins = {}        # (rule, key) -> (spinbox, default)
        self._row_items = {}           # row idx -> [leaf items]
        self._row_haystacks = []       # row idx -> lowercase searchable text
        self._rows_by_type = {}        # interaction_type -> [row idx]
        self._type_checkboxes = {}     # interaction_type -> QCheckBox
        self._interchain_only = False  # hide intra-chain interactions (display filter)
        self._syncing = False
        self._traj_run = None          # TrajectoryRun from the last trajectory analysis
        self._traj_item_keys = {}      # occupancy-table item -> interaction key
        self._traj_table_total = 0     # distinct interactions; table may show a capped subset
        self._traj_table_shown = 0
        self._coordset_handler = None  # 'changes' handler tracking coordset playback
        self._suppress_coordset_sync = False
        self._pose_comparison = None   # PoseComparison from the last pose run
        self._pose_dialog = None
        self._pose_3d_groups = {}      # pose_index -> global pseudobond group name
        self._panels = []              # [CollapsiblePanel] needing height re-pinning
        self._repinning = False
        self._build_ui()
        self.tool_window.manage("side")
        self._set_initial_dock_width(_PREFERRED_DOCK_WIDTH)

    def _set_initial_dock_width(self, width):
        """Open the docked panel at a sane width instead of Qt's sizeHint cap.

        manage() sizes a docked tool from ui_area.sizeHint().width();
        NarrowScrollArea already reports _PREFERRED_DOCK_WIDTH there, and this
        pins it against QMainWindow's dock layout rounding it up. Skipped when
        the user has saved a tool position, so an explicit choice always wins.
        Deferred because the dock is not laid out until after manage() returns.
        """
        try:
            saved = self.session.ui.settings.tool_positions["windows"]
        except Exception:
            saved = {}
        if self.tool_name in saved:
            return

        def apply():
            try:
                # Private API, documented as "emergency access" -- never let it
                # break tool startup.
                dock = self.tool_window._dock_widget
                if dock is not None and not dock.isFloating():
                    self.session.ui.main_window.resizeDocks(
                        [dock], [width], Qt.Orientation.Horizontal)
            except Exception:
                pass

        from Qt.QtCore import QTimer
        QTimer.singleShot(0, apply)

    def delete(self):
        self._remove_coordset_handler()
        self._clear_pose_3d_groups()
        super().delete()

    def _remove_coordset_handler(self):
        if self._coordset_handler is not None:
            try:
                self._coordset_handler.remove()
            except Exception:
                pass
            self._coordset_handler = None

    # ----- UI construction -------------------------------------------------

    def _build_ui(self):
        from chimerax.atomic import AtomicStructure
        from chimerax.ui.widgets import ModelMenuButton

        parent = self.tool_window.ui_area
        # Host all content in an outer scroll area with an explicit small minimum,
        # so no child (e.g. the wide Advanced-options grid or a big result tree)
        # can pin the tool's minimum size. Without this the widest child sets the
        # floor for both the docked dock-widget and the floating window, which
        # blocks resizing the main ChimeraX window. Over-wide/tall content scrolls.
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(0, 0, 0, 0)
        # NarrowScrollArea, not QScrollArea: QScrollArea.sizeHint() is its
        # content's sizeHint bounded to 36 x font height (~540 px), and that is
        # the number ChimeraX reads to size a docked tool.
        scroll_host = NarrowScrollArea()
        scroll_host.setWidgetResizable(True)
        scroll_host.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_host.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_host.setMinimumWidth(180)
        outer.addWidget(scroll_host)

        # Expanded CollapsiblePanels pin their height when toggled, from a
        # sizeHint that ignores heightForWidth; re-pin them whenever the dock is
        # resized so wrapped rows/labels neither clip nor over-reserve.
        content = ResizeNotifyWidget(self._repin_panels)
        scroll_host.setWidget(content)
        layout = QVBoxLayout()
        layout.setContentsMargins(9, 9, 9, 9)
        layout.setSpacing(6)
        content.setLayout(layout)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Structure:"))
        self._model_button = ModelMenuButton(self.session, class_filter=AtomicStructure)
        _shrinkable(_mirror_text_to_tooltip(self._model_button))
        chooser.addWidget(self._model_button, 1)
        layout.addLayout(chooser)

        # Primary options (always visible). A flow row so these wrap onto extra
        # lines in a narrow dock instead of setting a ~520 px horizontal floor.
        self._addh_cb = QCheckBox("Add hydrogens")
        self._addh_cb.setChecked(True)
        self._addh_cb.setToolTip("Run 'addh' first so H-bonds can be detected.")
        self._protonate_cb = QCheckBox("Protonate ligands")
        self._protonate_cb.setToolTip(
            "Use Dimorphite-DL to protonate ligand SMILES at the pH set in "
            "Advanced options.")
        self._selected_cb = QCheckBox("Selected atoms only")
        self._selected_cb.setToolTip("Keep only interactions with at least one selected partner.")
        self._skip_bio_cb = QCheckBox("Skip biopolymer-internal")
        self._skip_bio_cb.setToolTip("Drop interactions internal to a biopolymer.")
        layout.addWidget(flow_row(self._addh_cb, self._protonate_cb,
                                  self._selected_cb, self._skip_bio_cb))

        self._add_panel(layout, self._build_advanced_panel())

        self._analyze_btn = QPushButton("Analyze interactions")
        self._analyze_btn.clicked.connect(self._analyze)
        layout.addWidget(self._analyze_btn)

        # Cue shown only when some non-standard residue failed protonation; the
        # link opens the protonation report. Hidden when nothing needs attention.
        self._protonation_cue = QLabel("")
        self._protonation_cue.setStyleSheet("color: #b35900;")
        self._protonation_cue.setWordWrap(True)
        self._protonation_cue.linkActivated.connect(
            lambda _link: self._open_protonation_dialog())
        self._protonation_cue.hide()
        layout.addWidget(self._protonation_cue)

        # Type legend (populated after each analysis).
        self._legend_area = QScrollArea()
        self._legend_area.setWidgetResizable(True)
        self._legend_area.setFixedHeight(44)
        # Scroll the legend horizontally instead of letting many type-swatches
        # widen (and pin) the docked panel.
        self._legend_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._legend_area.setMinimumWidth(80)
        self._legend_holder = QWidget()
        self._legend_layout = QHBoxLayout(self._legend_holder)
        self._legend_layout.setContentsMargins(2, 2, 2, 2)
        self._legend_area.setWidget(self._legend_holder)
        layout.addWidget(self._legend_area)

        # Show/hide-all + export controls, split across two flow rows. As one
        # QHBoxLayout these eight widgets set a ~800 px floor, which is what
        # pushed the export buttons off-screen in a docked panel.
        show_all = QPushButton("Show all")
        show_all.clicked.connect(lambda: self._set_all_visible(True))
        hide_all = QPushButton("Hide all")
        hide_all.clicked.connect(lambda: self._set_all_visible(False))

        self._interchain_cb = QCheckBox("Inter-chain only")
        self._interchain_cb.setToolTip(
            "Hide interactions whose two partners are in the same chain, leaving "
            "only inter-chain contacts. Live display filter (no re-analysis).\n"
            "Note: a type whose contacts are all intra-chain shows its legend "
            "checkbox unticked while this is on.")
        self._interchain_cb.toggled.connect(self._on_interchain_toggled)
        layout.addWidget(flow_row(show_all, hide_all, self._interchain_cb))

        self._export_scope = QComboBox()
        # (label, scope key). Scope decides which rows _export_interactions writes.
        for label, key in (
            ("All interactions", "all"),
            ("Selected rows", "rows"),
            ("Selected atoms (3D)", "atoms"),
        ):
            self._export_scope.addItem(label, key)
        self._export_scope.setToolTip(
            "Which interactions to export: all, the rows selected in the results "
            "tree, or those touching atoms selected in the 3D view.")
        self._export_scope.setEnabled(False)
        # No "Export:" label: the combo sits between two Export buttons and its
        # own tooltip says what it scopes. Worth ~55 px on this row.
        # Deliberately NOT _shrinkable(): it goes into a flow_row below, and an
        # Ignored horizontal policy makes QWidgetItem::sizeHint() report width 0,
        # which laid the combo out 0 px wide instead of 166.
        self._export_interactions_btn = QPushButton("Export data…")
        self._export_interactions_btn.setToolTip(
            "Export interaction data to a JSON or CSV file.")
        self._export_interactions_btn.setEnabled(False)
        self._export_interactions_btn.clicked.connect(self._export_interactions)

        self._export_btn = QPushButton("Export 2D…")
        self._export_btn.setToolTip("Export a 2D schematic of the selected residues' interactions.")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_diagram)
        layout.addWidget(flow_row(self._export_scope,
                                  self._export_interactions_btn, self._export_btn))

        # Free-text keyword filter for the results tree (list-only; does not
        # touch pseudobond display or the legend). Multiple whitespace-separated
        # keywords narrow with AND semantics.
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText(
            "filter by keyword — e.g. THR 133, chain A, OG1, hbond, pi-pi")
        self._search_box.setClearButtonEnabled(True)
        self._search_box.textChanged.connect(self._apply_search_filter)
        search_row.addWidget(self._search_box, 1)
        layout.addLayout(search_row)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        # Short headers with the units in tooltips: spelled out, "Distance (Å)"
        # alone eats 82 px of a 300 px panel and squeezes column 0 -- the actual
        # content -- down to 69 px, forcing the tree's own horizontal scrollbar.
        self._tree.setHeaderLabels(["Type / partner", "Dist", "Angle", "Offset"])
        for col, tip in ((1, "Distance (Å)"), (2, "Angle (°)"), (3, "Offset (Å)")):
            self._tree.headerItem().setToolTip(col, tip)
        header = self._tree.header()
        # Off, or the last column absorbs spare width instead of column 0.
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tree.itemSelectionChanged.connect(self._tree_selection_changed)
        self._tree.itemChanged.connect(self._on_item_changed)
        # Small minimum height so the tree scrolls internally rather than forcing
        # the outer scroll area's vertical scrollbar; column 0 stretches for width.
        self._tree.setMinimumHeight(80)
        layout.addWidget(self._tree, 1)

        # Elided, not plain: this is fed full export paths and exception strings,
        # which would otherwise widen the panel's minimum at runtime.
        self._status = ElidedLabel("")
        layout.addWidget(self._status)

        self._add_panel(layout, self._build_trajectory_panel())
        self._add_panel(layout, self._build_pose_panel())

    def _add_panel(self, layout, panel):
        """Add a CollapsiblePanel and keep its pinned height honest."""
        # ChimeraX forces spacing 0 on a panel's content layout on macOS
        # (chimerax/ui/widgets/composite.py), which leaves every control inside a
        # section flush against its neighbours. Match the main column instead.
        panel.content_area.layout().setSpacing(6)
        self._panels.append(panel)
        # Fires after CollapsiblePanel's own handler, so the panel has already
        # toggled and just needs its pinned height corrected.
        panel.toggle_button.clicked.connect(self._repin_panels)
        layout.addWidget(panel)

    def _repin_panels(self):
        """Re-pin expanded collapsible sections to the height they need now.

        CollapsiblePanel pins its content height from a sizeHint that ignores
        heightForWidth, so a word-wrapped label or a wrapped flow row is clipped
        once the dock narrows. Deferred to the next event-loop turn because on a
        resize the children have not been re-laid-out yet, and guarded because
        re-pinning resizes the content widget that triggered us.
        """
        if self._repinning:
            return
        self._repinning = True

        def apply():
            try:
                for panel in self._panels:
                    repin_collapsible(panel)
            finally:
                self._repinning = False

        from Qt.QtCore import QTimer
        QTimer.singleShot(0, apply)

    def _build_pose_panel(self):
        """Collapsible 'Compare docked poses' section: MSA-like pose comparison."""
        from chimerax.atomic import AtomicStructure
        from chimerax.ui.widgets import ModelMenuButton

        panel = CollapsiblePanel(self.tool_window.ui_area, title="Compare docked poses",
                                 margins=(12, 0, 0, 0))
        content = panel.content_area.layout()

        # ~1800 px as a single unwrapped line, and pinned the tool that wide even
        # while this panel was collapsed -- the single biggest cause of the
        # permanent horizontal scrollbar. Full text on hover.
        intro = QLabel("Compare docked ligand poses residue-by-residue.")
        intro.setWordWrap(True)
        intro.setToolTip(
            "Analyse several docked ligand poses (each a separate model) against one "
            "receptor, then compare their interaction patterns residue-by-residue in "
            "an MSA-like figure. Ligand-only poses are merged with the receptor; poses "
            "that already contain the protein are analysed as-is. Targets small-molecule "
            "ligands.")
        content.addWidget(intro)

        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Receptor:"))
        self._pose_receptor = ModelMenuButton(self.session, class_filter=AtomicStructure)
        _shrinkable(_mirror_text_to_tooltip(self._pose_receptor))
        self._pose_receptor.value_changed.connect(self._refresh_pose_list)
        r_row.addWidget(self._pose_receptor, 1)
        content.addLayout(r_row)

        poses_label = QLabel("Poses (tick the docked-ligand models):")
        poses_label.setWordWrap(True)
        content.addWidget(poses_label)
        self._pose_list = QListWidget()
        self._pose_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._pose_list.setFixedHeight(120)
        content.addWidget(self._pose_list)

        refresh = QPushButton("Refresh")
        refresh.setToolTip("Re-read the open models (other than the receptor).")
        refresh.clicked.connect(self._refresh_pose_list)
        sel_all = QPushButton("Select all")
        sel_all.clicked.connect(lambda: self._set_pose_checks(True))
        sel_none = QPushButton("None")
        sel_none.clicked.connect(lambda: self._set_pose_checks(False))
        content.addWidget(flow_row(refresh, sel_all, sel_none))

        ctx_row = QHBoxLayout()
        ctx_row.addWidget(QLabel("Sequence context ±"))
        self._pose_context = QSpinBox()
        self._pose_context.setRange(0, 3)
        self._pose_context.setToolTip(
            "Add this many flanking residues around each interacting residue so the "
            "columns read like a sequence stretch (adjustable later in the figure too).")
        ctx_row.addWidget(self._pose_context)
        ctx_row.addStretch(1)
        content.addLayout(ctx_row)

        self._pose_btn = QPushButton("Analyze poses")
        self._pose_btn.setToolTip(
            "Run Chemur on each ticked pose against the receptor and build the "
            "comparison figure (using the analysis options above).")
        self._pose_btn.clicked.connect(self._analyze_poses)
        content.addWidget(self._pose_btn)

        self._pose_status = ElidedLabel("")
        content.addWidget(self._pose_status)

        self._pose_show_btn = QPushButton("Show comparison figure…")
        self._pose_show_btn.setEnabled(False)
        self._pose_show_btn.clicked.connect(self._open_pose_dialog)
        content.addWidget(self._pose_show_btn)

        self._refresh_pose_list()
        return panel

    def _build_trajectory_panel(self):
        """Collapsible 'Trajectory' section: per-frame analysis + plots."""
        panel = CollapsiblePanel(self.tool_window.ui_area, title="Trajectory (MD)",
                                 margins=(12, 0, 0, 0))
        content = panel.content_area.layout()

        # Frame range / options. A flow row of label+control pairs rather than a
        # 6-column QGridLayout, whose columns are additive (~435 px floor);
        # labelled() keeps each label glued to its spin box across a wrap.
        self._traj_start = QSpinBox(); self._traj_start.setRange(0, 10_000_000)
        self._traj_stop = QSpinBox(); self._traj_stop.setRange(0, 10_000_000)
        self._traj_stop.setSpecialValueText("end"); self._traj_stop.setValue(0)
        self._traj_stop.setToolTip("Stop frame (exclusive). 'end' uses all frames.")
        self._traj_stride = QSpinBox(); self._traj_stride.setRange(1, 1_000_000)
        self._traj_stride.setValue(1)
        self._traj_processes = QSpinBox(); self._traj_processes.setRange(1, 256)
        self._traj_processes.setValue(1)
        self._traj_processes.setToolTip("Worker processes for parallel per-frame analysis.")
        self._traj_excl_solvent = QCheckBox("Exclude solvent")
        self._traj_excl_solvent.setChecked(True)
        self._traj_excl_ions = QCheckBox("Exclude ions")
        self._traj_excl_ions.setChecked(True)
        content.addWidget(flow_row(
            labelled("Start", self._traj_start),
            labelled("Stop", self._traj_stop),
            labelled("Stride", self._traj_stride),
            labelled("Processes", self._traj_processes),
            self._traj_excl_solvent, self._traj_excl_ions))

        self._traj_analyze_btn = QPushButton("Analyze trajectory")
        self._traj_analyze_btn.clicked.connect(self._analyze_trajectory)
        content.addWidget(self._traj_analyze_btn)

        self._traj_status = ElidedLabel("")
        content.addWidget(self._traj_status)

        # Occupancy table.
        self._traj_table = QTreeWidget()
        self._traj_table.setColumnCount(4)
        self._traj_table.setHeaderLabels(["Interaction", "Type", "Occupancy", "Mean dist (Å)"])
        traj_header = self._traj_table.header()
        # As with the results tree: let column 0, not "Mean dist (Å)", take the
        # spare width.
        traj_header.setStretchLastSection(False)
        traj_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            traj_header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._traj_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._traj_table.setMinimumHeight(120)
        self._traj_table.itemSelectionChanged.connect(self._on_occupancy_selected)
        content.addWidget(self._traj_table)

        # Frame scrubber (drives the 3D view's active coordset).
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frame:"))
        self._traj_slider = QSlider(Qt.Orientation.Horizontal)
        self._traj_slider.setEnabled(False)
        self._traj_slider.valueChanged.connect(self._on_frame_slider)
        slider_row.addWidget(self._traj_slider, 1)
        self._traj_frame_label = QLabel("—")
        slider_row.addWidget(self._traj_frame_label)
        content.addLayout(slider_row)

        # Live per-frame interactions: draw the current frame's interactions as
        # pseudobonds in 3D and feed them to the standard interactions section.
        # "(3D + list)" lives in the tooltip; spelled out it was the widest single
        # control left in the panel.
        self._traj_live_cb = QCheckBox("Show frame interactions")
        self._traj_live_cb.setToolTip(
            "Draw the current frame's interactions as pseudobonds and list them in the "
            "main interactions panel. Updates when you release the frame slider.")
        self._traj_live_cb.toggled.connect(self._on_live_toggled)
        content.addWidget(self._traj_live_cb)

        # Per-type include checkboxes controlling which types are drawn/listed per
        # frame (default all on; rebuilt per analysis).
        self._traj_frame_type_included = {}   # interaction_type -> bool
        self._traj_frame_type_cbs = {}        # interaction_type -> QCheckBox
        self._traj_frame_types_area = QScrollArea()
        self._traj_frame_types_area.setWidgetResizable(True)
        self._traj_frame_types_area.setFixedHeight(100)
        self._traj_frame_types_holder = QWidget()
        self._traj_frame_types_layout = QVBoxLayout(self._traj_frame_types_holder)
        self._traj_frame_types_layout.setContentsMargins(2, 2, 2, 2)
        self._traj_frame_types_area.setWidget(self._traj_frame_types_holder)
        content.addWidget(self._traj_frame_types_area)

        # Plots live in a separate, resizable pop-out window (created lazily so the
        # matplotlib import doesn't slow tool launch). With many interactions the
        # fingerprint/heatmap needs far more room than the docked panel can give.
        self._traj_plots = None        # TrajectoryPlotPanel inside the dialog
        self._traj_plots_dialog = None

        self._traj_show_plots_btn = QPushButton("Show plots…")
        self._traj_show_plots_btn.setEnabled(False)
        self._traj_show_plots_btn.clicked.connect(self._open_traj_plots)
        self._traj_export_btn = QPushButton("Export CSV…")
        self._traj_export_btn.setEnabled(False)
        self._traj_export_btn.clicked.connect(self._export_trajectory)
        content.addWidget(flow_row(self._traj_show_plots_btn, self._traj_export_btn))

        return panel

    def _build_advanced_panel(self):
        # margins: the default 30 px title indent is a tenth of a 300 px panel.
        panel = CollapsiblePanel(self.tool_window.ui_area, title="Advanced options",
                                 margins=(12, 0, 0, 0))
        content = panel.content_area.layout()

        ph_row = QHBoxLayout()
        ph_row.addWidget(QLabel("Protonation pH:"))
        self._ph_spin = QDoubleSpinBox()
        self._ph_spin.setRange(0.0, 14.0)
        self._ph_spin.setDecimals(1)
        self._ph_spin.setSingleStep(0.1)
        self._ph_spin.setValue(7.4)
        self._ph_spin.setToolTip(
            "pH used by Dimorphite-DL when 'Protonate ligands' is enabled.")
        ph_row.addWidget(self._ph_spin)
        ph_row.addStretch(1)
        content.addLayout(ph_row)

        cutoffs_label = QLabel("Geometry cutoffs (edited values are applied):")
        cutoffs_label.setWordWrap(True)
        content.addWidget(cutoffs_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(180)
        inner = QWidget()
        grid = QGridLayout(inner)
        scroll.setWidget(inner)
        content.addWidget(scroll)

        grid.addWidget(QLabel("Rule"), 0, 0)
        for col, (_key, label, *_rest) in enumerate(_CUTOFF_COLUMNS, start=1):
            grid.addWidget(QLabel(label), 0, col)
        row = 1
        for rule_name, rule in self._default_rules().items():
            grid.addWidget(QLabel(rule_name), row, 0)
            for col, (key, _label, lo, hi, decimals, step) in enumerate(_CUTOFF_COLUMNS, start=1):
                value = rule.get(key)
                if not isinstance(value, (int, float)):
                    continue
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
                spin.setValue(float(value))
                grid.addWidget(spin, row, col)
                self._cutoff_spins[(rule_name, key)] = (spin, float(value))
            row += 1

        reset = QPushButton("Reset cutoffs to defaults")
        reset.clicked.connect(self._reset_cutoffs)
        content.addWidget(reset)

        self._protonation_btn = QPushButton("Protonation report…")
        self._protonation_btn.setToolTip(
            "Show the protonation state of each non-standard residue and fix any "
            "that failed automatic protonation.")
        self._protonation_btn.setEnabled(False)
        self._protonation_btn.clicked.connect(self._open_protonation_dialog)
        content.addWidget(self._protonation_btn)
        return panel

    def _default_rules(self):
        try:
            from chemur.profile import load_profile
            return load_profile("default").get("rules", {})
        except Exception:
            return {}

    def _reset_cutoffs(self):
        for spin, default in self._cutoff_spins.values():
            spin.setValue(default)

    # ----- analysis --------------------------------------------------------

    def _collect_rule_overrides(self):
        overrides = {}
        for (rule_name, key), (spin, default) in self._cutoff_spins.items():
            if abs(spin.value() - default) > 1e-9:
                overrides.setdefault(rule_name, {})[key] = spin.value()
        return overrides or None

    def _analyze(self):
        structure = self._model_button.value
        if structure is None:
            self.session.logger.error("Chemur: choose a structure first.")
            return
        from .runner import run_interactions

        self._status.setText("Analyzing…")
        self._analyze_btn.setEnabled(False)
        try:
            run = run_interactions(
                self.session,
                structure,
                protonate=self._protonate_cb.isChecked(),
                protonate_ph=self._ph_spin.value(),
                ligand_smiles=self._ligand_smiles_overrides or None,
                add_hydrogens=self._addh_cb.isChecked(),
                rule_overrides=self._collect_rule_overrides(),
                selected_only=self._selected_cb.isChecked(),
                skip_biopolymer_internal=self._skip_bio_cb.isChecked(),
            )
        except Exception as e:
            self.session.logger.report_exception()
            self._status.setText("Error: %s" % e)
            return
        finally:
            self._analyze_btn.setEnabled(True)

        rows = run.rows
        self._rows = rows
        self._ligand_report = run.ligand_report
        self._populate(rows)
        self._build_legend(rows)
        if self._interchain_only:
            self._reapply_pseudobond_visibility()
        self._export_btn.setEnabled(bool(rows))
        self._export_scope.setEnabled(bool(rows))
        self._export_interactions_btn.setEnabled(bool(rows))
        # _populate already drew the tree (nested, or flat for an active query)
        # and set the status line accordingly.
        self._refresh_protonation_ui()

    # ----- trajectory analysis ---------------------------------------------

    def _analyze_trajectory(self):
        structure = self._model_button.value
        if structure is None:
            self.session.logger.error("Chemur: choose a structure first.")
            return
        if structure.num_coordsets < 2:
            self._traj_status.setText(
                "#%s has one frame; load a trajectory (multiple coordsets) first."
                % structure.id_string)
            return
        from .runner import run_trajectory

        stop = self._traj_stop.value() or None  # 0 ('end') -> all frames
        self._traj_status.setText("Analyzing trajectory…")
        self._traj_analyze_btn.setEnabled(False)
        try:
            run = run_trajectory(
                self.session,
                structure,
                exclude_solvent=self._traj_excl_solvent.isChecked(),
                exclude_ions=self._traj_excl_ions.isChecked(),
                frame_start=self._traj_start.value(),
                frame_stop=stop,
                frame_stride=self._traj_stride.value(),
                processes=self._traj_processes.value(),
                protonate=self._protonate_cb.isChecked(),
                protonate_ph=self._ph_spin.value(),
                ligand_smiles=self._ligand_smiles_overrides or None,
                rule_overrides=self._collect_rule_overrides(),
                progress=self._traj_progress,
            )
        except Exception as e:
            self.session.logger.report_exception()
            self._traj_status.setText("Error: %s" % e)
            return
        finally:
            self._traj_analyze_btn.setEnabled(True)

        self._traj_run = run
        self._populate_occupancy(run)
        self._build_frame_type_rows(run)
        self._ensure_plot_dialog()
        self._traj_plots.set_result(run.result, run.key_to_endpoints)
        self._setup_frame_slider(run)
        self._traj_show_plots_btn.setEnabled(True)
        self._traj_export_btn.setEnabled(True)
        self._open_traj_plots()  # pop the plots out into their own window
        if self._traj_live_cb.isChecked():
            self._redraw_frame_interactions()
        msg = "Analysed %d frames; %d distinct interactions." % (
            run.result.n_frames, self._traj_table_total)
        if self._traj_table_total > self._traj_table_shown:
            msg += " Table shows top %d by occupancy." % self._traj_table_shown
        self._traj_status.setText(msg)

    def _traj_progress(self, done, total):
        self._traj_status.setText("Analyzing trajectory… frame %d/%d" % (done, total))
        ui = getattr(self.session, "ui", None)
        if ui is not None and hasattr(ui, "processEvents"):
            ui.processEvents()

    def _ensure_plot_dialog(self):
        if self._traj_plots_dialog is None:
            self._traj_plots_dialog = TrajectoryPlotsDialog(
                self.tool_window.ui_area, self.session,
                on_key_selected=self._on_traj_key_selected)
            self._traj_plots = self._traj_plots_dialog.panel

    def _open_traj_plots(self):
        if self._traj_run is None:
            return
        self._ensure_plot_dialog()
        self._traj_plots_dialog.show()
        self._traj_plots_dialog.raise_()
        self._traj_plots_dialog.activateWindow()

    # Cap the occupancy table: listing every transient contact (often tens of
    # thousands) is slow to build and not useful; show the most-occupied ones.
    _OCCUPANCY_TABLE_CAP = 1000

    def _populate_occupancy(self, run):
        self._traj_table.clear()
        self._traj_item_keys.clear()
        agg = run.result.aggregates()   # single pass, cached
        occ = agg["occupancy"]
        labels = agg["labels"]
        mean_distance = agg["mean_distance"]
        ordered = agg["ordered_keys"]
        shown = ordered[:self._OCCUPANCY_TABLE_CAP]
        items = []
        for key in shown:
            mean = mean_distance.get(key)
            item = QTreeWidgetItem([
                labels.get(key, "?"),
                key[0],
                "%.0f%%" % (occ[key] * 100.0),
                "%.2f" % mean if mean is not None else "—",
            ])
            self._traj_item_keys[id(item)] = key
            items.append(item)
        self._traj_table.setUpdatesEnabled(False)
        self._traj_table.addTopLevelItems(items)
        self._traj_table.setUpdatesEnabled(True)
        self._traj_table_total = len(ordered)
        self._traj_table_shown = len(shown)

    def _on_occupancy_selected(self):
        items = self._traj_table.selectedItems()
        if not items:
            return
        key = self._traj_item_keys.get(id(items[0]))
        if key is not None:
            self._show_traj_key(key)

    def _on_traj_key_selected(self, key):
        """Called when a fingerprint heatmap row is clicked."""
        self._show_traj_key(key)
        # Mirror the selection in the occupancy table.
        for i in range(self._traj_table.topLevelItemCount()):
            item = self._traj_table.topLevelItem(i)
            if self._traj_item_keys.get(id(item)) == key:
                self._traj_table.setCurrentItem(item)
                break

    def _show_traj_key(self, key):
        if self._traj_run is None:
            return
        self._open_traj_plots()
        self._traj_plots.show_key(key)
        atoms = self._traj_run.key_to_atoms.get(key)
        if atoms:
            from chimerax.atomic import Atoms
            self.session.selection.clear()
            Atoms([a for a in atoms if a is not None]).selected = True

    def _setup_frame_slider(self, run):
        ids = run.frame_coordset_ids
        self._traj_slider.blockSignals(True)
        self._traj_slider.setEnabled(bool(ids))
        self._traj_slider.setMinimum(0)
        self._traj_slider.setMaximum(max(0, len(ids) - 1))
        self._traj_slider.setValue(0)
        self._traj_slider.blockSignals(False)
        self._install_coordset_handler(run.structure)
        if ids:
            self._on_frame_slider(0)

    def _on_frame_slider(self, pos):
        # User moved our slider: drive the 3D coordset and (live) redraw.
        self._apply_frame(pos, set_coordset=True)

    def _apply_frame(self, pos, *, set_coordset):
        run = self._traj_run
        if run is None or not run.frame_coordset_ids:
            return
        pos = max(0, min(pos, len(run.frame_coordset_ids) - 1))
        cs_id = run.frame_coordset_ids[pos]
        frame_index = run.result.frame_indices[pos]
        if set_coordset:
            # Suppress the coordset-change handler for the change we initiate.
            self._suppress_coordset_sync = True
            try:
                run.structure.active_coordset_id = cs_id
            except Exception:
                pass
            finally:
                self._suppress_coordset_sync = False
        self._traj_frame_label.setText("%d" % frame_index)
        if self._traj_plots is not None:
            self._traj_plots.mark_frame(frame_index)
        # Overwrite the interactions on every tick so they trace through.
        self._redraw_frame_interactions()

    # ----- track ChimeraX coordset playback -------------------------------

    def _install_coordset_handler(self, structure):
        self._remove_coordset_handler()
        self._coordset_handler = structure.triggers.add_handler(
            "changes", self._on_coordset_changed)

    def _on_coordset_changed(self, _trig_name, data):
        # Fired by ChimeraX when the structure changes; sync our slider when the
        # trajectory plays (active coordset moves) unless we initiated the change.
        if self._suppress_coordset_sync or self._traj_run is None:
            return
        structure, changes = data
        if structure is not self._traj_run.structure:
            return
        if "active_coordset changed" not in changes.structure_reasons():
            return
        cs_id = structure.active_coordset_id
        ids = self._traj_run.frame_coordset_ids
        try:
            pos = ids.index(cs_id)
        except ValueError:
            # Coordset between analysed frames: show the frame number, leave the
            # drawn interactions (and slider) on the last analysed frame.
            self._traj_frame_label.setText("%d (not analysed)" % cs_id)
            return
        self._traj_slider.blockSignals(True)
        self._traj_slider.setValue(pos)
        self._traj_slider.blockSignals(False)
        self._apply_frame(pos, set_coordset=False)

    # ----- live per-frame interactions ------------------------------------

    def _build_frame_type_rows(self, run):
        while self._traj_frame_types_layout.count():
            item = self._traj_frame_types_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._traj_frame_type_cbs = {}
        types = sorted({i.interaction_type for f in run.result.frames for i in f.interactions})
        self._traj_frame_type_included = {t: True for t in types}
        for itype in types:
            cb = QCheckBox(itype)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, t=itype: self._on_frame_type_toggled(t, checked))
            self._traj_frame_type_cbs[itype] = cb
            self._traj_frame_types_layout.addWidget(cb)
        self._traj_frame_types_layout.addStretch(1)

    def _on_frame_type_toggled(self, itype, checked):
        self._traj_frame_type_included[itype] = checked
        if self._traj_live_cb.isChecked():
            self._redraw_frame_interactions()

    def _on_live_toggled(self, checked):
        if checked:
            self._redraw_frame_interactions()

    def _redraw_frame_interactions(self):
        if self._traj_run is None or not self._traj_live_cb.isChecked():
            return
        pos = self._traj_slider.value()
        frame_indices = self._traj_run.result.frame_indices
        if not (0 <= pos < len(frame_indices)):
            return
        frame_index = frame_indices[pos]
        included = {t for t, on in self._traj_frame_type_included.items() if on}
        from .runner import draw_frame_interactions
        try:
            rows = draw_frame_interactions(
                self.session, self._traj_run, frame_index, included_types=included)
        except Exception as e:
            self.session.logger.report_exception()
            self._traj_status.setText("Error drawing frame: %s" % e)
            return
        self._rows = rows
        self._ligand_report = []
        self._populate(rows)
        self._build_legend(rows)
        if self._interchain_only:
            self._reapply_pseudobond_visibility()
        self._export_scope.setEnabled(bool(rows))
        self._export_interactions_btn.setEnabled(bool(rows))
        self._export_btn.setEnabled(bool(rows))

    def _export_trajectory(self):
        if self._traj_run is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area, "Export trajectory interactions",
            "chemur_trajectory.csv",
            "CSV (*.csv);;JSON (*.json)")
        if not path:
            return
        if path.lower().endswith(".json"):
            self._traj_run.result.to_json(path)
        else:
            self._traj_run.result.to_csv(path)
        self.session.logger.info("Chemur: exported trajectory data to %s" % path)

    # ----- docked-pose comparison ------------------------------------------

    def _refresh_pose_list(self):
        """List the open structures (other than the receptor) as checkable poses."""
        if not hasattr(self, "_pose_list"):
            return  # value_changed can fire while the panel is still being built
        from chimerax.atomic import AtomicStructure
        try:
            from chimerax.markers import MarkerSet
        except ImportError:
            MarkerSet = ()
        receptor = self._pose_receptor.value
        checked = set(self._checked_poses())
        self._pose_list.clear()
        for m in self.session.models.list(type=AtomicStructure):
            if m is receptor or (MarkerSet and isinstance(m, MarkerSet)):
                continue
            item = QListWidgetItem("#%s %s" % (m.id_string, m.name))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(_CHECKED if m in checked else _UNCHECKED)
            item.setData(_USER_ROLE, m)
            self._pose_list.addItem(item)

    def _checked_poses(self):
        out = []
        for i in range(self._pose_list.count()):
            item = self._pose_list.item(i)
            if item.checkState() == _CHECKED:
                out.append(item.data(_USER_ROLE))
        return out

    def _set_pose_checks(self, checked):
        state = _CHECKED if checked else _UNCHECKED
        for i in range(self._pose_list.count()):
            self._pose_list.item(i).setCheckState(state)

    def _analyze_poses(self):
        receptor = self._pose_receptor.value
        poses = self._checked_poses()
        if receptor is None or not poses:
            self._pose_status.setText("Choose a receptor and tick at least one pose.")
            return
        if receptor in poses:
            self._pose_status.setText("The receptor cannot also be a pose.")
            return
        from . import pose_compare

        # Drop any pseudobonds drawn for a previous pose run.
        self._clear_pose_3d_groups()
        if self._pose_dialog is not None:
            self._pose_dialog.panel.uncheck_all_3d()

        run_kwargs = dict(
            protonate=self._protonate_cb.isChecked(),
            protonate_ph=self._ph_spin.value(),
            ligand_smiles=self._ligand_smiles_overrides or None,
            add_hydrogens=self._addh_cb.isChecked(),
            rule_overrides=self._collect_rule_overrides(),
            skip_biopolymer_internal=True,
        )
        self._pose_status.setText("Analysing %d poses…" % len(poses))
        self._pose_btn.setEnabled(False)
        try:
            comparison = pose_compare.compute_pose_comparison(
                self.session, receptor, poses, run_kwargs,
                context=self._pose_context.value(), progress=self._pose_progress)
        except Exception as e:
            self.session.logger.report_exception()
            self._pose_status.setText("Error: %s" % e)
            return
        finally:
            self._pose_btn.setEnabled(True)

        self._pose_comparison = comparison
        has_data = bool(comparison.rows) and bool(comparison.columns)
        if not has_data:
            self._pose_show_btn.setEnabled(False)
            # Short on screen, actionable on hover: the status label is a single
            # elided line, so the "what to do about it" half would be cut off.
            msg = "No protein-ligand interactions found."
            tip = "No protein-ligand interactions found for the selected poses."
            if comparison.n_untemplated:
                msg += " %d ligand(s) not templated." % comparison.n_untemplated
                tip += (" %d pose ligand(s) could not be templated — try enabling "
                        "'Protonate ligands', or supply a ligand SMILES."
                        % comparison.n_untemplated)
            self._pose_status.setText(msg)
            self._pose_status.setToolTip(tip)
            return
        n_unmapped = sum(1 for r in comparison.rows if not r.fully_mapped)
        msg = "%d poses × %d residues; %d interaction type(s)." % (
            len(comparison.rows), len(comparison.columns), len(comparison.types))
        if n_unmapped:
            msg += " %d pose(s) figure-only (no 3D mapping)." % n_unmapped
        self._pose_status.setText(msg)
        self._pose_show_btn.setEnabled(True)
        self._open_pose_dialog()

    def _pose_progress(self, done, total):
        self._pose_status.setText("Analysing poses… %d/%d" % (done, total))
        ui = getattr(self.session, "ui", None)
        if ui is not None and hasattr(ui, "processEvents"):
            ui.processEvents()

    def _ensure_pose_dialog(self):
        if self._pose_dialog is None:
            from .pose_plot import PoseMsaDialog
            self._pose_dialog = PoseMsaDialog(
                self.tool_window.ui_area, self.session,
                on_cell_selected=self._on_pose_cell_selected,
                on_pose_show_3d=self._on_pose_show_3d)

    def _open_pose_dialog(self):
        if self._pose_comparison is None:
            return
        self._ensure_pose_dialog()
        self._pose_dialog.set_comparison(self._pose_comparison)
        self._pose_dialog.show()
        self._pose_dialog.raise_()
        self._pose_dialog.activateWindow()

    def _on_pose_cell_selected(self, row_index, col_key):
        """Select the clicked pose+residue's atoms in 3D (no-op for figure-only poses)."""
        rows = self._pose_comparison.cell_rows.get((row_index, col_key), [])
        from chimerax.atomic import Atoms
        atoms = [ep.atom for r in rows for ep in (r.endpoint1, r.endpoint2)
                 if ep.atom is not None]
        self.session.selection.clear()
        if atoms:
            Atoms(atoms).selected = True

    def _on_pose_show_3d(self, pose_index, show):
        """Draw or hide one pose's protein-ligand interactions as 3D pseudobonds."""
        from .runner import draw_pose_rows, clear_pose_group
        comparison = self._pose_comparison
        if comparison is None or not (0 <= pose_index < len(comparison.rows)):
            return
        pose_row = comparison.rows[pose_index]
        group_name = "Chemur pose %s" % pose_row.name
        if show:
            # All of this pose's protein-ligand rows live across its cell_rows.
            rows = [row for (idx, _key), rows in comparison.cell_rows.items()
                    if idx == pose_index for row in rows]
            try:
                draw_pose_rows(self.session, rows, group_name)
                self._pose_3d_groups[pose_index] = group_name
                self.session.logger.info(
                    "Chemur: drew %d interactions for %s in 3D."
                    % (len(rows), pose_row.name))
            except Exception as e:
                self.session.logger.warning(
                    "Chemur: could not draw %s in 3D (%s)" % (pose_row.name, e))
        else:
            clear_pose_group(self.session, group_name)
            self._pose_3d_groups.pop(pose_index, None)

    def _clear_pose_3d_groups(self):
        from .runner import clear_pose_group
        for group_name in list(self._pose_3d_groups.values()):
            try:
                clear_pose_group(self.session, group_name)
            except Exception:
                pass
        self._pose_3d_groups = {}

    # ----- protonation report ----------------------------------------------

    def _refresh_protonation_ui(self):
        """Sync the advanced-panel button, the main-window cue, and any open
        protonation dialog to the latest ``self._ligand_report``."""
        self._protonation_btn.setEnabled(bool(self._ligand_report))
        n_attention = sum(1 for r in self._ligand_report if r.status == "failed")
        if n_attention:
            self._protonation_cue.setText(
                '⚠ %d non-standard residue(s) need protonation review — '
                '<a href="#">open report</a>' % n_attention)
            self._protonation_cue.show()
        else:
            self._protonation_cue.hide()
        if self._protonation_dialog is not None:
            self._protonation_dialog.set_report(self._ligand_report)

    def _open_protonation_dialog(self):
        if not self._ligand_report:
            return
        if self._protonation_dialog is None:
            self._protonation_dialog = LigandProtonationDialog(
                self.tool_window.ui_area, self._rerun_with_ligand_smiles)
            self._protonation_dialog.finished.connect(self._on_protonation_dialog_closed)
        self._protonation_dialog.set_report(self._ligand_report)
        self._protonation_dialog.show()
        self._protonation_dialog.raise_()

    def _on_protonation_dialog_closed(self, _result):
        self._protonation_dialog = None

    def _rerun_with_ligand_smiles(self, overrides):
        """Merge dialog SMILES edits into the stored overrides and re-analyze.

        ``overrides`` maps an upper-cased residue name to a SMILES string; the
        edits accumulate so a fixed residue stays fixed on later analyses.
        """
        self._ligand_smiles_overrides.update(overrides)
        self._analyze()

    # ----- results tree ----------------------------------------------------

    def _populate(self, rows):
        """Index a fresh analysis, then draw the tree per the current search box.

        Only computes the per-row data (search haystacks, type index); the tree
        itself is built by ``_apply_search_filter`` so the nested-vs-flat layout
        always matches whether a query is active.
        """
        self._row_haystacks = []
        self._rows_by_type = {}
        for idx, row in enumerate(rows):
            self._rows_by_type.setdefault(row.interaction_type, []).append(idx)
            self._row_haystacks.append(_row_haystack(row))
        self._apply_search_filter()

    def _row_check_state(self, idx):
        """Tree checkbox state mirroring the row's pseudobond visibility, so the
        check marks survive a tree rebuild (e.g. when the search query changes)."""
        pb = self._rows[idx].pseudobond
        if pb is not None and not pb.deleted and not pb.display:
            return _UNCHECKED
        return _CHECKED

    def _build_nested(self, indices):
        """Draw the grouped tree (Section → Chain → Residue → interaction)."""
        section_nodes = {}
        chain_nodes = {}
        residue_nodes = {}

        def section_item(label):
            item = section_nodes.get(label)
            if item is None:
                item = QTreeWidgetItem([label])
                _make_group(item)
                self._tree.addTopLevelItem(item)
                section_nodes[label] = item
            return item

        def residue_item(label, has_chain, endpoint):
            parent = section_item(label)
            if has_chain:
                ckey = (label, endpoint.chain_id)
                cnode = chain_nodes.get(ckey)
                if cnode is None:
                    cnode = QTreeWidgetItem(["Chain %s" % endpoint.chain_id])
                    _make_group(cnode)
                    parent.addChild(cnode)
                    chain_nodes[ckey] = cnode
                parent = cnode
            rkey = (label, endpoint.chain_id, endpoint.residue_key)
            rnode = residue_nodes.get(rkey)
            if rnode is None:
                rnode = QTreeWidgetItem([endpoint.residue_label])
                _make_group(rnode)
                parent.addChild(rnode)
                residue_nodes[rkey] = rnode
            return rnode

        def section_for(mt):
            for label, predicate, has_chain in _SECTIONS:
                if predicate(mt):
                    return label, has_chain
            return _SECTIONS[-1][0], _SECTIONS[-1][2]

        for idx in indices:
            row = self._rows[idx]
            self._row_items[idx] = []
            seen_nodes = set()
            for endpoint, partner in ((row.endpoint1, row.endpoint2),
                                      (row.endpoint2, row.endpoint1)):
                label, has_chain = section_for(endpoint.molecule_type)
                rnode = residue_item(label, has_chain, endpoint)
                if id(rnode) in seen_nodes:
                    continue
                seen_nodes.add(id(rnode))
                leaf = QTreeWidgetItem([
                    "%s ⇄ %s" % (row.interaction_type, partner.atom_label),
                    _fmt(row.distance), _fmt(row.angle), _fmt(row.offset),
                ])
                leaf.setData(0, _USER_ROLE, idx)
                leaf.setToolTip(0, row.geometry_tooltip())
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, self._row_check_state(idx))
                rnode.addChild(leaf)
                self._row_items[idx].append(leaf)

        self._tree.expandToDepth(0)

    def _build_flat(self, indices):
        """Draw matching interactions as a flat, un-nested list (search mode)."""
        for idx in indices:
            row = self._rows[idx]
            leaf = QTreeWidgetItem([
                "%s: %s ⇄ %s" % (row.interaction_type,
                                 row.endpoint1.atom_label, row.endpoint2.atom_label),
                _fmt(row.distance), _fmt(row.angle), _fmt(row.offset),
            ])
            leaf.setData(0, _USER_ROLE, idx)
            leaf.setToolTip(0, row.geometry_tooltip())
            leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            leaf.setCheckState(0, self._row_check_state(idx))
            self._tree.addTopLevelItem(leaf)
            self._row_items[idx] = [leaf]

    def _build_legend(self, rows):
        # Clear previous legend widgets.
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._type_checkboxes = {}

        counts = {}
        for row in rows:
            counts[row.interaction_type] = counts.get(row.interaction_type, 0) + 1
        for itype in sorted(counts):
            cb = QCheckBox("%s (%d)" % (itype, counts[itype]))
            cb.setChecked(True)
            cb.setIcon(_swatch(itype))
            cb.toggled.connect(lambda checked, t=itype: self._set_type_visible(t, checked))
            self._legend_layout.addWidget(cb)
            self._type_checkboxes[itype] = cb
        self._legend_layout.addStretch(1)

    # ----- search filter ---------------------------------------------------

    def _apply_search_filter(self, text=None):
        """Redraw the tree for the current query.

        Empty query → the full grouped (nested) tree. Non-empty query → a flat
        list of only the interactions matching every keyword (AND, case-
        insensitive substring). Rebuilding (rather than hiding rows) guarantees
        the list visibly updates and shows just the matches. This is list-only:
        it never changes pseudobond display or the legend.
        """
        if text is None:
            text = self._search_box.text()
        tokens = text.lower().split()
        self._syncing = True
        try:
            self._tree.clear()
            self._row_items = {}
            indices = [idx for idx in range(len(self._rows))
                       if not (self._interchain_only
                               and self._row_is_intrachain(self._rows[idx]))]
            if tokens:
                indices = [idx for idx in indices
                           if all(tok in self._row_haystacks[idx] for tok in tokens)]
                self._build_flat(indices)
            elif self._interchain_only:
                self._build_nested(indices)
            else:
                self._build_nested(range(len(self._rows)))
        finally:
            self._syncing = False
        self._update_search_status(tokens)

    def _update_search_status(self, tokens):
        if not tokens:
            if self._interchain_only:
                self._status.setText("%d of %d interactions (inter-chain only)"
                                     % (len(self._row_items), len(self._rows)))
            else:
                self._status.setText("%d interactions" % len(self._rows))
            return
        self._status.setText("%d of %d interactions match"
                             % (len(self._row_items), len(self._rows)))

    # ----- visibility ------------------------------------------------------

    @staticmethod
    def _row_is_intrachain(row):
        return row.endpoint1.chain_id == row.endpoint2.chain_id

    def _on_interchain_toggled(self, checked):
        self._interchain_only = checked
        self._apply_search_filter()            # rebuild the (filtered) tree
        self._reapply_pseudobond_visibility()  # re-apply 3D visibility
        self._refresh_legend_states()

    def _reapply_pseudobond_visibility(self):
        """Set each row's pseudobond display to its type-checkbox state; the
        inter-chain gate in ``_apply_row_visibility`` then force-hides intra
        rows when the filter is on."""
        for idx, row in enumerate(self._rows):
            cb = self._type_checkboxes.get(row.interaction_type)
            self._apply_row_visibility(idx, cb.isChecked() if cb else True)

    def _apply_row_visibility(self, idx, visible):
        if idx < 0 or idx >= len(self._rows):
            return
        if self._interchain_only and self._row_is_intrachain(self._rows[idx]):
            visible = False
        was = self._syncing
        self._syncing = True
        try:
            pb = self._rows[idx].pseudobond
            if pb is not None and not pb.deleted:
                pb.display = visible
            state = _CHECKED if visible else _UNCHECKED
            for leaf in self._row_items.get(idx, ()):
                if leaf.checkState(0) != state:
                    leaf.setCheckState(0, state)
        finally:
            self._syncing = was

    def _on_item_changed(self, item, column):
        if self._syncing:
            return
        data = item.data(0, _USER_ROLE)
        if data is None:
            return  # group node: Qt cascades to leaf children, handled there
        visible = item.checkState(0) == _CHECKED
        self._apply_row_visibility(int(data), visible)
        self._refresh_legend_states()

    def _set_type_visible(self, itype, visible):
        if self._syncing:
            return
        for idx in self._rows_by_type.get(itype, ()):
            self._apply_row_visibility(idx, visible)

    def _set_all_visible(self, visible):
        for idx in range(len(self._rows)):
            self._apply_row_visibility(idx, visible)
        for cb in self._type_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(visible)
            cb.blockSignals(False)

    def _refresh_legend_states(self):
        # Sync each type checkbox to whether any of its rows are visible.
        for itype, cb in self._type_checkboxes.items():
            visible = any(
                self._rows[i].pseudobond is not None and not self._rows[i].pseudobond.deleted
                and self._rows[i].pseudobond.display
                for i in self._rows_by_type.get(itype, ())
            )
            cb.blockSignals(True)
            cb.setChecked(visible)
            cb.blockSignals(False)

    # ----- 3D selection + export ------------------------------------------

    def _tree_selection_changed(self):
        items = self._tree.selectedItems()
        if not items:
            return
        indices = set()
        for item in items:
            self._collect_row_indices(item, indices)
        if not indices:
            return
        from chimerax.atomic import Atoms
        atoms = []
        for idx in indices:
            if 0 <= idx < len(self._rows):
                row = self._rows[idx]
                atoms.append(row.endpoint1.atom)
                atoms.append(row.endpoint2.atom)
        if not atoms:
            return
        self.session.selection.clear()
        Atoms(atoms).selected = True

    def _collect_row_indices(self, item, indices):
        data = item.data(0, _USER_ROLE)
        if data is not None:
            indices.add(int(data))
            return
        for i in range(item.childCount()):
            self._collect_row_indices(item.child(i), indices)

    def _export_diagram(self):
        if not self._rows:
            self.session.logger.error("Chemur: run an analysis first.")
            return
        structure = self._model_button.value
        if structure is None:
            self.session.logger.error("Chemur: choose a structure first.")
            return
        selected_residues = {a.residue for a in structure.atoms if a.selected}
        if not selected_residues:
            self.session.logger.error(
                "Chemur: select a ligand or residue(s) first, then Export.")
            self._status.setText("Select a ligand or residue(s) first.")
            return

        from .diagram import build_scene
        title = structure.name
        try:
            scene = build_scene(self.session, structure, self._rows, selected_residues)
        except Exception as e:
            self.session.logger.report_exception()
            self._status.setText("Diagram error: %s" % e)
            return

        # Show an interactive preview; the user edits and saves from the dialog.
        self._preview_dialog = DiagramPreviewDialog(
            self.tool_window.ui_area, self.session, scene, title)
        self._preview_dialog.show()
        self._preview_dialog.raise_()
        summary = scene.summary
        scope = "selected residues" if summary.get("scoped") else "all shown interactions"
        self._status.setText("Diagram preview: %d residues, %d edges (%s)"
                             % (summary["nodes"], summary["edges"], scope))

    def _rows_for_scope(self):
        """The InteractionRows to export, per the export-scope dropdown."""
        scope = self._export_scope.currentData()
        if scope == "rows":
            indices = set()
            for item in self._tree.selectedItems():
                self._collect_row_indices(item, indices)
            return [self._rows[i] for i in sorted(indices)
                    if 0 <= i < len(self._rows)]
        if scope == "atoms":
            return [r for r in self._rows
                    if r.endpoint1.atom.selected or r.endpoint2.atom.selected]
        return list(self._rows)

    def _export_interactions(self):
        if not self._rows:
            self.session.logger.error("Chemur: run an analysis first.")
            return
        rows = self._rows_for_scope()
        if not rows:
            msg = ("No interactions in the chosen scope; select some rows or "
                   "atoms first, or switch the scope to 'All interactions'.")
            self.session.logger.error("Chemur: " + msg)
            self._status.setText(msg)
            return

        path, flt = QFileDialog.getSaveFileName(
            self.tool_window.ui_area, "Export interactions",
            "chemur_interactions.json", "JSON (*.json);;CSV (*.csv)")
        if not path:
            return

        from . import export
        is_csv = "csv" in flt.lower() or path.lower().endswith(".csv")
        try:
            if is_csv:
                export.write_csv(rows, path)
            else:
                export.write_json(rows, path)
        except Exception as e:
            self.session.logger.report_exception()
            self._status.setText("Export error: %s" % e)
            return
        self.session.logger.info(
            "Chemur exported %d interactions: %s" % (len(rows), path))
        self._status.setText("Exported %d interactions to %s" % (len(rows), path))


class TrajectoryPlotsDialog(QDialog):
    """Resizable pop-out window hosting the trajectory plots.

    Plots are split out of the docked tool because the fingerprint heatmap needs
    much more room than the side panel can give when there are many interactions.
    Non-modal (opened with ``show()``) so the main window stays usable.
    """

    def __init__(self, parent, session, on_key_selected=None):
        super().__init__(parent)
        self.setWindowTitle("Chemur trajectory plots")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._session = session

        from .trajectory_plots import TrajectoryPlotPanel
        self.panel = TrajectoryPlotPanel(on_key_selected=on_key_selected)

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

        self.resize(900, 680)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", "chemur_trajectory.png",
            "Images (*.png *.svg *.pdf)")
        if path:
            self.panel.save_current(path)
            self._session.logger.info("Chemur: saved plot to %s" % path)


class DiagramPreviewDialog(QDialog):
    """Interactive 2D interaction diagram: drag residues, tune what's shown,
    recolour types, switch structure style, then save the view."""

    _STYLE_OPTIONS = [("Colored", "default"), ("Black & white", "bw"),
                      ("Bold", "bold"), ("Comic", "comic")]

    def __init__(self, parent, session, scene, title=None):
        super().__init__(parent)
        self.setWindowTitle("Chemur interaction diagram")
        self._session = session
        self._scene = scene
        self._edges_by_id = {id(e): e for e in scene.edges}

        from .interactive import InteractiveDiagramView

        outer = QVBoxLayout(self)
        summary = scene.summary
        scope = "selected residues" if summary.get("scoped") else "all shown interactions"
        outer.addWidget(QLabel(
            "%d residues · %d interaction edges · %d interactions  (%s)"
            % (summary["nodes"], summary["edges"], summary["interactions"], scope)))
        hint = QLabel("Drag a residue or ligand to reposition it; its interactions "
                      "follow. Scroll to zoom.")
        hint.setStyleSheet("color: #666666;")
        outer.addWidget(hint)

        self._view = InteractiveDiagramView(scene)
        self._view.setMinimumSize(640, 520)
        body = QHBoxLayout()
        body.addWidget(self._view, 1)
        body.addWidget(self._build_controls(), 0)
        outer.addLayout(body, 1)

        buttons = QHBoxLayout()
        reset_btn = QPushButton("Reset layout")
        reset_btn.clicked.connect(self._view.reset_layout)
        buttons.addWidget(reset_btn)
        buttons.addStretch(1)
        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        outer.addLayout(buttons)

    # ----- controls column -------------------------------------------------

    def _build_controls(self):
        panel = QWidget()
        panel.setFixedWidth(264)
        col = QVBoxLayout(panel)
        col.setContentsMargins(6, 0, 0, 0)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Structure:"))
        combo = QComboBox()
        for label, key in self._STYLE_OPTIONS:
            combo.addItem(label, key)
        combo.currentIndexChanged.connect(
            lambda i, c=combo: self._view.set_structure_style(c.itemData(i)))
        style_row.addWidget(combo, 1)
        col.addLayout(style_row)

        col.addWidget(QLabel("Interaction types:"))
        for interaction_type in self._view.types_present():
            row = QHBoxLayout()
            cb = QCheckBox(interaction_type)
            cb.setChecked(True)
            self._set_check_swatch(cb, self._view.color_for_type(interaction_type))
            cb.toggled.connect(
                lambda on, t=interaction_type: self._view.set_type_visible(t, on))
            row.addWidget(cb, 1)
            color_btn = QPushButton("Colour…")
            color_btn.clicked.connect(
                lambda _=False, t=interaction_type, c=cb: self._pick_color(t, c))
            row.addWidget(color_btn)
            col.addLayout(row)

        col.addWidget(QLabel("Individual interactions:"))
        self._edge_tree = QTreeWidget()
        self._edge_tree.setHeaderHidden(True)
        self._populate_edge_tree()
        self._edge_tree.itemChanged.connect(self._on_edge_item_changed)
        col.addWidget(self._edge_tree, 1)

        legend_cb = QCheckBox("Show legend on diagram")
        legend_cb.setChecked(True)
        legend_cb.toggled.connect(self._view.set_legend_visible)
        col.addWidget(legend_cb)
        return panel

    def _populate_edge_tree(self):
        self._edge_tree.blockSignals(True)
        by_type = {}
        for edge in self._scene.edges:
            by_type.setdefault(edge.interaction_type, []).append(edge)
        for interaction_type in sorted(by_type):
            parent = QTreeWidgetItem([interaction_type])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            self._edge_tree.addTopLevelItem(parent)
            parent.setExpanded(True)
            for edge in by_type[interaction_type]:
                dist = "" if edge.distance is None else "  %.1f Å" % edge.distance
                leaf = QTreeWidgetItem(["%s – %s%s" % (edge.unitA.label, edge.unitB.label, dist)])
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, _CHECKED)
                leaf.setData(0, _USER_ROLE, id(edge))
                parent.addChild(leaf)
        self._edge_tree.blockSignals(False)

    def _on_edge_item_changed(self, item, column):
        eid = item.data(0, _USER_ROLE)
        if eid is None:
            return
        edge = self._edges_by_id.get(eid)
        if edge is not None:
            self._view.set_edge_visible(edge, item.checkState(0) == _CHECKED)

    def _pick_color(self, interaction_type, checkbox):
        color = QColorDialog.getColor(
            self._view.color_for_type(interaction_type), self,
            "Colour for %s" % interaction_type)
        if color.isValid():
            self._view.set_type_color(interaction_type, color)
            self._set_check_swatch(checkbox, color)

    def _set_check_swatch(self, checkbox, qcolor):
        pix = QPixmap(12, 12)
        pix.fill(qcolor)
        checkbox.setIcon(QIcon(pix))

    def _save(self):
        path, _flt = QFileDialog.getSaveFileName(
            self, "Save interaction diagram", "chemur_interactions.png",
            "PNG image (*.png);;SVG image (*.svg);;PDF document (*.pdf)")
        if not path:
            return
        try:
            self._view.render_to(path)
        except Exception:
            self._session.logger.report_exception()
            return
        self._session.logger.info("Chemur saved interaction diagram: %s" % path)


class LigandProtonationDialog(QDialog):
    """Per-non-standard-residue protonation status, with inline SMILES fixes.

    Lists each ligand residue: its protonation status (templated / failed), the
    SMILES that set its chemistry, and its net formal charge. The SMILES cell is
    editable, so a residue that failed automatic protonation can be given a
    corrected SMILES; "Re-run analysis" feeds the edits back through the engine.
    """

    _COLUMNS = ["Residue", "Status", "Net charge", "SMILES (editable)", "Reason"]

    def __init__(self, parent, rerun_callback):
        super().__init__(parent)
        self.setWindowTitle("Chemur protonation report")
        self._rerun_callback = rerun_callback
        self._rows = []  # [(name, original_smiles, QLineEdit)]

        outer = QVBoxLayout(self)
        intro = QLabel(
            "Each non-standard residue's protonation state comes from a SMILES "
            "template. Residues marked ✗ failed automatic protonation and need a "
            "SMILES below; edit any cell and re-run.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #666666;")
        outer.addWidget(intro)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(len(self._COLUMNS))
        self._tree.setHeaderLabels(self._COLUMNS)
        self._tree.setRootIsDecorated(False)
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._tree.header()
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for col in (0, 1, 2, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setMinimumSize(720, 280)
        outer.addWidget(self._tree, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        rerun_btn = QPushButton("Re-run analysis")
        rerun_btn.setToolTip("Re-analyze using the SMILES entered above.")
        rerun_btn.clicked.connect(self._rerun)
        buttons.addWidget(rerun_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        outer.addLayout(buttons)

    def set_report(self, report):
        self._tree.clear()
        self._rows = []
        for entry in report:
            failed = entry.status == "failed"
            item = QTreeWidgetItem([
                entry.residue_label,
                "✗ failed" if failed else "✓ templated",
                "" if failed else _format_charge(entry.net_charge),
                "",  # SMILES lives in an editable widget (set below)
                entry.reason or "",
            ])
            self._tree.addTopLevelItem(item)

            smiles = entry.smiles_used or entry.input_smiles or ""
            edit = QLineEdit(smiles)
            edit.setPlaceholderText("enter a SMILES for this residue")
            self._tree.setItemWidget(item, 3, edit)
            self._rows.append((entry.name, smiles, edit))

            if failed:
                color = QColor("#b35900")
                for col in range(self._tree.columnCount()):
                    item.setForeground(col, color)
                if entry.reason:
                    item.setToolTip(4, entry.reason)

    def _rerun(self):
        overrides = {}
        for name, original, edit in self._rows:
            text = edit.text().strip()
            if text and text != original:
                overrides[name.upper()] = text
        if not overrides:
            return
        self._rerun_callback(overrides)


def _format_charge(charge):
    if not charge:
        return "0"
    return "%+d" % charge


# Natural-language aliases so users can type "hydrogen bond" / "pi-pi" / "salt
# bridge" and match the engine's terse interaction_type strings.
_TYPE_ALIASES = {
    "hbond": "hydrogen bond h-bond",
    "weak_hbond": "weak hydrogen bond",
    "salt_bridge": "salt bridge ionic",
    "metal_coordination": "metal coordination",
    "amide_bridge": "amide bridge",
    "solvent_bridge": "solvent bridge water",
    "pipi_stack": "pi-pi pi pi stack stacking aromatic",
    "cation_pi": "cation-pi cation pi",
    "anion_pi": "anion-pi anion pi",
    "hbond_pi": "hbond-pi hydrogen bond pi",
    "amide_pi": "amide-pi amide pi",
    "halogen_bond": "halogen bond",
    "halogen_pi": "halogen-pi halogen pi",
    "ch_pi": "ch-pi ch pi",
    "hydrophobic": "hydrophobic",
}


def _row_haystack(row):
    """Lowercase searchable text for one interaction row: interaction type (plus
    aliases) and both endpoints' molecule type, chain, residue, and atom."""
    itype = row.interaction_type
    parts = [itype, itype.replace("_", " "), _TYPE_ALIASES.get(itype, "")]
    for ep in (row.endpoint1, row.endpoint2):
        parts += [ep.molecule_type or "", ep.chain_id or "",
                  ep.residue_name or "", str(ep.residue_id or ""),
                  ep.atom.name or "", ep.atom_label]
    return " ".join(parts).lower()


def _make_group(item):
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
    item.setCheckState(0, _CHECKED)


def _swatch(interaction_type):
    r, g, b, a = rgba_255(interaction_type)
    pix = QPixmap(12, 12)
    pix.fill(QColor(r, g, b, a))
    return QIcon(pix)


def _fmt(value):
    return "" if value is None else "%.2f" % value

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
from .lifecycle import ResultLifecycle, RunCancelled
from .view_state import ContactView, is_polymer_intrachain
from .widgets import (
    ElidedLabel, NarrowScrollArea, ResizeNotifyWidget, flow_row, labelled,
    repin_collapsible,
)

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
        self._lifecycle = ResultLifecycle()
        self._static_run = None
        self._active_result = None
        self._active_id_to_atom = {}
        self._view = ContactView([])
        self._source_overrides = {}
        self._assembly_paths = {}
        self._session_handlers = []
        self._preview_dialog = None
        self._pose_positions = {}
        self._cleanup_in_progress = False
        self._current_frame_index = None
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
        from chimerax.core.models import REMOVE_MODELS, MODEL_POSITION_CHANGED
        self._session_handlers = [
            session.triggers.add_handler(REMOVE_MODELS, self._on_models_removed),
            session.triggers.add_handler(MODEL_POSITION_CHANGED, self._on_model_position_changed),
        ]
        self._model_button.value_changed.connect(self._input_model_changed)
        self._input_model_changed()
        self._refresh_engine_info()

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
            if self._lifecycle.disposed:
                return
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
        if self._lifecycle.disposed:
            return
        graphics = self._lifecycle.dispose()
        self._remove_coordset_handler()
        for handler in self._session_handlers:
            handler.remove()
        self._session_handlers = []
        self._close_graphics(graphics)
        self._clear_pose_3d_groups()
        for name in ("_protonation_dialog", "_preview_dialog", "_traj_plots_dialog", "_pose_dialog"):
            self._close_dialog(name)
        self._rows = []
        self._view = ContactView([])
        self._row_items = {}
        self._rows_by_type = {}
        self._row_haystacks = []
        self._ligand_report = []
        self._traj_item_keys.clear()
        self._traj_plots = None
        self._pose_positions = {}
        self._active_result = self._static_run = self._traj_run = self._pose_comparison = None
        self._active_id_to_atom.clear()
        self._source_overrides.clear()
        self._assembly_paths.clear()
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
        self._addh_cb.setToolTip(
            "Run 'addh' first so H-bonds can be detected. Hydrogens directly bonded "
            "to either hydrophobic contact endpoint do not block that contact. "
            "Other hydrogens and heavy atoms still participate in occlusion checks.")
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

        self._interchain_cb = QCheckBox("Hide intra-chain polymer contacts")
        self._interchain_cb.setToolTip(
            "Hide contacts within the same polymer chain of the same structure. "
            "Ligand–protein and ligand–nucleic acid contacts remain visible, even "
            "when they share a chain label. Live display filter (no re-analysis).")
        self._interchain_cb.toggled.connect(self._on_interchain_toggled)
        layout.addWidget(flow_row(show_all, hide_all, self._interchain_cb))

        self._export_scope = QComboBox()
        # (label, scope key). Scope decides which rows _export_interactions writes.
        for label, key in (
            ("All displayed contacts", "all"),
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
        self._export_interactions_btn = QPushButton("Export displayed contacts…")
        self._export_interactions_btn.setToolTip(
            "Export visible contacts in the chosen scope; native groups remain available in assigned results.")
        self._export_interactions_btn.setEnabled(False)
        self._export_interactions_btn.clicked.connect(self._export_interactions)

        self._export_assigned_btn = QPushButton("Export assigned results…")
        self._export_assigned_btn.setToolTip("Complete native assigned results, including hidden or unmapped interactions; no raw candidates.")
        self._export_assigned_btn.setEnabled(False)
        self._export_assigned_btn.clicked.connect(self._export_assigned)
        self._owner_label = ElidedLabel("No analysis results")
        layout.addWidget(self._owner_label)
        layout.addWidget(flow_row(self._export_assigned_btn))
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
            if self._lifecycle.disposed:
                return
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
        self._pose_export_btn = QPushButton("Export assigned pose results…")
        self._pose_export_btn.setEnabled(False)
        self._pose_export_btn.clicked.connect(self._export_pose_results)
        content.addWidget(flow_row(self._pose_show_btn, self._pose_export_btn))

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
        self._traj_export_btn = QPushButton("Export assigned trajectory…")
        self._traj_export_btn.setEnabled(False)
        self._traj_export_btn.clicked.connect(self._export_trajectory)
        content.addWidget(flow_row(self._traj_show_plots_btn, self._traj_export_btn))

        return panel

    def _build_advanced_panel(self):
        # margins: the default 30 px title indent is a tenth of a 300 px panel.
        panel = CollapsiblePanel(self.tool_window.ui_area, title="Advanced options",
                                 margins=(12, 0, 0, 0))
        content = panel.content_area.layout()

        self._engine_info = QLabel("")
        self._engine_info.setWordWrap(True)
        content.addWidget(self._engine_info)
        self._enclosure_cb = QCheckBox("Detect hydrophobic enclosure")
        self._enclosure_cb.setChecked(True)
        self._enclosure_cb.setToolTip("Uses the hydrophobic distance cutoff. Disabling enclosure restores ordinary hydrophobic contacts.")
        content.addWidget(self._enclosure_cb)
        self._assembly_path_label = ElidedLabel("No assembly override")
        content.addWidget(self._assembly_path_label)
        assembly_load = QPushButton("Load assembly override JSON…")
        assembly_load.clicked.connect(self._load_assembly_override)
        assembly_clear = QPushButton("Clear override")
        assembly_clear.clicked.connect(self._clear_assembly_override)
        content.addWidget(flow_row(assembly_load, assembly_clear))
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

        grid.addWidget(QLabel("Geometry parameter"), 0, 0)
        grid.addWidget(QLabel("Value"), 0, 1)
        row = 1
        previous_rule = None
        for parameter in self._geometry_parameters():
            rule_name, key = parameter["rule"], parameter["key"]
            if rule_name != previous_rule:
                label = QLabel(parameter.get("rule_label", rule_name))
                label.setStyleSheet("font-weight: bold;")
                grid.addWidget(label, row, 0, 1, 2)
                row += 1
                previous_rule = rule_name
            text = parameter["label"]
            units = parameter.get("units", "")
            label = QLabel(text + (" (%s)" % units if units else ""))
            label.setWordWrap(True)
            description = parameter.get("description", text)
            references = parameter.get("references", ())
            tooltip = description + ("\n" + "\n".join(references) if references else "")
            label.setToolTip(tooltip)
            grid.addWidget(label, row, 0)
            value = float(parameter["value"])
            spin = QDoubleSpinBox()
            spin.setRange(parameter["minimum"], parameter["maximum"])
            spin.setDecimals(parameter["decimals"])
            spin.setSingleStep(parameter["step"])
            spin.setValue(value)
            spin.setToolTip(tooltip)
            grid.addWidget(spin, row, 1)
            self._cutoff_spins[(rule_name, key)] = (spin, value)
            row += 1

        reset = QPushButton("Reset cutoffs to defaults")
        reset.clicked.connect(self._reset_cutoffs)
        content.addWidget(reset)

        self._protonation_btn = QPushButton("Chemical product report…")
        self._protonation_btn.setToolTip(
            "Show ligand and modified-polymer products, membership and diagnostics; correct standalone ligand SMILES.")
        self._protonation_btn.setEnabled(False)
        self._protonation_btn.clicked.connect(self._open_protonation_dialog)
        content.addWidget(self._protonation_btn)
        return panel

    def _geometry_parameters(self):
        try:
            from chemur.geometry_catalogue import geometry_parameters
            return geometry_parameters("default")
        except (ImportError, AttributeError):
            # The compatibility gate displays the update instruction and prevents analysis.
            return []

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
        token = self._begin_run("static", [structure])
        if token is None:
            return
        from .runner import run_interactions
        self._status.setText("Analyzing…")
        try:
            run = run_interactions(self.session, structure, **self._analysis_options(structure),
                add_hydrogens=self._addh_cb.isChecked(), selected_only=self._selected_cb.isChecked(),
                skip_biopolymer_internal=self._skip_bio_cb.isChecked(), cancel_check=token.check_cancelled)
            token.check_cancelled()
            state = self._commit_run(token, run)
            self._static_run = run
            self._activate_result(state)
            self._rows = run.rows
            self._ligand_report = run.ligand_report
            self._populate(run.rows)
            self._build_legend(run.rows)
            self._reapply_pseudobond_visibility()
            for button in (self._export_btn, self._export_scope, self._export_interactions_btn):
                button.setEnabled(bool(run.rows))
            self._refresh_protonation_ui()
        except RunCancelled:
            if not self._lifecycle.disposed:
                self._status.setText("Analysis cancelled because its source changed or closed.")
        except Exception as error:
            if not self._lifecycle.disposed:
                self.session.logger.report_exception()
                self._status.setText("Error: %s" % error)
        finally:
            self._finish_run(token)

    # ----- trajectory analysis ---------------------------------------------

    def _analyze_trajectory(self):
        structure = self._model_button.value
        if structure is None or structure.num_coordsets < 2:
            self._traj_status.setText("Choose a structure with multiple coordinate sets.")
            return
        token = self._begin_run("trajectory", [structure])
        if token is None:
            return
        from .runner import run_trajectory
        self._traj_status.setText("Analyzing trajectory…")
        completed = False
        try:
            run = run_trajectory(self.session, structure, **self._analysis_options(structure),
                exclude_solvent=self._traj_excl_solvent.isChecked(), exclude_ions=self._traj_excl_ions.isChecked(),
                frame_start=self._traj_start.value(), frame_stop=self._traj_stop.value() or None,
                frame_stride=self._traj_stride.value(), processes=self._traj_processes.value(),
                progress=self._traj_progress, cancel_check=token.check_cancelled)
            token.check_cancelled()
            state = self._commit_run(token, run)
            self._traj_run = run
            self._activate_result(state)
            self._rows = []
            self._ligand_report = run.ligand_report
            self._populate([])
            self._build_legend([])
            self._refresh_protonation_ui()
            for button in (self._export_scope, self._export_interactions_btn, self._export_btn):
                button.setEnabled(False)
            self._populate_occupancy(run)
            self._build_frame_type_rows(run)
            self._ensure_plot_dialog()
            self._traj_plots.set_result(run.result, run.key_to_endpoints, frame_endpoints=run.frame_endpoints)
            self._setup_frame_slider(run)
            self._traj_show_plots_btn.setEnabled(True)
            self._traj_export_btn.setEnabled(True)
            self._open_traj_plots()
            self._traj_status.setText("Analysed %d frames; %d distinct interactions." % (run.result.n_frames, self._traj_table_total))
            completed = True
        except RunCancelled:
            if not self._lifecycle.disposed:
                self._traj_status.setText("Trajectory analysis cancelled.")
        except Exception as error:
            if not self._lifecycle.disposed:
                self.session.logger.report_exception()
                self._traj_status.setText("Error: %s" % error)
        finally:
            self._finish_run(token)
        if completed and not self._lifecycle.disposed:
            self._on_frame_slider(0)

    def _traj_progress(self, done, total):
        token = self._lifecycle.active_run
        if token is None:
            raise RunCancelled("No active analysis")
        token.check_cancelled()
        self._traj_status.setText("Analyzing trajectory… frame %d/%d" % (done, total))
        ui = getattr(self.session, "ui", None)
        if ui is not None and hasattr(ui, "processEvents"):
            ui.processEvents()
        token.check_cancelled()

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
        if self._traj_run is None or self._lifecycle.disposed:
            return
        self._open_traj_plots()
        self._traj_plots.show_key(key)
        frame = self._current_trajectory_frame()
        atoms = self._traj_run.frame_participants(frame, key)
        self._select_atoms(atoms)

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
        state = self._lifecycle.latest.get("trajectory")
        if state is None or self._lifecycle.active_run is not None:
            return
        if self._traj_live_cb.isChecked():
            self._activate_result(state)
            self._install_coordset_handler(state.payload.structure)
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
        owner = self._lifecycle.display_owner
        if (self._lifecycle.disposed or self._lifecycle.active_run is not None
                or owner is None or owner.mode != "trajectory"
                or self._suppress_coordset_sync or self._traj_run is None):
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
            self._traj_frame_label.setText("%d (not analysed)" % cs_id)
            self._redraw_frame_interactions()
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
            self._on_frame_slider(self._traj_slider.value())
        else:
            self._remove_coordset_handler()
            state = self._lifecycle.display_owner
            if state is not None and state.mode == "trajectory":
                self._close_graphics(state.graphics)
                state.graphics.clear()
                self._rows = []
                self._populate([])
                self._build_legend([])
                for button in (self._export_scope, self._export_interactions_btn, self._export_btn):
                    button.setEnabled(False)

    def _redraw_frame_interactions(self):
        state = self._lifecycle.display_owner
        if (self._lifecycle.disposed or self._lifecycle.active_run is not None
                or state is None or state.mode != "trajectory" or self._traj_run is None
                or not self._traj_live_cb.isChecked()):
            return
        frame_index = self._current_trajectory_frame()
        if frame_index is None:
            self._rows = []
            self._populate([])
            self._build_legend([])
            self._close_graphics(state.graphics)
            state.graphics.clear()
            return
        included = {kind for kind, on in self._traj_frame_type_included.items() if on}
        from .runner import draw_frame_interactions
        try:
            rows = draw_frame_interactions(self.session, self._traj_run, frame_index, included_types=included,
                selected_only=self._selected_cb.isChecked(),
                skip_biopolymer_internal=self._skip_bio_cb.isChecked())
        except Exception as error:
            self._traj_status.setText("Error drawing frame: %s" % error)
            return
        self._close_dialog("_preview_dialog")
        state.graphics[:] = [model for model in state.graphics if not getattr(model, "deleted", False)]
        self._lifecycle.register_graphics(state, getattr(self._traj_run, "graphics", ()))
        self._rows = rows
        self._ligand_report = self._traj_run.ligand_report
        self._populate(rows)
        self._build_legend(rows)
        self._reapply_pseudobond_visibility()
        for button in (self._export_scope, self._export_interactions_btn, self._export_btn):
            button.setEnabled(bool(rows))
        self._refresh_protonation_ui()

    def _export_trajectory(self):
        if self._traj_run is not None:
            self._write_native_result(self._traj_run.result, "chemur_trajectory")

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
        receptor, poses = self._pose_receptor.value, self._checked_poses()
        if receptor is None or not poses or receptor in poses:
            self._pose_status.setText("Choose a receptor and at least one different pose.")
            return
        token = self._begin_run("poses", [receptor, *poses])
        if token is None:
            return
        from . import pose_compare
        self._pose_status.setText("Analysing %d poses…" % len(poses))
        try:
            options = self._analysis_options(receptor)
            options.update(add_hydrogens=self._addh_cb.isChecked(), skip_biopolymer_internal=True)
            comparison = pose_compare.compute_pose_comparison(self.session, receptor, poses, options,
                context=self._pose_context.value(), progress=self._pose_progress, cancel_check=token.check_cancelled)
            token.check_cancelled()
            self._clear_pose_3d_groups()
            self._commit_run(token, comparison)
            self._pose_comparison = comparison
            self._pose_positions = self._capture_pose_positions(comparison)
            self._pose_show_btn.setEnabled(bool(comparison.rows))
            self._pose_export_btn.setEnabled(bool(comparison.native_runs))
            self._pose_status.setText("%d poses × %d residues; failed analyses excluded from consensus." % (len(comparison.rows), len(comparison.columns)))
            self._open_pose_dialog()
        except RunCancelled:
            if not self._lifecycle.disposed:
                self._pose_status.setText("Pose analysis cancelled.")
        except Exception as error:
            if not self._lifecycle.disposed:
                self.session.logger.report_exception()
                self._pose_status.setText("Error: %s" % error)
        finally:
            self._finish_run(token)

    def _pose_progress(self, done, total):
        token = self._lifecycle.active_run
        if token is None:
            raise RunCancelled("No active analysis")
        token.check_cancelled()
        self._pose_status.setText("Analysing poses… %d/%d" % (done, total))
        ui = getattr(self.session, "ui", None)
        if ui is not None and hasattr(ui, "processEvents"):
            ui.processEvents()
        token.check_cancelled()

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

    def _on_pose_cell_selected(self, pose, col_key):
        comparison = self._pose_comparison
        if comparison is None or self._lifecycle.disposed:
            return
        index = comparison.index_for_pose(pose)
        if index is None:
            return
        rows = comparison.cell_rows.get((index, col_key), [])
        self._select_atoms(atom for row in rows for ep in (row.endpoint1, row.endpoint2)
                           for atom in (getattr(ep, "center_atoms", None) or [ep.atom]))

    def _on_pose_show_3d(self, pose, show):
        from .runner import draw_pose_rows
        comparison = self._pose_comparison
        if comparison is None or self._lifecycle.disposed:
            return
        index = comparison.index_for_pose(pose)
        if index is None:
            return
        self._close_graphics(self._pose_3d_groups.pop(pose, ()))
        if not show:
            return
        rows = {id(row): row for (idx, _), contacts in comparison.cell_rows.items()
                if idx == index for row in contacts}
        graphics = []
        try:
            draw_pose_rows(self.session, list(rows.values()),
                           "Chemur pose %s (%x)" % (comparison.rows[index].name, id(self)), graphics=graphics)
            self._pose_3d_groups[pose] = graphics
            state = self._lifecycle.latest.get("poses")
            if state is not None:
                self._lifecycle.register_graphics(state, graphics, [comparison.receptor, pose])
        except Exception as error:
            self.session.logger.warning("Could not draw pose: %s" % error)

    def _clear_pose_3d_groups(self):
        graphics = [model for handles in self._pose_3d_groups.values() for model in handles]
        self._pose_3d_groups = {}
        self._close_graphics(graphics)

    # ----- protonation report ----------------------------------------------

    def _refresh_protonation_ui(self):
        """Sync the advanced-panel button, the main-window cue, and any open
        protonation dialog to the latest ``self._ligand_report``."""
        self._protonation_btn.setEnabled(bool(self._ligand_report))
        n_attention = sum(1 for r in self._ligand_report if r.status == "failed")
        if n_attention:
            self._protonation_cue.setText(
                '⚠ %d chemical product(s) need chemistry review — '
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
        state = self._lifecycle.display_owner
        if state is None or state.mode not in {"static", "trajectory"} or self._lifecycle.disposed:
            return
        source = state.source_models[0]
        self._source_overrides.setdefault(source, {}).update(overrides)
        self._model_button.value = source
        self._input_model_changed()
        if state.mode == "trajectory":
            self._analyze_trajectory()
        else:
            self._analyze()

    # ----- results tree ----------------------------------------------------

    def _populate(self, rows):
        self._view = ContactView(rows)
        self._row_haystacks = []
        self._rows_by_type = {}
        for idx, row in enumerate(rows):
            self._rows_by_type.setdefault(row.interaction_type, []).append(idx)
            self._row_haystacks.append(_row_haystack(row))
        self._apply_search_filter()

    def _row_check_state(self, idx, center=None):
        return _CHECKED if self._view.member_visible(idx, center) else _UNCHECKED

    def _build_residue_tree(self, indices):
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
        indices = list(indices)
        for idx in indices:
            row = self._rows[idx]
            if row.interaction_type == "hydrophobic_enclosure":
                continue
            leaf = QTreeWidgetItem(["%s: %s ⇄ %s" % (row.interaction_type, row.endpoint1.atom_label, row.endpoint2.atom_label),
                                   _fmt(row.distance), _fmt(row.angle), _fmt(row.offset)])
            leaf.setData(0, _USER_ROLE, idx)
            leaf.setToolTip(0, row.geometry_tooltip())
            leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            leaf.setCheckState(0, self._row_check_state(idx))
            self._tree.addTopLevelItem(leaf)
            self._row_items[idx] = [leaf]
        self._build_enclosure_tree(indices)

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
            if itype == "hydrophobic_enclosure":
                label = "%s (%d groups, %d contacts)" % (itype, len(self._view.groups), counts[itype])
            else:
                label = "%s (%d contacts)" % (itype, counts[itype])
            cb = QCheckBox(label)
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
        self._status.setText("%d interaction groups · %d/%d contacts in view" %
                             (self._view.native_count, len(self._row_items), len(self._rows)))

    # ----- visibility ------------------------------------------------------

    @staticmethod
    def _row_is_intrachain(row):
        return is_polymer_intrachain(row)

    def _on_interchain_toggled(self, checked):
        self._interchain_only = checked
        self._apply_search_filter()            # rebuild the (filtered) tree
        self._reapply_pseudobond_visibility()  # re-apply 3D visibility
        self._refresh_legend_states()

    def _reapply_pseudobond_visibility(self):
        for idx, row in enumerate(self._rows):
            pb = row.pseudobond
            if pb is not None and not getattr(pb, "deleted", False):
                pb.display = self._row_visible(idx)
        self._sync_tree_checks()
        self._refresh_legend_states()

    def _apply_row_visibility(self, idx, visible):
        if 0 <= idx < len(self._rows):
            self._view.set_visible([idx], visible)
            self._reapply_pseudobond_visibility()

    def _on_item_changed(self, item, column):
        if self._syncing or self._lifecycle.disposed:
            return
        data = item.data(0, _USER_ROLE)
        visible = item.checkState(0) == _CHECKED
        if isinstance(data, tuple) and data[0] == "group":
            self._view.set_group_visible(data[1], visible)
        elif isinstance(data, tuple) and data[0] == "leg":
            self._view.set_visible([data[1]], visible, data[2])
        else:
            indices = set()
            self._collect_row_indices(item, indices)
            self._view.set_visible(indices, visible)
        self._reapply_pseudobond_visibility()

    def _set_type_visible(self, itype, visible):
        if self._syncing:
            return
        self._view.set_visible(self._rows_by_type.get(itype, ()), visible)
        self._reapply_pseudobond_visibility()

    def _set_all_visible(self, visible):
        self._view.set_visible(range(len(self._rows)), visible)
        self._reapply_pseudobond_visibility()

    def _refresh_legend_states(self):
        for itype, cb in self._type_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(any(self._row_visible(i) for i in self._rows_by_type.get(itype, ())))
            cb.blockSignals(False)

    # ----- 3D selection + export ------------------------------------------

    def _tree_selection_changed(self):
        if self._lifecycle.disposed:
            return
        indices, atoms = set(), []
        def groups(item):
            data = item.data(0, _USER_ROLE)
            if isinstance(data, tuple) and data[0] == "group":
                meta = self._view.groups[data[1]]["metadata"]
                atoms.extend(self._active_id_to_atom.get(aid) for aid in [data[1], *meta.get("neighbor_atom_ids", ())])
            for i in range(item.childCount()):
                groups(item.child(i))
        for item in self._tree.selectedItems():
            self._collect_row_indices(item, indices)
            groups(item)
        for index in indices:
            if 0 <= index < len(self._rows):
                row = self._rows[index]
                atoms.extend(atom for ep in (row.endpoint1, row.endpoint2)
                             for atom in (getattr(ep, "center_atoms", None) or [ep.atom]))
        if atoms:
            self._select_atoms(atoms)

    def _collect_row_indices(self, item, indices, *, visible_only=False):
        data = item.data(0, _USER_ROLE)
        if isinstance(data, int):
            indices.add(data)
        elif isinstance(data, tuple) and data[0] == "leg":
            if not visible_only or self._view.member_visible(data[1], data[2]):
                indices.add(data[1])
        else:
            for i in range(item.childCount()):
                self._collect_row_indices(item.child(i), indices, visible_only=visible_only)

    def _export_diagram(self):
        state = self._lifecycle.display_owner
        if state is None or not self._rows:
            return
        structure = state.source_models[0]
        selected_residues = {a.residue for model in state.source_models for a in model.atoms if a.selected}
        if not selected_residues:
            self.session.logger.error("Select a ligand or residue first.")
            return
        from .diagram import build_scene
        try:
            scene = build_scene(self.session, structure, self._rows_for_scope(), selected_residues,
                                native_result=self._active_result, id_to_atom=self._active_id_to_atom)
        except Exception as error:
            self.session.logger.error("Diagram error: %s" % error)
            return
        self._close_dialog("_preview_dialog")
        self._preview_dialog = DiagramPreviewDialog(self.tool_window.ui_area, self.session, scene, structure.name)
        self._preview_dialog.show()
        self._preview_dialog.raise_()

    def _rows_for_scope(self):
        scope = self._export_scope.currentData()
        indices = {i for i in self._row_items if self._row_visible(i)}
        if scope == "rows":
            selected = set()
            for item in self._tree.selectedItems():
                self._collect_row_indices(item, selected, visible_only=True)
            indices &= selected
        elif scope == "atoms":
            indices = {i for i in indices
                       if any(atom.selected for ep in (self._rows[i].endpoint1, self._rows[i].endpoint2)
                              for atom in (getattr(ep, "center_atoms", None) or [ep.atom]))}
        return [self._rows[i] for i in sorted(indices)]

    def _export_interactions(self):
        rows = self._rows_for_scope()
        if not rows:
            self.session.logger.error("No displayed contacts in the chosen scope.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(self.tool_window.ui_area, "Export displayed contacts", "chemur_displayed", "JSON (*.json);;CSV (*.csv)")
        if not path:
            return
        from . import export
        try:
            path, kind = export.resolve_output_path(path, selected_filter)
            (export.write_csv if kind == "csv" else export.write_json)(rows, path)
        except Exception as error:
            self.session.logger.error("Export error: %s" % error)
            return
        self.session.logger.info("Chemur exported %d displayed contacts to %s" % (len(rows), path))

    def _close_dialog(self, name):
        dialog = getattr(self, name, None)
        setattr(self, name, None)
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()

    def _close_graphics(self, graphics):
        seen = set()
        for model in list(graphics):
            if model is None or id(model) in seen or getattr(model, "deleted", False):
                continue
            seen.add(id(model))
            if getattr(model, "id", None) is not None:
                self.session.models.close([model])
            else:
                model.delete()

    def _refresh_engine_info(self):
        import importlib
        try:
            from .engine_support import require_engine
            capabilities = require_engine()
            engine = importlib.import_module("chemur")
            core = importlib.import_module("chemur.core")
            backend = "C++" if core.USING_CPP_CORE else "Python"
            text = "Chemur %s · %s geometry · definitions %s" % (
                engine.__version__, backend, capabilities.geometry_revision)
            self._engine_info.setText(text)
            self._engine_info.setToolTip(str(engine.__file__))
        except Exception as error:
            self._engine_info.setText("Chemur update required")
            self._engine_info.setToolTip(str(error))

    def _input_model_changed(self, *_args):
        if self._lifecycle.disposed:
            return
        model = self._model_button.value
        self._ligand_smiles_overrides = self._source_overrides.setdefault(model, {}) if model is not None else {}
        path = self._assembly_paths.get(model)
        self._assembly_path_label.setText(path or "No assembly override")

    def _load_assembly_override(self):
        model = self._model_button.value
        if model is None:
            self.session.logger.error("Choose the structure whose chemistry will be corrected first.")
            return
        path, _ = QFileDialog.getOpenFileName(self.tool_window.ui_area, "Load Chemur assembly override", "", "JSON (*.json)")
        if not path:
            return
        try:
            from chemur.assemblies import load_assembly_overrides
            load_assembly_overrides(path)
        except Exception as error:
            self.session.logger.error("Invalid assembly override: %s" % error)
            return
        self._assembly_paths[model] = path
        self._input_model_changed()

    def _clear_assembly_override(self):
        self._assembly_paths.pop(self._model_button.value, None)
        self._input_model_changed()

    def _analysis_options(self, structure):
        return dict(
            protonate=self._protonate_cb.isChecked(), protonate_ph=self._ph_spin.value(),
            ligand_smiles=dict(self._source_overrides.get(structure, {})) or None,
            rule_overrides=self._collect_rule_overrides(),
            assembly_overrides=self._assembly_paths.get(structure),
            hydrophobic_enclosure=self._enclosure_cb.isChecked(),
        )

    def _begin_run(self, mode, source_models):
        if self._lifecycle.disposed or self._lifecycle.active_run is not None:
            return None
        try:
            from .engine_support import require_engine
            require_engine()
            token = self._lifecycle.begin_run(mode, source_models)
        except Exception as error:
            self.session.logger.error(str(error))
            return None
        self._remove_coordset_handler()
        for button in (self._analyze_btn, self._traj_analyze_btn, self._pose_btn):
            button.setEnabled(False)
        return token

    def _finish_run(self, token):
        self._lifecycle.finish_run(token)
        if self._lifecycle.disposed:
            return
        for button in (self._analyze_btn, self._traj_analyze_btn, self._pose_btn):
            button.setEnabled(True)
        owner = self._lifecycle.display_owner
        if owner is not None and owner.mode == "trajectory" and self._traj_live_cb.isChecked():
            self._install_coordset_handler(owner.payload.structure)

    def _commit_run(self, token, payload):
        previous = self._lifecycle.latest.get(token.mode)
        state = self._lifecycle.commit(token, payload)
        retained = {id(g) for current in self._lifecycle.latest.values() for g in current.graphics}
        if previous is not None:
            self._close_graphics([g for g in previous.graphics if id(g) not in retained])
        return state

    def _activate_result(self, state):
        if self._lifecycle.disposed:
            return
        if self._lifecycle.display_owner is not state:
            self._close_dialog("_preview_dialog")
            self._close_dialog("_protonation_dialog")
        for current in self._lifecycle.latest.values():
            if current.mode in ("static", "trajectory"):
                for model in current.graphics:
                    if not getattr(model, "deleted", False):
                        model.display = current is state
        self._lifecycle.activate(state)
        run = state.payload
        self._active_result = getattr(run, "result", None)
        self._active_id_to_atom = getattr(run, "id_to_atom", {})
        label = ", ".join("#%s %s" % (m.id_string, m.name) for m in state.source_models)
        self._owner_label.setText("Results: %s (%s)" % (label, state.mode))
        self._export_assigned_btn.setEnabled(self._active_result is not None)
        if state.mode != "trajectory":
            self._remove_coordset_handler()

    def _clear_display(self):
        self._rows = []
        self._ligand_report = []
        self._active_result = None
        self._active_id_to_atom = {}
        self._populate([])
        self._build_legend([])
        self._owner_label.setText("No analysis results")
        for button in (self._export_scope, self._export_interactions_btn, self._export_assigned_btn, self._export_btn):
            button.setEnabled(False)
        self._refresh_protonation_ui()
        self._close_dialog("_preview_dialog")
        self._close_dialog("_protonation_dialog")

    def _on_models_removed(self, _trigger, models):
        if self._lifecycle.disposed or self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        try:
            owner = self._lifecycle.display_owner
            removed = self._lifecycle.remove_models(models)
            for model in models:
                self._source_overrides.pop(model, None)
                self._assembly_paths.pop(model, None)
            self._close_graphics(removed.graphics)
            modes = {state.mode for state in removed.invalidated}
            if owner is not None and self._lifecycle.display_owner is None:
                self._clear_display()
            if "static" in modes:
                self._static_run = None
            if "trajectory" in modes:
                self._remove_coordset_handler()
                self._traj_run = None
                self._traj_table.clear()
                self._traj_item_keys.clear()
                self._traj_slider.setEnabled(False)
                self._traj_show_plots_btn.setEnabled(False)
                self._traj_export_btn.setEnabled(False)
                if self._traj_plots is not None:
                    self._traj_plots.set_result(None)
                self._close_dialog("_traj_plots_dialog")
                self._traj_plots = None
                self._traj_status.setText("Source structure closed.")
            if "poses" in modes:
                self._pose_comparison = None
                self._clear_pose_3d_groups()
                self._close_dialog("_pose_dialog")
                self._pose_show_btn.setEnabled(False)
                self._pose_export_btn.setEnabled(False)
                self._pose_positions = {}
                self._pose_status.setText("Comparison source closed.")
            for state in removed.updated:
                if state.mode == "poses":
                    self._pose_comparison = state.payload
                    live = {row.pose for row in state.payload.rows}
                    for pose in list(self._pose_3d_groups):
                        if pose not in live:
                            self._close_graphics(self._pose_3d_groups.pop(pose))
                    if self._pose_dialog is not None:
                        self._pose_dialog.set_comparison(state.payload)
                    self._pose_status.setText("%d remaining poses" % len(state.payload.rows))
            self._refresh_pose_list()
        finally:
            self._cleanup_in_progress = False

    def _capture_pose_positions(self, comparison):
        receptor = comparison.receptor
        if receptor is None:
            return {}
        inverse = receptor.scene_position.inverse()
        return {row.pose: tuple((inverse * row.pose.scene_position).matrix.ravel()) for row in comparison.rows}

    def _on_model_position_changed(self, _trigger, model):
        if self._lifecycle.disposed:
            return
        token = self._lifecycle.active_run
        if token is not None and token.mode == "poses" and any(model is source for source in token.source_models):
            token.cancelled = True
        comparison = self._pose_comparison
        if comparison is None:
            return
        try:
            positions = self._capture_pose_positions(comparison)
            import numpy as np
            changed = any(pose not in self._pose_positions or not np.allclose(pos, self._pose_positions[pose], atol=1e-7, rtol=0)
                          for pose, pos in positions.items())
        except (AttributeError, RuntimeError):
            changed = True
        if changed:
            removed = self._lifecycle.invalidate("poses")
            self._close_graphics(removed.graphics)
            self._pose_comparison = None
            self._clear_pose_3d_groups()
            self._close_dialog("_pose_dialog")
            self._pose_show_btn.setEnabled(False)
            self._pose_export_btn.setEnabled(False)
            self._pose_positions = {}
            self._pose_status.setText("Relative model positions changed; re-analyse poses.")

    def _build_nested(self, indices):
        indices = list(indices)
        self._build_residue_tree([i for i in indices if self._rows[i].interaction_type != "hydrophobic_enclosure"])
        self._build_enclosure_tree(indices)

    def _build_enclosure_tree(self, indices):
        allowed = set(indices)
        records = {atom.atom_id: atom for atom in getattr(self._active_result, "atoms", ())}
        for center, group in self._view.groups.items():
            children = [i for i in group["indices"] if i in allowed]
            if not children:
                continue
            metadata = group["metadata"]
            record = records.get(center)
            label = record.label if record is not None else "atom %s" % center
            neighbors = metadata.get("neighbor_atom_ids", ())
            maximum = max((float(leg["distance"]) for leg in metadata.get("legs", ())), default=None)
            parent = QTreeWidgetItem(["Hydrophobic enclosure: %s (%d neighbours)" % (label, len(neighbors)), _fmt(maximum), "", ""])
            _make_group(parent)
            parent.setData(0, _USER_ROLE, ("group", center))
            parent.setToolTip(0, "One enclosed centre; distance is the maximum constituent distance. Expand for individual contacts.")
            self._tree.addTopLevelItem(parent)
            for idx in children:
                row = self._rows[idx]
                child = QTreeWidgetItem(["%s ⇄ %s" % (row.endpoint1.atom_label, row.endpoint2.atom_label), _fmt(row.distance), "", ""])
                child.setData(0, _USER_ROLE, ("leg", idx, center))
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, self._row_check_state(idx, center))
                child.setToolTip(0, row.geometry_tooltip())
                parent.addChild(child)
                self._row_items.setdefault(idx, []).append(child)
        self._sync_tree_checks()

    def _sync_tree_checks(self):
        was = self._syncing
        self._syncing = True
        partial = Qt.CheckState.PartiallyChecked
        def sync(item):
            data = item.data(0, _USER_ROLE)
            child_states = [sync(item.child(i)) for i in range(item.childCount())]
            if isinstance(data, tuple) and data[0] == "leg":
                state = self._row_check_state(data[1], data[2])
            elif isinstance(data, int):
                state = self._row_check_state(data)
            elif child_states:
                state = child_states[0] if all(v == child_states[0] for v in child_states) else partial
            else:
                state = _UNCHECKED
            item.setCheckState(0, state)
            return state
        try:
            for index in range(self._tree.topLevelItemCount()):
                sync(self._tree.topLevelItem(index))
        finally:
            self._syncing = was

    def _row_visible(self, idx):
        row = self._rows[idx]
        return (self._view.row_visible(idx)
                and not (row.pseudobond is not None and getattr(row.pseudobond, "deleted", False))
                and not (self._interchain_only and self._row_is_intrachain(row))
                and all(not getattr(atom, "deleted", False) for ep in (row.endpoint1, row.endpoint2)
                        for atom in (getattr(ep, "center_atoms", None) or [ep.atom])))

    def _select_atoms(self, atoms):
        from chimerax.atomic import Atoms
        unique = {id(atom): atom for atom in atoms if atom is not None and not getattr(atom, "deleted", False)}
        self.session.selection.clear()
        if unique:
            Atoms(list(unique.values())).selected = True

    def _export_assigned(self):
        if self._active_result is None:
            return
        self._write_native_result(self._active_result, "chemur_assigned")

    def _write_native_result(self, result, default_name):
        path, selected_filter = QFileDialog.getSaveFileName(self.tool_window.ui_area, "Export assigned results", default_name, "JSON (*.json);;CSV (*.csv)")
        if not path:
            return
        from . import export
        try:
            path, kind = export.resolve_output_path(path, selected_filter)
            (export.write_native_csv if kind == "csv" else export.write_native_json)(result, path)
        except Exception as error:
            self.session.logger.error("Export error: %s" % error)
            return
        self.session.logger.info("Chemur exported complete assigned results to %s" % path)

    def _current_trajectory_frame(self):
        run = self._traj_run
        if run is None:
            return None
        try:
            pos = run.frame_coordset_ids.index(run.structure.active_coordset_id)
            return run.result.frame_indices[pos]
        except ValueError:
            return None

    def _export_pose_results(self):
        comparison = self._pose_comparison
        if comparison is None:
            return
        directory = QFileDialog.getExistingDirectory(self.tool_window.ui_area, "Export assigned pose JSON and CSV reports")
        if not directory:
            return
        from pathlib import Path
        from .export import write_native_json, write_native_csv
        import re
        for index, pose_row in enumerate(comparison.rows, 1):
            run = comparison.native_runs.get(pose_row.pose)
            if run is None:
                continue
            name = "%03d_%s" % (index, re.sub(r"[^A-Za-z0-9_.-]+", "_", pose_row.name).strip("_"))
            try:
                write_native_json(run.result, Path(directory) / (name + ".json"))
                write_native_csv(run.result, Path(directory) / (name + ".csv"))
            except Exception as error:
                self.session.logger.error("Pose export failed: %s" % error)
                return
        self.session.logger.info("Exported assigned pose reports to %s" % directory)


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

    _STYLE_OPTIONS = [("Colour", "default"), ("Black & white", "bw"),
                      ("Bold", "bold")]

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

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font size (pt):"))
        self._font_size = QSpinBox()
        self._font_size.setRange(6, 24)
        self._font_size.setValue(10)
        self._font_size.setToolTip("Size of all diagram text, including chemical atom labels and charges.")
        self._font_size.valueChanged.connect(self._view.set_font_size)
        font_row.addWidget(self._font_size)
        col.addLayout(font_row)

        col.addWidget(QLabel("Residues / ligand assemblies:"))
        self._unit_tree = QTreeWidget()
        self._unit_tree.setHeaderHidden(True)
        self._unit_tree.setMaximumHeight(150)
        self._units_by_id = {id(unit): unit for unit in self._scene.units}
        for unit in self._scene.units:
            item = QTreeWidgetItem([unit.label])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, _CHECKED)
            item.setData(0, _USER_ROLE, id(unit))
            item.setToolTip(0, "Show this complete residue or ligand assembly and its enabled contacts.")
            self._unit_tree.addTopLevelItem(item)
        self._unit_tree.itemChanged.connect(self._on_unit_item_changed)
        col.addWidget(self._unit_tree)

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

    def _on_unit_item_changed(self, item, column):
        unit = self._units_by_id.get(item.data(0, _USER_ROLE))
        if unit is not None:
            self._view.set_unit_visible(unit, item.checkState(0) == _CHECKED)

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
        path, selected_format = QFileDialog.getSaveFileName(
            self, "Save interaction diagram", "chemur_interactions.png",
            "PNG image (*.png);;SVG image (*.svg);;PDF document (*.pdf)")
        if not path:
            return
        try:
            from .export import resolve_output_path
            path, _ = resolve_output_path(path, selected_format, allowed=("png", "svg", "pdf"))
            self._view.render_to(path)
        except Exception:
            self._session.logger.report_exception()
            return
        self._session.logger.info("Chemur saved interaction diagram: %s" % path)


class LigandProtonationDialog(QDialog):
    """Validated or unresolved ligand and modified-polymer product chemistry.

    Standalone ligand SMILES can be edited by component identity. Connected and
    polymer products stay read-only and use complete assembly-override files.
    """

    _COLUMNS = ["Components / assembly", "Status", "Net charge", "Product SMILES", "Diagnostic"]

    def __init__(self, parent, rerun_callback):
        super().__init__(parent)
        self.setWindowTitle("Chemur chemical product report")
        self._rerun_callback = rerun_callback
        self._rows = []  # [(name, original_smiles, QLineEdit)]

        outer = QVBoxLayout(self)
        intro = QLabel(
            "Validated ligand and modified-polymer products. Edit standalone ligand "
            "SMILES by component identity; load assembly override JSON for complex, "
            "covalently attached or modified-polymer products.")
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
                "✗ unresolved" if failed else "✓ validated",
                "" if failed else _format_charge(entry.net_charge),
                "",  # SMILES lives in an editable widget (set below)
                entry.reason or "",
            ])
            self._tree.addTopLevelItem(item)
            item.setToolTip(0, "Assembly: %s\nComponents: %s" % (
                getattr(entry, "assembly_id", None) or entry.component_id,
                ", ".join(getattr(entry, "component_ids", ()) or [entry.component_id])))

            smiles = entry.smiles_used or entry.input_smiles or ""
            edit = QLineEdit(smiles)
            edit.setPlaceholderText("enter a SMILES for this residue")
            self._tree.setItemWidget(item, 3, edit)
            editable = getattr(entry, "editable", True)
            edit.setReadOnly(not editable)
            if not editable:
                edit.setToolTip("Complex, attached or modified-polymer product: use an assembly override JSON file.")
            self._rows.append((entry.component_id, smiles, edit))

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
            if not edit.isReadOnly() and text and text != original:
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
    "hydrophobic_enclosure": "hydrophobic enclosure enclosed surround surrounding",
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
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(0, _CHECKED)


def _swatch(interaction_type):
    r, g, b, a = rgba_255(interaction_type)
    pix = QPixmap(12, 12)
    pix.fill(QColor(r, g, b, a))
    return QIcon(pix)


def _fmt(value):
    return "" if value is None else "%.2f" % value

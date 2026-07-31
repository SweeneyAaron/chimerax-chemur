"""Small Qt widgets/layouts that let the docked tool work at ~300 px wide.

The tool is one tall column inside a widgetResizable QScrollArea. Qt never
shrinks that column below its minimumSizeHint(), so any child with a large
horizontal minimum forces a permanent horizontal scrollbar and pushes controls
off-screen -- the scroll area protects the *tool's* minimum, not the content's.
Everything here exists to keep horizontal minimums small.
"""

from __future__ import annotations

from Qt.QtCore import Qt, QPoint, QRect, QSize
from Qt.QtGui import QPainter
from Qt.QtWidgets import QLabel, QLayout, QScrollArea, QSizePolicy, QWidget

from .flow import flow_geometry


class ElidedLabel(QLabel):
    """Single-line label that elides overlong text instead of pinning layout width.

    A QLabel with word wrap off reports the full single-line text width from
    BOTH sizeHint() and minimumSizeHint(), so feeding one a file path or an
    exception string permanently widens the panel.

    Word wrap is not the fix for these: three of the four status labels live
    inside a CollapsiblePanel, and CollapsiblePanel.resize_panel() pins
    content_area's max height to QFrame.sizeHint().height(), which ignores
    heightForWidth. Expand a panel wide, then narrow the dock, and a wrapped
    label that now needs more lines is clipped with no way to recover.

    Elides in paintEvent rather than rewriting the text, so there is no
    setText/resizeEvent feedback loop and text() stays truthful.
    """

    #: Cap on the *preferred* width. sizeHint() feeds the enclosing scroll
    #: area's sizeHint, which is what ChimeraX reads to size a docked tool, so
    #: an unbounded status message would widen the panel on first show.
    MAX_HINT_WIDTH = 240

    def __init__(self, text="", parent=None, mode=Qt.TextElideMode.ElideMiddle):
        super().__init__(text, parent)
        self._elide_mode = mode
        self.setTextFormat(Qt.TextFormat.PlainText)
        # Ignored is the only horizontal policy that actually zeroes a widget's
        # width contribution: qSmartMinSize() skips the width term for Ignored,
        # while Preferred still uses minimumSizeHint().width().
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        if text:
            self.setToolTip(text)

    def setText(self, text):
        super().setText(text)
        # The elided form drops characters, so keep the full text reachable.
        self.setToolTip(text or "")

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(min(hint.width(), self.MAX_HINT_WIDTH), hint.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            rect = self.contentsRect()
            elided = self.fontMetrics().elidedText(
                self.text(), self._elide_mode, rect.width())
            alignment = self.alignment()
            # PyQt6 hands back an AlignmentFlag; drawText wants the raw int.
            flags = int(alignment.value) if hasattr(alignment, "value") else int(alignment)
            painter.drawText(rect, flags, elided)
        finally:
            painter.end()


class FlowLayout(QLayout):
    """Left-to-right layout that wraps onto a new line when it runs out of width.

    Qt ships this as an example but not as a class, and ChimeraX ships no
    equivalent. Used for rows of buttons/checkboxes that would otherwise set a
    ~800 px horizontal minimum on the whole tool.

    Items are placed at sizeHint() and never squashed, so controls stay
    readable; the row just gets taller. Qt 6's QWidget::hasHeightForWidth()
    delegates to the widget's layout, so a plain QWidget host propagates the
    reflow height up through the outer QVBoxLayout with no extra size-policy
    plumbing on the host.

    Children must carry WA_LayoutUsesWidgetRect (flow_row() sets it): on the
    macOS style a widget DRAWS larger than the layout-item rect it is handed --
    measured, a QPushButton given 108x20 renders 108x28, and a QCheckBox given
    106x12 renders 115x16. Sizing rows from item.sizeHint() without that
    attribute clips the overhang, because the row is exactly as tall as the
    layout rects it laid out.

    The wrap arithmetic lives in flow.py so it can be unit-tested without Qt.
    """

    def __init__(self, parent=None, margin=0, hspacing=6, vspacing=6):
        super().__init__(parent)
        self._items = []
        self._hspacing = hspacing
        self._vspacing = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    # ----- QLayout plumbing (all required overrides) -----------------------
    # Note: deliberately no __del__. The usual PyQt port adds one that drains
    # takeAt(), which can double-free items Qt's parent chain already reaped.

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        # Wrapping already consumes any extra width; don't ask for more.
        return Qt.Orientation(0)

    def spacing(self):
        return self._hspacing

    # ----- sizing ----------------------------------------------------------

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._geometry(width)[1]

    def setGeometry(self, rect):
        super().setGeometry(rect)
        rects, _height = self._geometry(rect.width())
        for item, (x, y, w, h) in zip(self._items, rects):
            item.setGeometry(QRect(QPoint(rect.x() + x, rect.y() + y), QSize(w, h)))

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        # sizeHint(), not minimumSize(), per item: this layout never squashes
        # items, so the row's floor must be the widest item. Using each item's
        # minimumSize() would let a row of Ignored-policy children claim it fits
        # in 20 px and then overflow its own bounds.
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.sizeHint())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _geometry(self, width):
        margins = self.contentsMargins()
        sizes = [(item.sizeHint().width(), item.sizeHint().height())
                 for item in self._items]
        return flow_geometry(
            sizes, width, self._hspacing, self._vspacing,
            (margins.left(), margins.top(), margins.right(), margins.bottom()))


class _FlowHost(QWidget):
    """Widget wrapper for a FlowLayout that reports its wrapped height.

    QWidget.sizeHint() would otherwise come from FlowLayout.sizeHint(), which is
    a single line's height. That matters because CollapsiblePanel.resize_panel()
    pins its content area to QFrame.sizeHint().height() -- a plain height query
    that never consults heightForWidth -- so a row wrapped onto three lines
    inside a collapsible section would be clipped to one.
    """

    def sizeHint(self):
        layout = self.layout()
        hint = layout.minimumSize()
        width = self.width()
        if width > 0:
            return QSize(hint.width(), layout.heightForWidth(width))
        return hint

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # sizeHint() depends on our width, and Qt caches sizeHint per widget, so
        # without this the row keeps reporting the height it needed at whatever
        # width it was first laid out at -- and gets clipped once it wraps.
        if event.oldSize().width() != event.size().width():
            self.updateGeometry()


def _exact_layout_size(widget):
    """Make `widget`'s layout-item rect equal the rect it actually draws in.

    The macOS style insets a control's layout rect relative to its widget rect,
    so a widget handed its own item.sizeHint() still overhangs -- measured, a
    QPushButton given 108x20 draws 108x28 (8 px below) and a QCheckBox given
    106x12 draws 115x16 (9 px right). Qt's own box layouts absorb that in their
    style-derived margins; FlowLayout computes exact geometry, so it needs the
    two rects to agree. WA_LayoutUsesWidgetRect is Qt's documented opt-out; it
    is a no-op under Fusion, where the rects already match.
    """
    widget.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect, True)
    # An Ignored horizontal policy makes QWidgetItem::sizeHint() report width 0,
    # and FlowLayout places items at exactly that -- the widget vanishes. This is
    # a real bug we shipped once (the export-scope combo), so normalise rather
    # than trust the caller. Widgets that genuinely need Ignored (ElidedLabel,
    # ModelMenuButton) belong in a QHBoxLayout with a stretch factor, not here.
    policy = widget.sizePolicy()
    if policy.horizontalPolicy() == QSizePolicy.Policy.Ignored:
        policy.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
        widget.setSizePolicy(policy)
    return widget


def flow_row(*widgets, margin=0, hspacing=6, vspacing=6):
    """A QWidget hosting `widgets` in a FlowLayout, ready for addWidget().

    Children are normalised by _exact_layout_size(): never pass a widget that
    relies on an Ignored horizontal size policy (see that function).
    """
    host = _exact_layout_size(_FlowHost())
    layout = FlowLayout(host, margin=margin, hspacing=hspacing, vspacing=vspacing)
    for widget in widgets:
        layout.addWidget(_exact_layout_size(widget))
    # QWidgetItem::hasHeightForWidth() reads the size POLICY, not
    # QWidget::hasHeightForWidth(), so the enclosing QVBoxLayout only honours the
    # reflow height if the flag is set here explicitly. Fixed vertically because
    # heightForWidth is exact: anything growable lets the row soak up spare
    # vertical space and render a one-line row three lines tall.
    policy = host.sizePolicy()
    policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
    policy.setHeightForWidth(True)
    host.setSizePolicy(policy)
    return host


class ResizeNotifyWidget(QWidget):
    """QWidget that invokes a callback after every resize."""

    def __init__(self, on_resize, parent=None):
        super().__init__(parent)
        self._on_resize = on_resize

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._on_resize()


def repin_collapsible(panel):
    """Re-pin an expanded CollapsiblePanel to the height its content needs *now*.

    CollapsiblePanel.resize_panel() pins content_area to
    QFrame.sizeHint().height(), which routes through QLayout::totalSizeHint():
    that evaluates heightForWidth at the *layout's* preferred width, not the
    width the panel actually has. For a word-wrapped QLabel the preferred width
    is Qt's ~2000 px "natural" width, so the panel is pinned to a one- or
    two-line height and clips as soon as the dock is narrowed. (For a flow row
    it errs the other way and reserves the worst-case height forever.)

    Evaluating heightForWidth at the real width fixes both directions. No-op on
    a collapsed panel, so it can be called unconditionally on resize.
    """
    if not panel.shown:
        return
    content = panel.content_area
    layout = content.layout()
    if layout is None:
        return
    width = content.width()
    if width <= 0:
        return
    if layout.hasHeightForWidth():
        height = layout.totalHeightForWidth(width)
    else:
        height = layout.totalSizeHint().height()
    if height != content.maximumHeight():
        content.setMaximumHeight(height)
        content.setMinimumHeight(height)


def labelled(text, widget, spacing=4):
    """A label+control pair that flows as one unit inside a FlowLayout.

    Without this a FlowLayout would happily wrap between "Start" and its spin
    box, leaving orphaned labels at the end of a line.
    """
    from Qt.QtWidgets import QHBoxLayout

    host = _exact_layout_size(QWidget())
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    # Same exact-geometry requirement as flow_row's own children: this host is
    # measured by a FlowLayout, so its contents must not overhang either.
    row.addWidget(_exact_layout_size(QLabel(text)))
    row.addWidget(_exact_layout_size(widget))
    return host


class NarrowScrollArea(QScrollArea):
    """QScrollArea that asks for a modest width.

    QScrollArea.sizeHint() is its content's sizeHint bounded to
    36 x fontMetrics().height() (~540 px here). ChimeraX reads
    ui_area.sizeHint().width() when placing a docked tool
    (chimerax/ui/gui.py:2323-2332), so without this the panel opens ~540-650 px
    wide even once its minimum is small.
    """

    PREFERRED_WIDTH = 340

    def sizeHint(self):
        return QSize(self.PREFERRED_WIDTH, super().sizeHint().height())

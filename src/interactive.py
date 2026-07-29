"""Interactive Qt canvas for the 2D interaction diagram.

Hosts a :class:`~.diagram.DiagramScene` in a ``QGraphicsView``: each
residue/ligand is a draggable item and its dashed interaction lines + distance
labels re-route live (via :func:`~.diagram._route_edge`) as it moves. The view
also exposes per-type / per-interaction visibility, per-type colour overrides,
and a switchable chemical-structure style. Exports the *current* (hand-edited)
view to PNG/SVG/PDF.
"""

from __future__ import annotations

import os

from Qt.QtCore import Qt, QPointF, QRectF, QSize
from Qt.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from Qt.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView,
)

from .colors import rgba_255
from .diagram import _endpoint_pos, _obstacles_for_edge, _route_edge

_FALLBACK_FACE = QColor("#e0e0e0")
_FALLBACK_EDGE = QColor("#666666")
_LABEL_COLOR = QColor("#444444")

_FLAG = QGraphicsItem.GraphicsItemFlag
_CHANGE = QGraphicsItem.GraphicsItemChange


def _qcolor(interaction_type):
    r, g, b, a = rgba_255(interaction_type)
    return QColor(r, g, b, a)


class _EdgeRec:
    """One interaction's scene items + the model edge it draws."""

    __slots__ = ("edge", "path", "label")

    def __init__(self, edge, path, label):
        self.edge = edge
        self.path = path
        self.label = label


class _PixmapUnitItem(QGraphicsPixmapItem):
    """Draggable rendered structure for one unit."""

    def __init__(self, unit, view):
        super().__init__(_pixmap_from(unit.png))
        self._unit = unit
        self._view = view
        self.label_item = None
        self.setPos(float(unit.global_origin[0]), float(unit.global_origin[1]))
        _configure_movable(self)

    def refresh_pixmap(self):
        """Swap in the unit's current PNG (after a structure restyle)."""
        self.setPixmap(_pixmap_from(self._unit.png))
        self.reposition_label()

    def reposition_label(self):
        if self.label_item is None:
            return
        br = self.boundingRect()
        tb = self.label_item.boundingRect()
        self.label_item.setPos(br.center().x() - tb.width() / 2.0,
                               br.top() - tb.height() - 2.0)

    def itemChange(self, change, value):
        if change == _CHANGE.ItemPositionHasChanged:
            self._view.on_unit_moved(self._unit)
        return super().itemChange(change, value)


class _FallbackUnitItem(QGraphicsEllipseItem):
    """Draggable placeholder for a unit whose structure couldn't be drawn."""

    def __init__(self, unit, view):
        super().__init__(-30.0, -30.0, 60.0, 60.0)
        self._unit = unit
        self._view = view
        self.label_item = None
        self.setBrush(_FALLBACK_FACE)
        self.setPen(QPen(_FALLBACK_EDGE, 1.5))
        self.setPos(float(unit.global_origin[0]), float(unit.global_origin[1]))
        _configure_movable(self)

    def itemChange(self, change, value):
        if change == _CHANGE.ItemPositionHasChanged:
            self._view.on_unit_moved(self._unit)
        return super().itemChange(change, value)


def _pixmap_from(png_bytes):
    pm = QPixmap()
    if png_bytes:
        pm.loadFromData(png_bytes)
    return pm


def _configure_movable(item):
    item.setFlag(_FLAG.ItemIsMovable, True)
    item.setFlag(_FLAG.ItemIsSelectable, True)
    item.setFlag(_FLAG.ItemSendsGeometryChanges, True)
    item.setCursor(Qt.CursorShape.OpenHandCursor)
    item.setZValue(2.0)


class InteractiveDiagramView(QGraphicsView):
    """Editable 2D interaction diagram. Drag a unit; its lines follow."""

    def __init__(self, scene_model, parent=None):
        super().__init__(parent)
        import numpy as np
        self._np = np
        self._model = scene_model
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QColor("white"))
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("white"))

        # Live atom positions (atom -> np.array); rebuilt per-unit on drag/restyle.
        self._atom_pos = {a: np.array(p, dtype=float) for a, p in scene_model.atom_pos.items()}
        self._atom_unit = scene_model.atom_unit
        self._unit_items = {}                 # id(unit) -> item
        self._edges_by_unit = {}              # id(unit) -> [_EdgeRec]
        self._edge_recs = []                  # [_EdgeRec]
        self._rec_by_edge = {}                # id(edge) -> _EdgeRec
        self._recs_by_type = {}               # type -> [_EdgeRec]

        # User-controllable state.
        self._type_visible = {}               # type -> bool (default True)
        self._edge_visible = {}               # id(edge) -> bool (default True)
        self._type_color = {}                 # type -> QColor override
        self._style = "default"               # current structure style
        self._legend_on = True
        self._legend_group = None
        self._fitted = False

        self._build()

    # ----- construction ----------------------------------------------------

    def _build(self):
        label_edges = self._model.show_distances and len(self._model.edges) <= 60
        for unit in self._model.units:
            item = (_PixmapUnitItem(unit, self) if (unit.ok and unit.png)
                    else _FallbackUnitItem(unit, self))
            self._scene.addItem(item)
            self._unit_items[id(unit)] = item
            self._add_unit_label(unit, item)

        for edge in self._model.edges:
            path_item = QGraphicsPathItem()
            pen = QPen(QColor("black"), 1.8)
            pen.setStyle(Qt.PenStyle.DashLine)
            path_item.setPen(pen)
            path_item.setZValue(3.0)
            self._scene.addItem(path_item)

            label_item = None
            if label_edges and edge.distance is not None:
                label_item = QGraphicsSimpleTextItem("%.1f" % edge.distance)
                label_item.setPen(QPen(QColor("white"), 0.6))
                label_item.setZValue(4.0)
                self._scene.addItem(label_item)

            rec = _EdgeRec(edge, path_item, label_item)
            self._edge_recs.append(rec)
            self._rec_by_edge[id(edge)] = rec
            self._recs_by_type.setdefault(edge.interaction_type, []).append(rec)
            self._edges_by_unit.setdefault(id(edge.unitA), []).append(rec)
            if id(edge.unitB) != id(edge.unitA):
                self._edges_by_unit.setdefault(id(edge.unitB), []).append(rec)
            self._apply_edge_style(rec)
            self._reroute(rec)

        self._rebuild_legend()

    def _add_unit_label(self, unit, item):
        text = QGraphicsSimpleTextItem(unit.label, item)
        text.setBrush(_LABEL_COLOR)
        text.setZValue(5.0)
        item.label_item = text
        br = item.boundingRect()
        tb = text.boundingRect()
        # Centred just above the structure, in the item's local coordinates.
        text.setPos(br.center().x() - tb.width() / 2.0, br.top() - tb.height() - 2.0)

    # ----- styling / visibility (public API for the dialog) ----------------

    def types_present(self):
        """Interaction types in this diagram, sorted."""
        return sorted(self._recs_by_type)

    def color_for_type(self, interaction_type):
        return self._color_for(interaction_type)

    def set_type_visible(self, interaction_type, visible):
        self._type_visible[interaction_type] = bool(visible)
        for rec in self._recs_by_type.get(interaction_type, ()):
            self._apply_edge_style(rec)
        self._rebuild_legend()

    def set_edge_visible(self, edge, visible):
        self._edge_visible[id(edge)] = bool(visible)
        rec = self._rec_by_edge.get(id(edge))
        if rec is not None:
            self._apply_edge_style(rec)

    def set_type_color(self, interaction_type, qcolor):
        self._type_color[interaction_type] = QColor(qcolor)
        for rec in self._recs_by_type.get(interaction_type, ()):
            self._apply_edge_style(rec)
        self._rebuild_legend()

    def set_legend_visible(self, on):
        self._legend_on = bool(on)
        self._rebuild_legend()

    def set_structure_style(self, style):
        """Re-render every structure in ``style``, keeping each one in place."""
        from . import diagram
        if style == self._style:
            return
        self._style = style
        np = self._np
        for unit in self._model.units:
            item = self._unit_items.get(id(unit))
            if not (unit.ok and unit.mol) or not isinstance(item, _PixmapUnitItem):
                continue
            old = unit.centroid_px
            centroid_scene = (item.pos().x() + float(old[0]),
                              item.pos().y() + float(old[1]))
            diagram.restyle_unit(unit, style)          # updates png/img/atom_px/centroid_px
            item.refresh_pixmap()
            new = unit.centroid_px
            item.setPos(centroid_scene[0] - float(new[0]),
                        centroid_scene[1] - float(new[1]))
        # Rebuild atom positions from the new pixels, then reroute everything.
        self._atom_pos = {}
        for unit in self._model.units:
            item = self._unit_items.get(id(unit))
            if item is None:
                continue
            ox, oy = item.pos().x(), item.pos().y()
            for atom, px in unit.atom_px.items():
                self._atom_pos[atom] = np.array([ox + float(px[0]), oy + float(px[1])])
        for rec in self._edge_recs:
            self._reroute(rec)
        self._rebuild_legend()

    def _color_for(self, interaction_type):
        return self._type_color.get(interaction_type) or _qcolor(interaction_type)

    def _apply_edge_style(self, rec):
        col = self._color_for(rec.edge.interaction_type)
        pen = rec.path.pen()
        pen.setColor(col)
        rec.path.setPen(pen)
        if rec.label is not None:
            rec.label.setBrush(col)
        visible = (self._type_visible.get(rec.edge.interaction_type, True)
                   and self._edge_visible.get(id(rec.edge), True))
        rec.path.setVisible(visible)
        if rec.label is not None:
            rec.label.setVisible(visible)

    # ----- in-scene legend -------------------------------------------------

    def _rebuild_legend(self):
        if self._legend_group is not None:
            self._scene.removeItem(self._legend_group)
            self._legend_group = None
        if not self._legend_on:
            return
        types = [t for t in self.types_present() if self._type_visible.get(t, True)]
        if not types:
            return
        group = QGraphicsItemGroup()
        group.setZValue(10.0)
        y = 0.0
        for t in types:
            swatch = QGraphicsRectItem(0.0, y, 12.0, 12.0)
            swatch.setBrush(self._color_for(t))
            swatch.setPen(QPen(QColor("#888888"), 0.5))
            group.addToGroup(swatch)
            text = QGraphicsSimpleTextItem(t)
            text.setBrush(QColor("#222222"))
            text.setPos(16.0, y - 1.0)
            group.addToGroup(text)
            y += 16.0
        self._scene.addItem(group)
        self._legend_group = group
        self._position_legend()

    def _position_legend(self):
        if self._legend_group is None:
            return
        rect = self._content_rect()
        lb = self._legend_group.childrenBoundingRect()
        self._legend_group.setPos(rect.left(), rect.top() - lb.height() - 12.0)

    def _content_rect(self):
        rect = None
        for item in self._unit_items.values():
            b = item.sceneBoundingRect()
            rect = b if rect is None else rect.united(b)
        for rec in self._edge_recs:
            if rec.path.isVisible():
                b = rec.path.sceneBoundingRect()
                rect = b if rect is None else rect.united(b)
        return rect if rect is not None else QRectF(0.0, 0.0, 1.0, 1.0)

    # ----- live re-routing -------------------------------------------------

    def on_unit_moved(self, unit):
        np = self._np
        item = self._unit_items.get(id(unit))
        if item is None:
            return
        ox, oy = item.pos().x(), item.pos().y()
        for atom, px in unit.atom_px.items():
            self._atom_pos[atom] = np.array([ox + float(px[0]), oy + float(px[1])])
        for rec in self._edges_by_unit.get(id(unit), ()):  # only this unit's edges
            self._reroute(rec)

    def _reroute(self, rec):
        np = self._np
        pA = _endpoint_pos(rec.edge.atomA, rec.edge.ringA, self._atom_pos, np)
        pB = _endpoint_pos(rec.edge.atomB, rec.edge.ringB, self._atom_pos, np)
        if pA is None or pB is None:
            return
        obs_pos, obs_w = _obstacles_for_edge(rec.edge, self._atom_pos, self._atom_unit, np)
        pts = _route_edge(pA, pB, obs_pos, obs_w, np)
        path = QPainterPath(QPointF(float(pts[0][0]), float(pts[0][1])))
        for p in pts[1:]:
            path.lineTo(QPointF(float(p[0]), float(p[1])))
        rec.path.setPath(path)
        if rec.label is not None:
            m = pts[len(pts) // 2]
            br = rec.label.boundingRect()
            rec.label.setPos(float(m[0]) - br.width() / 2.0,
                             float(m[1]) - br.height() / 2.0)

    # ----- view behaviour --------------------------------------------------

    def reset_layout(self):
        """Restore the automatic (non-crossing) arrangement."""
        for unit in self._model.units:
            item = self._unit_items.get(id(unit))
            if item is not None:
                item.setPos(float(unit.global_origin[0]), float(unit.global_origin[1]))
        self._position_legend()
        self.fit()

    def fit(self):
        rect = self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            self.fit()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    # ----- export ----------------------------------------------------------

    def render_to(self, path, *, dpi=300):
        """Render the current scene (with any manual edits) to ``path``."""
        self._position_legend()
        rect = self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".svg":
            self._render_svg(path, rect)
        elif ext == ".pdf":
            self._render_pdf(path, rect, dpi)
        else:
            self._render_raster(path, rect, dpi)

    def _render_raster(self, path, rect, dpi):
        scale = max(1.0, dpi / 96.0)
        w = max(1, int(rect.width() * scale))
        h = max(1, int(rect.height() * scale))
        image = QImage(w, h, QImage.Format.Format_ARGB32)
        image.fill(QColor("white"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self._scene.render(painter, QRectF(0, 0, w, h), rect)
        painter.end()
        image.save(path)

    def _render_svg(self, path, rect):
        try:                                      # ChimeraX's Qt shim omits QtSvg
            from Qt.QtSvg import QSvgGenerator
        except ImportError:
            from PyQt6.QtSvg import QSvgGenerator
        gen = QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(QSize(int(rect.width()), int(rect.height())))
        gen.setViewBox(QRectF(0, 0, rect.width(), rect.height()))
        painter = QPainter(gen)
        self._scene.render(painter, QRectF(0, 0, rect.width(), rect.height()), rect)
        painter.end()

    def _render_pdf(self, path, rect, dpi):
        from Qt.QtCore import QSizeF
        from Qt.QtGui import QPdfWriter, QPageSize
        writer = QPdfWriter(path)
        writer.setResolution(dpi)
        # Page sized to the diagram (points = px / 96 * 72).
        pts = QSizeF(max(1.0, rect.width() * 72.0 / 96.0),
                     max(1.0, rect.height() * 72.0 / 96.0))
        writer.setPageSize(QPageSize(pts, QPageSize.Unit.Point))
        painter = QPainter(writer)
        target = QRectF(0, 0, writer.width(), writer.height())
        self._scene.render(painter, target, rect)
        painter.end()

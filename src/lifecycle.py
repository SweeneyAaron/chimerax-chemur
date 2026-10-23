"""Pure result ownership and cooperative cancellation; no Qt/ChimeraX imports."""

from __future__ import annotations

from dataclasses import dataclass, field


class RunCancelled(Exception):
    """The tool or one of a computation's source models is no longer available."""


def _intersects(first, second):
    return any(a is b for a in first for b in second)


def _graphics_list(graphics):
    if graphics is None:
        return []
    return list(graphics) if isinstance(graphics, (list, tuple)) else [graphics]


@dataclass(eq=False)
class RunToken:
    mode: str
    generation: int
    source_models: tuple
    cancelled: bool = False
    reason: str = "Analysis cancelled"

    def cancel(self, reason="Analysis cancelled"):
        self.cancelled = True
        self.reason = reason

    def check_cancelled(self):
        if not self.cancelled and any(getattr(model, "deleted", False) for model in self.source_models):
            self.cancel("A source model was closed")
        if self.cancelled:
            raise RunCancelled(self.reason)

    cancel_check = check_cancelled


@dataclass(eq=False)
class ResultState:
    mode: str
    generation: int
    payload: object
    source_models: tuple
    graphics: list = field(default_factory=list)
    _graphics_sources: dict = field(default_factory=dict, repr=False)


@dataclass
class RemovalResult:
    invalidated: tuple = ()
    updated: tuple = ()
    graphics: tuple = ()
    cancelled: bool = False


class ResultLifecycle:
    """One latest result per mode and one active computation for a tool instance.

    The caller removes returned graphic handles on the GUI thread. Source
    identity is deliberately independent of mutable ChimeraX model numbers.
    """

    def __init__(self):
        self.latest = {}
        self.active_run = None
        self.display_owner = None
        self.disposed = False
        self._generation = 0

    def _next_generation(self):
        self._generation += 1
        return self._generation

    def begin_run(self, mode, source_models):
        if self.disposed:
            raise RunCancelled("The Chemur tool was closed")
        if self.active_run is not None:
            raise RuntimeError("A Chemur analysis is already running")
        token = RunToken(mode, self._next_generation(), tuple(source_models))
        token.check_cancelled()
        self.active_run = token
        return token

    def commit(self, token, payload, source_models=None):
        token.check_cancelled()
        if self.disposed or self.active_run is not token:
            raise RunCancelled("This analysis is no longer active")
        sources = source_models
        if sources is None:
            sources = getattr(payload, "source_models", None) or token.source_models
        state = ResultState(token.mode, self._next_generation(), payload, tuple(sources))
        self.register_graphics(state, getattr(payload, "graphics", None))
        old = self.latest.get(token.mode)
        if self.display_owner is old:
            self.display_owner = None
        self.latest[token.mode] = state
        return state

    def finish_run(self, token):
        if self.active_run is token:
            self.active_run = None
        token.cancel("This analysis has finished")

    def is_current(self, state):
        return not self.disposed and state is not None and self.latest.get(state.mode) is state

    def activate(self, state):
        if not self.is_current(state):
            raise RunCancelled("The displayed result is no longer available")
        self.display_owner = state

    def register_graphics(self, state, graphics, source_models=None):
        sources = tuple(source_models) if source_models is not None else state.source_models
        for handle in _graphics_list(graphics):
            if not any(handle is old for old in state.graphics):
                state.graphics.append(handle)
            state._graphics_sources[id(handle)] = sources

    def invalidate(self, mode):
        cancelled = False
        if self.active_run is not None and self.active_run.mode == mode:
            self.active_run.cancel("The analysis source changed")
            cancelled = True
        state = self.latest.pop(mode, None)
        if state is None:
            return RemovalResult(cancelled=cancelled)
        if self.display_owner is state:
            self.display_owner = None
        graphics = tuple(state.graphics)
        state.graphics.clear()
        state._graphics_sources.clear()
        return RemovalResult(invalidated=(state,), graphics=graphics, cancelled=cancelled)

    def remove_models(self, models):
        removed = tuple(models)
        invalidated, updated, graphics = [], [], []
        cancelled = False
        token = self.active_run
        if token is not None and _intersects(token.source_models, removed):
            token.cancel("A source model was closed")
            cancelled = True
        for mode, state in tuple(self.latest.items()):
            # A user can close a group itself without invalidating its data.
            for handle in state.graphics:
                if _intersects((handle,), removed):
                    state._graphics_sources.pop(id(handle), None)
            state.graphics[:] = [handle for handle in state.graphics
                                 if not _intersects((handle,), removed)]
            if not _intersects(state.source_models, removed):
                continue
            prune = getattr(state.payload, "prune_models", None)
            keep = bool(prune(removed)) if prune is not None else False
            if keep:
                retained = []
                for handle in state.graphics:
                    sources = state._graphics_sources.get(id(handle), state.source_models)
                    if _intersects(sources, removed):
                        graphics.append(handle)
                        state._graphics_sources.pop(id(handle), None)
                    else:
                        retained.append(handle)
                state.graphics[:] = retained
                state.source_models = tuple(state.payload.source_models)
                state.generation = self._next_generation()
                updated.append(state)
            else:
                self.latest.pop(mode, None)
                if self.display_owner is state:
                    self.display_owner = None
                graphics.extend(state.graphics)
                state.graphics.clear()
                state._graphics_sources.clear()
                invalidated.append(state)
        return RemovalResult(tuple(invalidated), tuple(updated), tuple(graphics), cancelled)

    def dispose(self):
        if self.disposed:
            return ()
        self.disposed = True
        if self.active_run is not None:
            self.active_run.cancel("The Chemur tool was closed")
        self.active_run = None
        graphics = []
        for state in self.latest.values():
            for handle in state.graphics:
                if not any(handle is old for old in graphics):
                    graphics.append(handle)
            state.graphics.clear()
            state._graphics_sources.clear()
        self.latest.clear()
        self.display_owner = None
        return tuple(graphics)


def track_graphics(session, source_models, graphics):
    """Close command-created graphics with their source models, by identity.

    ``graphics`` is the runner's mutable list, so redraws can replace its handles
    without installing another trigger. Imports remain deferred for pure tests.
    """
    if not graphics:
        return None
    trackers = getattr(session, "_chemur_graphics_trackers", None)
    if trackers is None:
        trackers = session._chemur_graphics_trackers = []
    for tracker in trackers:
        if tracker.graphics is graphics:
            tracker.source_models = tuple(source_models)
            return tracker
    tracker = _GraphicsTracker(session, tuple(source_models), graphics)
    trackers.append(tracker)
    return tracker


class _GraphicsTracker:
    def __init__(self, session, source_models, graphics):
        from chimerax.core.models import REMOVE_MODELS
        self.session = session
        self.source_models = source_models
        self.graphics = graphics
        self.handler = session.triggers.add_handler(REMOVE_MODELS, self._removed)

    def _removed(self, _name, models):
        if _intersects(self.source_models, models):
            self.close()
        else:
            self.graphics[:] = [handle for handle in self.graphics
                                if not _intersects((handle,), models)]
            if not self.graphics:
                self.close()

    def close(self):
        if self.handler is None:
            return
        self.handler.remove()
        self.handler = None
        trackers = getattr(self.session, "_chemur_graphics_trackers", [])
        if self in trackers:
            trackers.remove(self)
        graphics = list(self.graphics)
        self.graphics.clear()
        for handle in graphics:
            if getattr(handle, "deleted", False):
                continue
            if getattr(handle, "id", None) is None:
                handle.delete()
            else:
                self.session.models.close([handle])

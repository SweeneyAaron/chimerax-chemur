"""Model identity, selective ownership and in-flight cancellation without Qt."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "chemur_lifecycle_test", Path(__file__).resolve().parents[1] / "src" / "lifecycle.py")
lc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lc
spec.loader.exec_module(lc)


class Model:
    deleted = False

    def __init__(self, number=1):
        self.id = (number,)


def _commit(lifecycle, mode, model, graphics=()):
    token = lifecycle.begin_run(mode, (model,))
    payload = SimpleNamespace(source_models=(model,), graphics=list(graphics))
    state = lifecycle.commit(token, payload)
    lifecycle.finish_run(token)
    return state


def test_close_matches_identity_before_deleted_and_retains_other_mode():
    state = lc.ResultLifecycle()
    first, other, recycled_id = Model(1), Model(2), Model(1)
    graphic = object()
    static = _commit(state, "static", first, [graphic])
    trajectory = _commit(state, "trajectory", other)
    state.activate(trajectory)
    assert not state.remove_models([recycled_id]).invalidated
    removed = state.remove_models([first])
    assert first.deleted is False
    assert removed.invalidated == (static,) and removed.graphics == (graphic,)
    assert state.latest == {"trajectory": trajectory}
    assert state.display_owner is trajectory
    assert not state.remove_models([first]).invalidated


def test_only_one_run_and_source_close_cancels_stale_publication():
    state, source, unrelated = lc.ResultLifecycle(), Model(), Model()
    token = state.begin_run("static", [source])
    with pytest.raises(RuntimeError):
        state.begin_run("poses", [unrelated])
    state.remove_models([unrelated])
    token.check_cancelled()
    assert state.remove_models([source]).cancelled
    with pytest.raises(lc.RunCancelled):
        state.commit(token, object())
    state.finish_run(token)
    assert state.active_run is None and not state.latest


def test_failure_keeps_previous_result_and_disposal_invalidates_callbacks():
    state, source = lc.ResultLifecycle(), Model()
    graphic = object()
    previous = _commit(state, "static", source, [graphic])
    state.activate(previous)
    token = state.begin_run("static", [source])
    state.finish_run(token)
    assert state.latest["static"] is previous and state.display_owner is previous
    with pytest.raises(lc.RunCancelled):
        token.check_cancelled()
    late = state.begin_run("trajectory", [source])
    assert state.dispose() == (graphic,)
    with pytest.raises(lc.RunCancelled):
        late.check_cancelled()
    assert state.dispose() == ()
    assert not state.latest and state.display_owner is None


def test_mode_invalidation_does_not_clear_other_results():
    state = lc.ResultLifecycle()
    static = _commit(state, "static", Model())
    pose = _commit(state, "poses", Model(), ["pose-graphic"])
    state.activate(pose)
    token = state.begin_run("poses", pose.source_models)
    removed = state.invalidate("poses")
    assert removed.invalidated == (pose,) and removed.graphics == ("pose-graphic",)
    assert removed.cancelled and token.cancelled
    assert state.latest == {"static": static} and state.display_owner is None


def test_partial_payload_pruning_keeps_unaffected_graphics():
    receptor, first, second = Model(), Model(), Model()
    class Comparison:
        source_models = (receptor, first, second)
        graphics = []
        def prune_models(self, removed):
            self.source_models = tuple(model for model in self.source_models
                                       if not any(model is r for r in removed))
            return True
    lifecycle = lc.ResultLifecycle()
    token = lifecycle.begin_run("poses", Comparison.source_models)
    state = lifecycle.commit(token, Comparison())
    lifecycle.finish_run(token)
    first_graphic, second_graphic = object(), object()
    lifecycle.register_graphics(state, [first_graphic], (receptor, first))
    lifecycle.register_graphics(state, [second_graphic], (receptor, second))
    generation = state.generation
    result = lifecycle.remove_models([first])
    assert result.updated == (state,) and not result.invalidated
    assert result.graphics == (first_graphic,) and state.graphics == [second_graphic]
    assert state.source_models == (receptor, second) and state.generation != generation


def _graphics_session(monkeypatch):
    monkeypatch.setitem(sys.modules, "chimerax.core.models",
                        SimpleNamespace(REMOVE_MODELS="remove models"))
    handlers = []
    closed = []
    def add_handler(_name, callback):
        handler = SimpleNamespace(callback=callback)
        handler.remove = lambda: handlers.remove(handler)
        handlers.append(handler)
        return handler
    def close(models):
        closed.extend(models)
        for handler in tuple(handlers):
            handler.callback("remove models", models)
        for model in models:
            model.deleted = True
    session = SimpleNamespace(triggers=SimpleNamespace(add_handler=add_handler),
                              models=SimpleNamespace(close=close))
    return session, handlers, closed


def test_command_graphics_close_with_source_by_identity_and_unregister(monkeypatch):
    session, handlers, closed = _graphics_session(monkeypatch)
    source, same_number, graphic = Model(1), Model(1), Model(2)
    graphics = [graphic]
    tracker = lc.track_graphics(session, [source], graphics)
    assert lc.track_graphics(session, [source], graphics) is tracker
    tracker._removed("remove models", [same_number])
    assert not closed and len(handlers) == 1
    tracker._removed("remove models", [source])
    assert closed == [graphic] and not graphics and not handlers
    assert session._chemur_graphics_trackers == []
    tracker.close()
    assert closed == [graphic]


def test_manually_closing_last_graphic_removes_tracker_without_closing_source(monkeypatch):
    session, handlers, closed = _graphics_session(monkeypatch)
    source, first, second = Model(), Model(2), Model(3)
    graphics = [first, second]
    lc.track_graphics(session, [source], graphics)
    session.models.close([first])
    assert graphics == [second] and len(handlers) == 1
    session.models.close([second])
    assert not graphics and not handlers
    assert closed == [first, second] and not source.deleted
    assert lc.track_graphics(session, [source], []) is None

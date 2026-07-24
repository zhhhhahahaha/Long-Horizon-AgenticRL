from types import SimpleNamespace

from slime.utils import logging_utils


def test_finish_tracking_finishes_run_and_tears_down_service(monkeypatch):
    calls = []
    monkeypatch.setattr(logging_utils.wandb, "run", object())
    monkeypatch.setattr(logging_utils.wandb, "finish", lambda: calls.append("finish"))
    monkeypatch.setattr(logging_utils.wandb, "teardown", lambda: calls.append("teardown"))

    logging_utils.finish_tracking(SimpleNamespace(use_wandb=True, wandb_explicit_teardown=True))

    assert calls == ["finish", "teardown"]


def test_finish_tracking_tears_down_service_after_finish_error(monkeypatch):
    calls = []
    monkeypatch.setattr(logging_utils.wandb, "run", object())

    def fail_finish():
        calls.append("finish")
        raise RuntimeError("finish failed")

    monkeypatch.setattr(logging_utils.wandb, "finish", fail_finish)
    monkeypatch.setattr(logging_utils.wandb, "teardown", lambda: calls.append("teardown"))

    logging_utils.finish_tracking(SimpleNamespace(use_wandb=True, wandb_explicit_teardown=True))

    assert calls == ["finish", "teardown"]


def test_finish_tracking_does_not_tear_down_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(logging_utils.wandb, "run", object())
    monkeypatch.setattr(logging_utils.wandb, "finish", lambda: calls.append("finish"))
    monkeypatch.setattr(logging_utils.wandb, "teardown", lambda: calls.append("teardown"))

    logging_utils.finish_tracking(SimpleNamespace(use_wandb=True))

    assert calls == ["finish"]


def test_finish_tracking_skips_process_without_run(monkeypatch):
    calls = []
    monkeypatch.setattr(logging_utils.wandb, "run", None)
    monkeypatch.setattr(logging_utils.wandb, "finish", lambda: calls.append("finish"))
    monkeypatch.setattr(logging_utils.wandb, "teardown", lambda: calls.append("teardown"))

    logging_utils.finish_tracking(SimpleNamespace(use_wandb=True, wandb_explicit_teardown=True))

    assert calls == []


def test_finish_tracking_ignores_teardown_system_exit(monkeypatch):
    monkeypatch.setattr(logging_utils.wandb, "run", object())
    monkeypatch.setattr(logging_utils.wandb, "finish", lambda: None)

    def fail_teardown():
        raise SystemExit(1)

    monkeypatch.setattr(logging_utils.wandb, "teardown", fail_teardown)

    logging_utils.finish_tracking(SimpleNamespace(use_wandb=True, wandb_explicit_teardown=True))

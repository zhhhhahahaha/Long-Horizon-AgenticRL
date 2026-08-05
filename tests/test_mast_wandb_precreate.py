from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_precreate_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "supo_browsecomp"
        / "mast"
        / "wandb"
        / "wandb_precreate.py"
    )
    module_name = "test_mast_wandb_precreate_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_offline_metadata(monkeypatch, precreate, **overrides):
    metadata = {
        "run_id": "abc123",
        "entity": "entity",
        "project": "project",
        "group": "expected-group",
    }
    metadata.update(overrides)
    monkeypatch.setattr(precreate, "_run_metadata", lambda path: metadata)


@pytest.mark.unit
def test_run_metadata_reads_offline_transaction_log(tmp_path):
    precreate = _load_precreate_module()
    run_dir = tmp_path / "offline-run-20260724_120000-abc123"
    run_dir.mkdir()
    store = precreate.DataStore()
    store.open_for_write(str(run_dir / "run-abc123.wandb"))
    record = precreate.wandb_internal_pb2.Record()
    record.run.run_id = "abc123"
    record.run.entity = "entity"
    record.run.project = "project"
    record.run.run_group = "expected-group"
    store.write(record)
    store.close()

    assert precreate._run_metadata(str(run_dir)) == {
        "run_id": "abc123",
        "entity": "entity",
        "project": "project",
        "group": "expected-group",
    }


@pytest.mark.unit
@pytest.mark.parametrize("group", [None, ""])
def test_existing_ungrouped_run_allows_offline_sync_retry(monkeypatch, group, capsys):
    precreate = _load_precreate_module()
    run = SimpleNamespace(group=group)
    _patch_offline_metadata(monkeypatch, precreate)
    monkeypatch.setattr(
        precreate.wandb,
        "Api",
        lambda timeout: SimpleNamespace(default_entity="entity", run=lambda path: run),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_precreate.py",
            "--entity",
            "entity",
            "--project",
            "project",
            "--group",
            "expected-group",
            "offline-run-20260724_120000-abc123",
        ],
    )

    precreate.main()

    assert "continuing so offline sync can populate it" in capsys.readouterr().out


@pytest.mark.unit
def test_existing_run_in_another_group_is_rejected(monkeypatch):
    precreate = _load_precreate_module()
    run = SimpleNamespace(group="other-group")
    _patch_offline_metadata(monkeypatch, precreate)
    monkeypatch.setattr(
        precreate.wandb,
        "Api",
        lambda timeout: SimpleNamespace(default_entity="entity", run=lambda path: run),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_precreate.py",
            "--entity",
            "entity",
            "--project",
            "project",
            "--group",
            "expected-group",
            "offline-run-20260724_120000-abc123",
        ],
    )

    with pytest.raises(RuntimeError, match="unexpected group 'other-group'"):
        precreate.main()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "actual", "message"),
    [
        ("run_id", "other123", "has run_id='other123'; expected 'abc123'"),
        ("project", "other-project", "has project='other-project'; expected 'project'"),
        ("group", "other-group", "has group='other-group'; expected 'expected-group'"),
    ],
)
def test_offline_metadata_mismatch_is_rejected_before_remote_lookup(monkeypatch, field, actual, message):
    precreate = _load_precreate_module()
    _patch_offline_metadata(monkeypatch, precreate, **{field: actual})
    api = SimpleNamespace(
        default_entity="entity",
        run=lambda path: (_ for _ in ()).throw(AssertionError("remote lookup must not run")),
    )
    monkeypatch.setattr(precreate.wandb, "Api", lambda timeout: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_precreate.py",
            "--entity",
            "entity",
            "--project",
            "project",
            "--group",
            "expected-group",
            "offline-run-20260724_120000-abc123",
        ],
    )

    with pytest.raises(RuntimeError, match=message):
        precreate.main()


@pytest.mark.unit
def test_implicit_offline_entity_must_match_authenticated_default(monkeypatch):
    precreate = _load_precreate_module()
    _patch_offline_metadata(monkeypatch, precreate, entity="")
    api = SimpleNamespace(
        default_entity="other-entity",
        run=lambda path: (_ for _ in ()).throw(AssertionError("remote lookup must not run")),
    )
    monkeypatch.setattr(precreate.wandb, "Api", lambda timeout: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_precreate.py",
            "--entity",
            "entity",
            "--project",
            "project",
            "--group",
            "expected-group",
            "offline-run-20260724_120000-abc123",
        ],
    )

    with pytest.raises(RuntimeError, match="targets entity 'other-entity'; expected 'entity'"):
        precreate.main()

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite

from scripts.make_submission import (
    CheckpointInfo,
    SubmissionError,
    SubmissionPlan,
    build_submission,
    inspect_checkpoint,
    kaggle_command,
    main,
    resolve_checkpoint,
    select_checkpoint,
    submit_archive,
    validate_bundled_imports,
    validate_deck,
)
from src.checkpoint_registry import append_checkpoint_record, sha256_file
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.ppo_actor import build_actor_critic


def checkpoint_config() -> dict:
    """Minimal portable config representative of a versioned training checkpoint."""
    return {
        "model": {
            "embed_dim": 8,
            "backbone": {
                "_target_": "src.models.mlp.MLPBackbone",
                "num_cells": [8],
                "activation": "tanh",
            },
            "head": {"_target_": "src.models.heads.LinearPolicyHead"},
            "value_head": {"num_cells": [8]},
        },
        "env": {
            "encoder": "structured",
            "max_options": 96,
            "deck0": "decks/example.csv",
        },
    }


def write_checkpoint(path: Path, marker: int, *, frames: int = 100) -> Path:
    """Write a small self-describing checkpoint for packager tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create(checkpoint_config())
    max_options = int(cfg.env.max_options)
    encoder = StructuredObservationEncoder(max_options=max_options)
    obs_spec = Composite(
        observation=encoder.spec(),
        action_mask=Binary(n=max_options + 1, dtype=torch.bool),
    )
    actor_critic = build_actor_critic(
        cfg,
        obs_spec,
        Categorical(max_options + 1, dtype=torch.int64),
    )
    with torch.no_grad():
        next(actor_critic.parameters()).fill_(marker)
    torch.save(
        {
            "format_version": 1,
            "state_dict": actor_critic.state_dict(),
            "config": checkpoint_config(),
            "frames": frames,
        },
        path,
    )
    return path


def transformer_checkpoint_config() -> dict:
    """Minimal portable transformer+pointer config for packager tests.

    Pointer-head shape is the interesting case: unlike ``checkpoint_config``'s
    MLP+linear-head baseline, this exercises ``option_tokens`` (per-option
    tokens feeding ``PointerPolicyHead``) through the actual bundled runtime.
    """
    return {
        "model": {
            "embed_dim": 8,
            "backbone": {
                "_target_": "src.models.transformer.TransformerBackbone",
                "num_heads": 2,
                "num_layers": 1,
                "ff_dim": 8,
                "activation": "gelu",
                "option_tokens": True,
            },
            "head": {"_target_": "src.models.heads.PointerPolicyHead"},
            "value_head": {"num_cells": [8]},
        },
        "env": {
            "encoder": "structured",
            "max_options": 96,
            "deck0": "decks/example.csv",
        },
    }


def write_transformer_checkpoint(path: Path, marker: int, *, frames: int = 100) -> Path:
    """Write a small self-describing transformer+pointer checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create(transformer_checkpoint_config())
    max_options = int(cfg.env.max_options)
    encoder = StructuredObservationEncoder(max_options=max_options)
    obs_spec = Composite(
        observation=encoder.spec(),
        action_mask=Binary(n=max_options + 1, dtype=torch.bool),
    )
    actor_critic = build_actor_critic(
        cfg,
        obs_spec,
        Categorical(max_options + 1, dtype=torch.int64),
    )
    with torch.no_grad():
        next(actor_critic.parameters()).fill_(marker)
    torch.save(
        {
            "format_version": 1,
            "state_dict": actor_critic.state_dict(),
            "config": transformer_checkpoint_config(),
            "frames": frames,
        },
        path,
    )
    return path


def test_registry_bottom_is_latest_and_hash_is_resolved_from_registry(tmp_path) -> None:
    """Append order defines latest, and a printed key selects its exact row."""
    older = write_checkpoint(tmp_path / "older.pt", 1)
    newer = write_checkpoint(tmp_path / "newer.pt", 2)
    registry = tmp_path / "logs" / "checkpoint_keys.csv"
    older_record = append_checkpoint_record(
        registry,
        older,
        digest=sha256_file(older),
        frames=1,
        repo_root=tmp_path,
    )
    append_checkpoint_record(
        registry,
        newer,
        digest=sha256_file(newer),
        frames=2,
        repo_root=tmp_path,
    )

    assert (
        select_checkpoint(None, [tmp_path], tmp_path, registry).path == newer.resolve()
    )
    assert (
        select_checkpoint(older_record.key, [tmp_path], tmp_path, registry).path
        == older.resolve()
    )


def test_explicit_unregistered_path_name_and_hash_remain_supported(tmp_path) -> None:
    """Legacy checkpoints can still be selected without registry records."""
    checkpoint = write_checkpoint(tmp_path / "legacy.pt", 1)
    info = inspect_checkpoint(checkpoint)

    assert (
        resolve_checkpoint(str(checkpoint), [tmp_path], tmp_path)
        == checkpoint.resolve()
    )
    assert resolve_checkpoint("legacy", [tmp_path], tmp_path) == checkpoint.resolve()
    assert resolve_checkpoint(info.key, [tmp_path], tmp_path) == checkpoint.resolve()
    with pytest.raises(SubmissionError, match="authoritative completion order"):
        resolve_checkpoint(None, [tmp_path], tmp_path)


def test_embedded_config_warns_that_explicit_override_is_ignored(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Self-describing checkpoints retain precedence over legacy overrides."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint.pt", 2)
    override = tmp_path / "override.yaml"
    override.write_text("invalid: legacy-only\n", encoding="utf-8")

    info = inspect_checkpoint(checkpoint, override)

    assert info.config_source == "embedded checkpoint config"
    assert info.config == checkpoint_config()
    assert (
        f"warning: config override {override} was ignored because the checkpoint "
        "contains an embedded configuration." in capsys.readouterr().err
    )


def test_build_submission_has_kaggle_root_shape_and_audited_runtime(tmp_path) -> None:
    """The tarball exposes main.py/deck.csv at root and excludes local binaries."""
    root = Path(__file__).parents[1]
    checkpoint_path = write_checkpoint(
        tmp_path / "checkpoints" / "model.pt", 3, frames=456
    )
    checkpoint = inspect_checkpoint(checkpoint_path)
    plan = SubmissionPlan(
        repo_root=root,
        checkpoint=checkpoint,
        deck=root / "decks" / "example.csv",
        label="test-agent",
        staging_dir=tmp_path / "submissions" / "test-agent",
        archive=tmp_path / "submissions" / "test-agent.tar.gz",
        competition="pokemon-tcg-ai-battle",
        message="test-agent",
    )

    archive_digest = build_submission(plan)

    assert len(archive_digest) == 64
    with tarfile.open(plan.archive, "r:gz") as bundle:
        names = bundle.getnames()
        assert "main.py" in names
        assert "cg_api.py" in names
        assert "model.pt" in names
        assert "deck.csv" in names
        assert "runtime.py" in names
        assert "model_config.json" in names
        assert "submission_manifest.json" in names
        assert all(not name.startswith("test-agent/") for name in names)
        assert all(not name.endswith((".dylib", ".dll")) for name in names)

    manifest = json.loads((plan.staging_dir / "submission_manifest.json").read_text())
    assert manifest["checkpoint"]["sha256"] == checkpoint.sha256
    assert manifest["checkpoint"]["key"] == checkpoint.key
    assert manifest["checkpoint"]["frames"] == 456
    assert manifest["inference"] == {"action_selection": "sample"}
    model_config = json.loads((plan.staging_dir / "model_config.json").read_text())
    assert model_config["inference"] == {"action_selection": "sample"}
    assert manifest["bundled_files"] == [
        "cg_api.py",
        "deck.csv",
        "main.py",
        "model.pt",
        "model_config.json",
        "runtime.py",
        "submission_manifest.json",
    ]
    assert manifest["external_runtime_dependencies"] == [
        "torch",
    ]
    assert manifest["preflight"] == {
        "archive_reextracted": True,
        "calls": ["deck_setup", "model_inference"],
        "external_import_allowlist": ["torch"],
        "kaggle_loader": "compile-exec-last-callable",
        "python_version": "3.11",
    }

    with pytest.raises(SubmissionError, match="already exists"):
        build_submission(plan)
    assert len(build_submission(plan, force=True)) == 64


def test_build_submission_succeeds_for_transformer_pointer_checkpoint(
    tmp_path,
) -> None:
    """The preflight rebuilds the bundle and runs real inference through the
    extracted runtime for a transformer trunk + pointer head, not just the
    MLP + linear-head baseline every other packaging test uses -- this is the
    only place the packaged ``runtime.py`` (not the in-process module) is
    exercised for that combination.
    """
    root = Path(__file__).parents[1]
    checkpoint_path = write_transformer_checkpoint(
        tmp_path / "checkpoints" / "model.pt", 3, frames=123
    )
    plan = SubmissionPlan(
        repo_root=root,
        checkpoint=inspect_checkpoint(checkpoint_path),
        deck=root / "decks" / "example.csv",
        label="transformer-agent",
        staging_dir=tmp_path / "submissions" / "transformer-agent",
        archive=tmp_path / "submissions" / "transformer-agent.tar.gz",
        competition="pokemon-tcg-ai-battle",
        message="transformer-agent",
    )

    archive_digest = build_submission(plan)

    assert len(archive_digest) == 64
    with tarfile.open(plan.archive, "r:gz") as bundle:
        assert "model.pt" in bundle.getnames()


def test_failed_forced_rebuild_preserves_previous_validated_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late archive-validation failure cannot destroy a known-good build."""
    root = Path(__file__).parents[1]
    checkpoint_path = write_checkpoint(tmp_path / "checkpoint.pt", 3)
    plan = SubmissionPlan(
        repo_root=root,
        checkpoint=inspect_checkpoint(checkpoint_path),
        deck=root / "decks" / "example.csv",
        label="atomic-agent",
        staging_dir=tmp_path / "submissions" / "atomic-agent",
        archive=tmp_path / "submissions" / "atomic-agent.tar.gz",
        competition="pokemon-tcg-ai-battle",
        message="atomic-agent",
    )
    build_submission(plan)
    archive_before = plan.archive.read_bytes()
    stage_before = {
        path.relative_to(plan.staging_dir): path.read_bytes()
        for path in plan.staging_dir.iterdir()
        if path.is_file()
    }

    def reject_archive(_plan: SubmissionPlan) -> None:
        raise SubmissionError("simulated finished-archive failure")

    monkeypatch.setattr(
        "scripts.make_submission.validate_submission_archive", reject_archive
    )

    with pytest.raises(SubmissionError, match="simulated finished-archive failure"):
        build_submission(plan, force=True)

    assert plan.archive.read_bytes() == archive_before
    assert {
        path.relative_to(plan.staging_dir): path.read_bytes()
        for path in plan.staging_dir.iterdir()
        if path.is_file()
    } == stage_before


def test_extracted_bundle_strictly_rebuilds_generated_policy(tmp_path) -> None:
    """Generated fixtures exercise the same extracted runtime used by Kaggle."""
    root = Path(__file__).parents[1]
    checkpoint_path = write_checkpoint(tmp_path / "checkpoint.pt", 4)
    plan = SubmissionPlan(
        repo_root=root,
        checkpoint=inspect_checkpoint(checkpoint_path),
        deck=root / "decks" / "example.csv",
        label="runtime-agent",
        staging_dir=tmp_path / "submissions" / "runtime-agent",
        archive=tmp_path / "submissions" / "runtime-agent.tar.gz",
        competition="pokemon-tcg-ai-battle",
        message="runtime-agent",
    )
    build_submission(plan)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(plan.archive, "r:gz") as bundle:
        bundle.extractall(extracted, filter="data")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(extracted)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins; original = builtins.__import__; "
                "blocked = {'torchrl', 'tensordict', 'hydra', 'omegaconf'}; "
                "builtins.__import__ = lambda name, *a, **k: "
                "(_ for _ in ()).throw(ImportError(name)) "
                "if name.split('.')[0] in blocked else original(name, *a, **k); "
                "import main; first = main._load_policy(); "
                "second = main._load_policy(); "
                "print(type(first).__name__, first is second, "
                "main._CACHED_POLICY is first)"
            ),
        ],
        cwd=extracted,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Policy True True"


def test_registry_staleness_and_malformed_rows_block_latest(tmp_path) -> None:
    """Latest never falls back to a scan when the authoritative row is bad."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint.pt", 5)
    registry = tmp_path / "logs" / "checkpoint_keys.csv"
    append_checkpoint_record(
        registry,
        checkpoint,
        digest=sha256_file(checkpoint),
        frames=5,
        repo_root=tmp_path,
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(SubmissionError, match="hash mismatch"):
        select_checkpoint(None, [tmp_path], tmp_path, registry)

    registry.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="Malformed checkpoint registry header"):
        select_checkpoint(None, [tmp_path], tmp_path, registry)


def test_registered_missing_config_blocks_selection(tmp_path) -> None:
    """A legacy config recorded in the CSV must still exist."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint.pt", 6)
    config = tmp_path / "legacy.yaml"
    config.write_text("model: {}\n", encoding="utf-8")
    registry = tmp_path / "logs" / "checkpoint_keys.csv"
    append_checkpoint_record(
        registry,
        checkpoint,
        digest=sha256_file(checkpoint),
        frames=6,
        repo_root=tmp_path,
        config_path=config,
    )
    config.unlink()

    with pytest.raises(SubmissionError, match="config no longer exists"):
        select_checkpoint(None, [tmp_path], tmp_path, registry)


def test_checkpoint_and_deck_validation_fail_before_build(tmp_path) -> None:
    """Unreadable/incompatible checkpoints and malformed decks are rejected."""
    unreadable = tmp_path / "unreadable.pt"
    unreadable.write_bytes(b"not a torch checkpoint")
    with pytest.raises(SubmissionError, match="Could not read checkpoint"):
        inspect_checkpoint(unreadable)

    incompatible = tmp_path / "incompatible.pt"
    torch.save(
        {
            "state_dict": {"wrong": torch.ones(1)},
            "config": checkpoint_config(),
        },
        incompatible,
    )
    with pytest.raises(SubmissionError, match="incompatible"):
        inspect_checkpoint(incompatible)

    non_structured = write_checkpoint(tmp_path / "flat.pt", 7)
    payload = torch.load(non_structured, map_location="cpu", weights_only=True)
    payload["config"]["env"]["encoder"] = "flat"
    torch.save(payload, non_structured)
    with pytest.raises(SubmissionError, match="only env.encoder=structured"):
        inspect_checkpoint(non_structured)

    bad_count = tmp_path / "bad-count.csv"
    bad_count.write_text("1\n" * 59, encoding="utf-8")
    with pytest.raises(SubmissionError, match="exactly 60"):
        validate_deck(bad_count)
    bad_integer = tmp_path / "bad-integer.csv"
    bad_integer.write_text("1\n" * 59 + "not-an-integer\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="one integer"):
        validate_deck(bad_integer)


def test_bundle_import_audit_rejects_unbundled_dependencies(tmp_path) -> None:
    """Every non-stdlib dependency except torch must exist in the archive."""
    (tmp_path / "main.py").write_text("import cg\n", encoding="utf-8")

    with pytest.raises(SubmissionError, match="neither bundled nor allowed: cg"):
        validate_bundled_imports(tmp_path)


def test_bundle_import_audit_rejects_dynamic_imports(tmp_path) -> None:
    """Runtime dependencies must remain visible to the static allowlist audit."""
    (tmp_path / "main.py").write_text(
        '__import__("surprise_dependency")\n', encoding="utf-8"
    )

    with pytest.raises(SubmissionError, match="dynamic imports"):
        validate_bundled_imports(tmp_path)


def test_bundle_import_audit_rejects_training_packages_first(tmp_path) -> None:
    """Known Kaggle-missing training packages get the most actionable error."""
    (tmp_path / "main.py").write_text("import torchrl\n", encoding="utf-8")

    with pytest.raises(SubmissionError, match="unavailable training packages: torchrl"):
        validate_bundled_imports(tmp_path)


def test_bundle_import_audit_rejects_missing_local_modules(tmp_path) -> None:
    """Allowlisted local module names are still required to be packaged."""
    (tmp_path / "main.py").write_text("import runtime\n", encoding="utf-8")

    with pytest.raises(SubmissionError, match="missing from the bundle: runtime"):
        validate_bundled_imports(tmp_path)


def test_bundle_import_audit_enforces_kaggle_python_version(tmp_path) -> None:
    """Python syntax newer than Kaggle's 3.11 image is rejected locally."""
    (tmp_path / "main.py").write_text("type Alias = int\n", encoding="utf-8")

    with pytest.raises(SubmissionError, match="not valid Python 3.11"):
        validate_bundled_imports(tmp_path)


def test_noninteractive_confirmation_requires_yes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Batch execution without stdin gets an actionable --yes hint."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint.pt", 8)
    registry = tmp_path / "logs" / "checkpoint_keys.csv"
    append_checkpoint_record(
        registry,
        checkpoint,
        digest=sha256_file(checkpoint),
        frames=8,
        repo_root=tmp_path,
    )
    deck = tmp_path / "decks" / "example.csv"
    deck.parent.mkdir()
    deck.write_text("1\n" * 60, encoding="utf-8")
    monkeypatch.setattr("scripts.make_submission.repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError)
    )
    monkeypatch.delenv("CHECKPOINT_KEYS_FILE", raising=False)
    monkeypatch.delenv("KAGGLE_DECK_PATH", raising=False)
    monkeypatch.delenv("KAGGLE_SUBMISSIONS_DIR", raising=False)

    assert main([]) == 2
    assert "pass --yes for batch use" in capsys.readouterr().out


def test_kaggle_cli_command_is_used_for_submission(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submission invokes the official CLI command with the built archive."""
    root = Path(__file__).parents[1]
    archive = tmp_path / "submission.tar.gz"
    archive.write_bytes(b"archive")
    checkpoint = CheckpointInfo(
        path=tmp_path / "model.pt",
        sha256="a" * 64,
        config=checkpoint_config(),
        config_source="test",
        frames=1,
    )
    plan = SubmissionPlan(
        repo_root=root,
        checkpoint=checkpoint,
        deck=root / "decks" / "example.csv",
        label="agent",
        staging_dir=tmp_path / "agent",
        archive=archive,
        competition="pokemon-tcg-ai-battle",
        message="message with spaces",
    )
    calls: list[tuple[list[str], Path, bool]] = []
    monkeypatch.setattr(
        "scripts.make_submission.shutil.which", lambda _command: "/bin/kaggle"
    )

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.make_submission.subprocess.run", fake_run)

    submit_archive(plan)

    assert calls == [(kaggle_command(plan), root, True)]
    assert calls[0][0][:4] == [
        "kaggle",
        "competitions",
        "submit",
        "pokemon-tcg-ai-battle",
    ]


def _isolate_local_paths(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect every local-file default into tmp_path.

    Leaves ``repo_root()`` pointing at the real repository so a build can
    still find ``submission/main.py`` and friends; registry rows below store
    absolute checkpoint paths, so they resolve correctly regardless. The deck
    stays the repository's real example deck: the native battle engine
    enforces per-card copy limits a synthetic all-one-card deck would violate.
    """
    monkeypatch.setenv(
        "CHECKPOINT_KEYS_FILE", str(tmp_path / "logs" / "checkpoint_keys.csv")
    )
    monkeypatch.setenv("KAGGLE_DECK_PATH", str(root / "decks" / "example.csv"))
    monkeypatch.setenv("KAGGLE_SUBMISSIONS_DIR", str(tmp_path / "submissions"))


def _prepare_registered_checkpoint(root: Path, tmp_path: Path, marker: int) -> str:
    """Write a checkpoint under tmp_path and register it against the real root.

    :param root: The real repository root, matching what ``repo_root()``
        returns in ``main()`` so the registry's stored path resolves later.
    :return: The checkpoint's 12-character registry key.
    """
    checkpoint = write_checkpoint(tmp_path / "checkpoint.pt", marker)
    registry = tmp_path / "logs" / "checkpoint_keys.csv"
    digest = sha256_file(checkpoint)
    append_checkpoint_record(
        registry, checkpoint, digest=digest, frames=marker, repo_root=root
    )
    return digest[:12]


def test_main_completes_a_successful_build(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    _prepare_registered_checkpoint(root, tmp_path, 9)
    _isolate_local_paths(root, tmp_path, monkeypatch)

    exit_code = main(["--label", "test-agent", "--yes"])

    assert exit_code == 0
    assert (tmp_path / "submissions" / "test-agent.tar.gz").is_file()

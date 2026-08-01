"""Build and optionally upload a Kaggle Pokémon TCG agent submission."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

_DIRECT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_DIRECT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIRECT_REPO_ROOT))

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite

from src.checkpoint_registry import (
    CHECKPOINT_KEY_LENGTH,
    CheckpointRecord,
    CheckpointRegistryError,
    CheckpointRegistryLookupError,
    find_checkpoint_record,
    latest_checkpoint_record,
    resolve_record_path,
    sha256_file,
    validate_checkpoint_record,
)
from src.env.battle_handle import BattleHandle
from src.env.structured_observation_encoder import StructuredObservationEncoder
from src.policies.greedy_policy_opponent import checkpoint_state_dict
from src.policies.ppo_actor import build_actor_critic
from submission.runtime import Policy as PortablePolicy

COMPETITION = "pokemon-tcg-ai-battle"
_HASH_SELECTOR = re.compile(r"[0-9a-fA-F]{8,64}")

# The competition supplies ``torch`` but neither TorchRL nor the ``cg`` Python
# package. These self-contained files use only torch and the standard library.
RUNTIME_FILES = ("cg_api.py", "runtime.py")
_BUNDLED_MODULES = {"cg_api", "main", "runtime"}
_ALLOWED_EXTERNAL_MODULES = {"torch"}
_ALLOWED_STDLIB_MODULES = {
    "__future__",
    "collections",
    "dataclasses",
    "enum",
    "itertools",
    "json",
    "pathlib",
    "sys",
    "typing",
}
_BLOCKED_TRAINING_MODULES = {"hydra", "omegaconf", "tensordict", "torchrl"}

# Mirrors Kaggle's file-agent path: compile/exec with an empty globals mapping,
# the agent directory appended to sys.path, and the last callable selected.
# https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/agent.py
_KAGGLE_PREFLIGHT = r"""
import builtins
import json
import pathlib
import sys

main_path = pathlib.Path(sys.argv[1]).resolve()
blocked = {"hydra", "omegaconf", "tensordict", "torchrl"}
original_import = builtins.__import__

def restricted_import(name, *args, **kwargs):
    if name.partition(".")[0] in blocked:
        raise ImportError(f"Kaggle runtime does not provide {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = restricted_import
sys.path.append(str(main_path.parent))
namespace = {}
source = main_path.read_text(encoding="utf-8")
exec(compile(source, str(main_path), "exec"), namespace)
callables = [value for value in namespace.values() if callable(value)]
if not callables or namespace.get("agent") is not callables[-1]:
    raise RuntimeError("main.py must expose agent as its last callable")
for module_name in ("cg_api", "runtime"):
    module_path = pathlib.Path(sys.modules[module_name].__file__).resolve()
    if module_path.parent != main_path.parent:
        raise RuntimeError(f"{module_name} resolved outside the submission bundle")
agent = callables[-1]

setup_action = agent({"select": None, "logs": [], "current": None})
if not isinstance(setup_action, list) or len(setup_action) != 60:
    raise RuntimeError("agent setup call must return a 60-card list")
if not all(isinstance(card, int) for card in setup_action):
    raise RuntimeError("agent setup deck must contain only integers")

observation = json.loads(sys.stdin.read())
action = agent(observation)
select = observation["select"]
if not isinstance(action, list) or not all(isinstance(index, int) for index in action):
    raise RuntimeError("agent inference call must return a list of integers")
if len(action) != len(set(action)):
    raise RuntimeError("agent inference call returned duplicate option indices")
if not select["minCount"] <= len(action) <= select["maxCount"]:
    raise RuntimeError("agent inference action violates selection count bounds")
if not all(0 <= index < len(select["option"]) for index in action):
    raise RuntimeError("agent inference action contains an invalid option index")
print("kaggle-preflight-ok")
"""


class SubmissionError(RuntimeError):
    """User-facing validation failure while planning or building a submission."""


@dataclass(frozen=True)
class CheckpointInfo:
    """Resolved checkpoint identity and portable inference configuration."""

    path: Path
    sha256: str
    config: dict[str, Any]
    config_source: str
    frames: int | None

    @property
    def key(self) -> str:
        """Short SHA-256 key printed by training and accepted by this CLI."""
        return self.sha256[:CHECKPOINT_KEY_LENGTH]


@dataclass(frozen=True)
class CheckpointSelection:
    """Checkpoint path and optional legacy config selected for packaging."""

    path: Path
    config_path: Path | None = None
    record: CheckpointRecord | None = None


@dataclass(frozen=True)
class SubmissionPlan:
    """Validated inputs and destinations for one build."""

    repo_root: Path
    checkpoint: CheckpointInfo
    deck: Path
    label: str
    staging_dir: Path
    archive: Path
    competition: str
    message: str
    action_selection: str = "sample"


def repo_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def discover_checkpoints(roots: Sequence[Path]) -> list[Path]:
    """Find checkpoint ``.pt`` files below the configured roots."""
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".pt":
            paths.add(root.resolve())
        elif root.is_dir():
            paths.update(
                path.resolve() for path in root.rglob("*.pt") if path.is_file()
            )
    return sorted(paths)


def resolve_checkpoint(selector: str | None, roots: Sequence[Path], root: Path) -> Path:
    """Resolve an explicit checkpoint path, filename, or legacy hash prefix."""
    if selector and selector.lower() != "latest":
        requested = Path(selector).expanduser()
        path_candidates = [requested]
        if not requested.is_absolute():
            path_candidates.insert(0, root / requested)
        for candidate in path_candidates:
            if candidate.is_file():
                return candidate.resolve()

    candidates = discover_checkpoints(roots)
    if not candidates:
        joined = ", ".join(str(path) for path in roots)
        raise SubmissionError(f"No .pt checkpoints found under: {joined}")
    if selector is None or selector.lower() == "latest":
        raise SubmissionError(
            "'latest' must be resolved through logs/checkpoint_keys.csv; "
            "the registry is the authoritative completion order."
        )

    name_matches = [path for path in candidates if selector in {path.name, path.stem}]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        raise SubmissionError(f"Checkpoint name {selector!r} is ambiguous.")

    if not _HASH_SELECTOR.fullmatch(selector):
        raise SubmissionError(
            f"Checkpoint {selector!r} is neither an existing path, unique filename, "
            "nor a SHA-256 prefix of at least 8 hexadecimal characters."
        )
    lowered = selector.lower()
    digest_matches = [
        path for path in candidates if sha256_file(path).startswith(lowered)
    ]
    if len(digest_matches) == 1:
        return digest_matches[0]
    if not digest_matches:
        raise SubmissionError(f"No checkpoint matches SHA-256 prefix {selector!r}.")
    raise SubmissionError(
        f"Checkpoint key {selector!r} is ambiguous; provide more characters."
    )


def _selection_from_record(record: CheckpointRecord, root: Path) -> CheckpointSelection:
    """Validate a registry row and resolve its optional legacy config."""
    try:
        checkpoint = validate_checkpoint_record(record, root)
    except CheckpointRegistryError as error:
        raise SubmissionError(str(error)) from error
    config_path = None
    if record.config_path:
        config_path = resolve_record_path(record.config_path, root)
        if not config_path.is_file():
            raise SubmissionError(
                f"Registered model config no longer exists: {config_path}"
            )
    return CheckpointSelection(checkpoint, config_path, record)


def select_checkpoint(
    selector: str | None,
    roots: Sequence[Path],
    root: Path,
    registry_path: Path,
) -> CheckpointSelection:
    """Select latest from the registry, or resolve an explicit path/name/key."""
    if selector is None or selector.lower() == "latest":
        try:
            return _selection_from_record(latest_checkpoint_record(registry_path), root)
        except CheckpointRegistryError as error:
            raise SubmissionError(str(error)) from error

    if _HASH_SELECTOR.fullmatch(selector) and registry_path.is_file():
        try:
            return _selection_from_record(
                find_checkpoint_record(registry_path, selector), root
            )
        except CheckpointRegistryLookupError:
            pass
        except CheckpointRegistryError as error:
            raise SubmissionError(str(error)) from error
    return CheckpointSelection(resolve_checkpoint(selector, roots, root))


def _portable_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a resolved training config to portable inference settings."""
    model = raw.get("model")
    env = raw.get("env")
    if not isinstance(model, Mapping) or not isinstance(env, Mapping):
        raise SubmissionError(
            "Checkpoint config must contain mapping sections 'model' and 'env'."
        )
    if "max_options" not in env:
        raise SubmissionError("Checkpoint config is missing env.max_options.")
    if env.get("encoder", "structured") != "structured":
        raise SubmissionError(
            "The Kaggle runtime currently supports only env.encoder=structured."
        )
    portable_env = {
        key: env[key] for key in ("encoder", "max_options", "deck0") if key in env
    }
    return cast(
        dict[str, Any],
        OmegaConf.to_container(
            OmegaConf.create({"model": dict(model), "env": portable_env}),
            resolve=True,
        ),
    )


def _legacy_config_path(checkpoint: Path) -> Path | None:
    """Find Hydra's config.yaml above a legacy bare-state-dict checkpoint."""
    sidecars = (
        checkpoint.with_suffix(".config.yaml"),
        Path(f"{checkpoint}.config.yaml"),
    )
    for candidate in sidecars:
        if candidate.is_file():
            return candidate
    for ancestor in checkpoint.parents:
        candidate = ancestor / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def _validate_state_dict(payload: object) -> None:
    """Validate either supported checkpoint envelope before it is packaged."""
    state_dict: object = payload
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise SubmissionError("Checkpoint does not contain a non-empty state dict.")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise SubmissionError("Checkpoint state dict must map string keys to tensors.")


def _validate_checkpoint_compatibility(
    payload: object,
    config: Mapping[str, Any],
) -> None:
    """Rebuild the Kaggle policy and strictly load all checkpoint tensors."""
    try:
        cfg = OmegaConf.create(config)
        max_options = int(cfg.env.max_options)
        if max_options <= 0:
            raise ValueError("env.max_options must be positive")
        encoder = StructuredObservationEncoder(max_options=max_options)
        obs_spec = Composite(
            observation=encoder.spec(),
            action_mask=Binary(n=max_options + 1, dtype=torch.bool),
        )
        action_spec = Categorical(max_options + 1, dtype=torch.int64)
        actor_critic = build_actor_critic(cfg, obs_spec, action_spec)
        actor_critic.load_state_dict(checkpoint_state_dict(payload), strict=True)
        PortablePolicy(payload, config)
    except Exception as error:
        raise SubmissionError(
            "Checkpoint weights are incompatible with the structured model config: "
            f"{error}"
        ) from error


def inspect_checkpoint(
    path: Path, explicit_config: Path | None = None
) -> CheckpointInfo:
    """Hash a checkpoint and recover the config needed to serve it."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SubmissionError(f"Could not read checkpoint {path}: {error}") from error
    _validate_state_dict(payload)
    config: Mapping[str, Any] | None = None
    frames: int | None = None
    config_source: str | None = None
    if isinstance(payload, Mapping):
        embedded = payload.get("config")
        if isinstance(embedded, Mapping):
            config = embedded
            config_source = "embedded checkpoint config"
        raw_frames = payload.get("frames")
        if isinstance(raw_frames, int):
            frames = raw_frames

    config_path = explicit_config
    if config is not None and config_path is not None:
        print(
            f"warning: config override {config_path} was ignored because the "
            "checkpoint contains an embedded configuration.",
            file=sys.stderr,
        )
    if config is None and config_path is None:
        config_path = _legacy_config_path(path)
    if config is None and config_path is not None:
        config_path = config_path.expanduser().resolve()
        if not config_path.is_file():
            raise SubmissionError(f"Model config does not exist: {config_path}")
        try:
            loaded = OmegaConf.load(config_path)
        except Exception as error:
            raise SubmissionError(
                f"Could not read model config {config_path}: {error}"
            ) from error
        config = cast(Mapping[str, Any], OmegaConf.to_container(loaded, resolve=True))
        config_source = config_path.name
    if config is None:
        raise SubmissionError(
            "This legacy checkpoint has no embedded model config and no nearby "
            ".hydra/config.yaml. Pass --config PATH."
        )

    portable_config = _portable_config(config)
    _validate_checkpoint_compatibility(payload, portable_config)
    return CheckpointInfo(
        path=path,
        sha256=sha256_file(path),
        config=portable_config,
        config_source=config_source or "unknown",
        frames=frames,
    )


def _resolve_local_path(value: str | Path, root: Path) -> Path:
    """Resolve a user/config path relative to the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def default_deck(checkpoint: CheckpointInfo, root: Path) -> Path:
    """Resolve the training deck recorded in the checkpoint, or example deck."""
    configured = checkpoint.config["env"].get("deck0", "decks/example.csv")
    return _resolve_local_path(str(configured), root)


def validate_deck(path: Path) -> None:
    """Require exactly 60 non-empty lines containing integer card IDs."""
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        card_ids = [int(line) for line in lines if line]
    except (OSError, UnicodeError, ValueError) as error:
        raise SubmissionError(
            f"Deck CSV must contain one integer card ID per line: {path}"
        ) from error
    if len(card_ids) != 60:
        raise SubmissionError(
            f"Deck CSV must contain exactly 60 card IDs, found {len(card_ids)}: {path}"
        )


def _bundle_file_names() -> list[str]:
    """Return the deterministic file list shown before confirmation."""
    files = {
        "main.py",
        "model.pt",
        "model_config.json",
        "deck.csv",
        "submission_manifest.json",
        *RUNTIME_FILES,
    }
    return sorted(files)


def _git_commit(root: Path) -> str | None:
    """Return the source commit without making Git part of the build contract."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_bundled_imports(destination: Path) -> None:
    """Reject direct imports unavailable in Kaggle's isolated agent process."""
    imported_roots: set[str] = set()
    dynamic_imports: list[str] = []
    for path in sorted(destination.rglob("*.py")):
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=(3, 11),
            )
        except (OSError, SyntaxError, UnicodeError) as error:
            raise SubmissionError(
                f"Kaggle runtime source is not valid Python 3.11: {path}: {error}"
            ) from error
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])
            elif isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                )
            ):
                dynamic_imports.append(f"{path.name}:{node.lineno}")

    if dynamic_imports:
        raise SubmissionError(
            "Kaggle runtime uses dynamic imports that cannot be statically audited: "
            + ", ".join(dynamic_imports)
        )

    missing_local = {
        module
        for module in imported_roots & _BUNDLED_MODULES
        if not (destination / f"{module}.py").is_file()
    }
    if missing_local:
        raise SubmissionError(
            "Kaggle runtime imports local modules missing from the bundle: "
            + ", ".join(sorted(missing_local))
        )

    forbidden = imported_roots & _BLOCKED_TRAINING_MODULES
    if forbidden:
        raise SubmissionError(
            "Kaggle runtime imports unavailable training packages: "
            + ", ".join(sorted(forbidden))
        )

    external = imported_roots - (
        _ALLOWED_STDLIB_MODULES | _BUNDLED_MODULES | _ALLOWED_EXTERNAL_MODULES
    )
    if external:
        raise SubmissionError(
            "Kaggle runtime imports packages that are neither bundled nor allowed: "
            + ", ".join(sorted(external))
        )


def _preflight_observation(plan: SubmissionPlan) -> dict[str, Any]:
    """Create one real selection mapping for end-to-end agent inference."""
    cards = [
        int(line)
        for line in plan.deck.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    handle = BattleHandle()
    try:
        observation = handle.start(cards, cards)
        if observation.select is None or observation.current is None:
            raise SubmissionError(
                "The local simulator did not produce an inference selection."
            )
        return asdict(observation)
    finally:
        handle.finish()


def validate_staged_submission(plan: SubmissionPlan, destination: Path) -> None:
    """Run Kaggle's loader contract and real inference in an isolated process."""
    validate_bundled_imports(destination)
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _KAGGLE_PREFLIGHT,
                str(destination / "main.py"),
            ],
            cwd=destination,
            env=environment,
            input=json.dumps(_preflight_observation(plan)),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise SubmissionError(
            "Kaggle-style isolated preflight exceeded 60 seconds."
        ) from error
    if result.returncode != 0 or result.stdout.strip() != "kaggle-preflight-ok":
        details = result.stderr.strip() or result.stdout.strip() or "no output"
        raise SubmissionError(
            "Kaggle-style isolated preflight failed before packaging:\n" + details
        )


def _write_stage(plan: SubmissionPlan, destination: Path) -> None:
    """Populate a temporary staging directory with the audited runtime set."""
    root = plan.repo_root
    shutil.copy2(root / "submission" / "main.py", destination / "main.py")
    for filename in RUNTIME_FILES:
        shutil.copy2(root / "submission" / filename, destination / filename)
    shutil.copy2(plan.checkpoint.path, destination / "model.pt")
    shutil.copy2(plan.deck, destination / "deck.csv")
    runtime_config = dict(plan.checkpoint.config)
    runtime_config["inference"] = {"action_selection": plan.action_selection}
    (destination / "model_config.json").write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "format_version": 1,
        "competition": plan.competition,
        "label": plan.label,
        "inference": {"action_selection": plan.action_selection},
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": _git_commit(root),
        "checkpoint": {
            "filename": plan.checkpoint.path.name,
            "sha256": plan.checkpoint.sha256,
            "key": plan.checkpoint.key,
            "frames": plan.checkpoint.frames,
            "config_source": plan.checkpoint.config_source,
        },
        "deck": {
            "filename": plan.deck.name,
            "sha256": sha256_file(plan.deck),
        },
        "external_runtime_dependencies": [
            "torch",
        ],
        "preflight": {
            "archive_reextracted": True,
            "calls": ["deck_setup", "model_inference"],
            "external_import_allowlist": sorted(_ALLOWED_EXTERNAL_MODULES),
            "kaggle_loader": "compile-exec-last-callable",
            "python_version": "3.11",
        },
        "bundled_files": _bundle_file_names(),
    }
    (destination / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalized_tar_info(path: Path, arcname: str) -> tarfile.TarInfo:
    """Create portable, deterministic tar metadata for one staged file."""
    info = tarfile.TarInfo(arcname)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _create_tar_gz(staging_dir: Path, archive: Path) -> None:
    """Create a gzip-compressed tar whose root is the staging directory's contents."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("wb") as raw_file,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as gzip_file,
        tarfile.open(fileobj=gzip_file, mode="w") as tar_file,
    ):
        for path in sorted(item for item in staging_dir.rglob("*") if item.is_file()):
            arcname = path.relative_to(staging_dir).as_posix()
            with path.open("rb") as input_file:
                tar_file.addfile(_normalized_tar_info(path, arcname), input_file)


def validate_submission_archive(plan: SubmissionPlan) -> None:
    """Extract and re-run preflight against the exact bytes to be uploaded."""
    with tempfile.TemporaryDirectory(
        prefix=f".{plan.label}-archive-check-", dir=plan.archive.parent
    ) as temporary:
        destination = Path(temporary)
        try:
            with tarfile.open(plan.archive, "r:gz") as bundle:
                names = bundle.getnames()
                expected = _bundle_file_names()
                if sorted(names) != expected:
                    raise SubmissionError(
                        "Submission archive root differs from the audited bundle: "
                        f"expected {expected}, found {sorted(names)}"
                    )
                bundle.extractall(destination, filter="data")
        except (OSError, tarfile.TarError) as error:
            raise SubmissionError(
                f"Could not extract the finished submission archive: {error}"
            ) from error
        if sha256_file(destination / "model.pt") != plan.checkpoint.sha256:
            raise SubmissionError(
                "Extracted archive checkpoint hash differs from the selected checkpoint."
            )
        if sha256_file(destination / "deck.csv") != sha256_file(plan.deck):
            raise SubmissionError(
                "Extracted archive deck hash differs from the selected deck."
            )
        validate_staged_submission(plan, destination)


def build_submission(plan: SubmissionPlan, *, force: bool = False) -> str:
    """Build the staged bundle and Kaggle-required ``.tar.gz`` archive."""
    staging = plan.staging_dir.resolve()
    archive = plan.archive.resolve()
    protected = {plan.repo_root.resolve(), Path.home().resolve(), Path(staging.anchor)}
    if staging in protected or plan.repo_root.resolve().is_relative_to(staging):
        raise SubmissionError(f"Refusing unsafe staging directory: {staging}")
    if archive.is_relative_to(staging):
        raise SubmissionError("Archive path must be outside the staging directory.")
    if plan.staging_dir.exists() and not force:
        raise SubmissionError(
            f"Output directory already exists: {plan.staging_dir} (use --force)"
        )
    if plan.archive.exists() and not force:
        raise SubmissionError(f"Archive already exists: {plan.archive} (use --force)")
    plan.staging_dir.parent.mkdir(parents=True, exist_ok=True)
    plan.archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{plan.label}-", dir=plan.staging_dir.parent)
    )
    archive_fd, archive_name = tempfile.mkstemp(
        prefix=f".{plan.label}-", suffix=".tar.gz", dir=plan.archive.parent
    )
    os.close(archive_fd)
    temporary_archive = Path(archive_name)
    try:
        _write_stage(plan, temporary)
        validate_staged_submission(plan, temporary)
        _create_tar_gz(temporary, temporary_archive)
        validate_submission_archive(
            replace(plan, staging_dir=temporary, archive=temporary_archive)
        )

        # Keep any previous validated output intact until the replacement has
        # passed every check against its exact archived bytes.
        if plan.staging_dir.exists():
            shutil.rmtree(plan.staging_dir)
        os.replace(temporary, plan.staging_dir)
        if plan.archive.exists():
            plan.archive.unlink()
        os.replace(temporary_archive, plan.archive)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if temporary_archive.exists():
            temporary_archive.unlink()
    return sha256_file(plan.archive)


def kaggle_command(plan: SubmissionPlan) -> list[str]:
    """Return the official Kaggle CLI competition-submission command."""
    return [
        "kaggle",
        "competitions",
        "submit",
        plan.competition,
        "-f",
        str(plan.archive),
        "-m",
        plan.message,
    ]


def submit_archive(plan: SubmissionPlan) -> None:
    """Upload the built archive through the required Kaggle CLI."""
    if shutil.which("kaggle") is None:
        raise SubmissionError(
            "The official 'kaggle' CLI is not available. Run 'uv sync' and invoke "
            "this script through 'uv run python scripts/make_submission.py'."
        )
    subprocess.run(kaggle_command(plan), cwd=plan.repo_root, check=True)


def _print_summary(plan: SubmissionPlan, submit: bool) -> None:
    """Print the complete plan before any output is changed."""
    print("\nSubmission summary")
    print(f"- Competition: {plan.competition}")
    print(f"- Checkpoint: {plan.checkpoint.path}")
    print(f"- Checkpoint key: {plan.checkpoint.key}")
    print(f"- Deck: {plan.deck}")
    print(f"- Action selection: {plan.action_selection}")
    print(f"- Output dir: {plan.staging_dir}")
    print(f"- Archive: {plan.archive}")
    print("- Bundle contents:")
    for relative in _bundle_file_names():
        print(f"  {relative}")
    print(f"- Submit now: {'yes' if submit else 'no'}")
    if submit:
        print(f"- Message: {plan.message}")


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Package a PPO checkpoint and optionally submit it to Kaggle."
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Checkpoint path, filename, or SHA-256 prefix "
            "(default: bottom row of CHECKPOINT_KEYS_FILE)."
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        help="Directory scanned recursively for checkpoints (default: CHECKPOINTS_DIR or outputs).",
    )
    parser.add_argument(
        "--config", help="Hydra config.yaml for a legacy bare state-dict checkpoint."
    )
    parser.add_argument("--deck", help="Deck CSV (default: checkpoint env.deck0).")
    parser.add_argument(
        "--action-selection",
        choices=("greedy", "sample"),
        default="sample",
        help=(
            "Choose the highest-scoring legal action or sample from the learned "
            "distribution (default: sample)."
        ),
    )
    parser.add_argument(
        "--label", help="Submission label (default: checkpoint-<hash key>)."
    )
    parser.add_argument("--output-dir", help="Staged bundle directory.")
    parser.add_argument("--archive", help="Output archive path (must end in .tar.gz).")
    parser.add_argument(
        "--competition", help=f"Kaggle competition (default: {COMPETITION})."
    )
    parser.add_argument("--message", help="Kaggle submission message (default: label).")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit through Kaggle CLI after packaging.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact output dir/archive if present.",
    )
    return parser


def _checkpoint_roots(args: argparse.Namespace, root: Path) -> list[Path]:
    """Resolve the configured search roots after .env has been loaded."""
    configured = args.checkpoints_dir or os.environ.get("CHECKPOINTS_DIR")
    if configured:
        return [_resolve_local_path(configured, root)]
    return [root / "outputs", root / "checkpoints"]


def _checkpoint_registry_path(root: Path) -> Path:
    """Resolve the authoritative completed-checkpoint CSV."""
    return _resolve_local_path(
        os.environ.get("CHECKPOINT_KEYS_FILE", "logs/checkpoint_keys.csv"), root
    )


def make_plan(args: argparse.Namespace, root: Path) -> SubmissionPlan:
    """Resolve defaults and validate every input before confirmation."""
    selection = select_checkpoint(
        args.checkpoint,
        _checkpoint_roots(args, root),
        root,
        _checkpoint_registry_path(root),
    )
    explicit_config = (
        _resolve_local_path(args.config, root) if args.config else selection.config_path
    )
    checkpoint = inspect_checkpoint(selection.path, explicit_config)
    if checkpoint.frames is None and selection.record is not None:
        checkpoint = replace(checkpoint, frames=selection.record.frames)

    deck_value = args.deck or os.environ.get("KAGGLE_DECK_PATH")
    deck = (
        _resolve_local_path(deck_value, root)
        if deck_value
        else default_deck(checkpoint, root)
    )
    if not deck.is_file():
        raise SubmissionError(f"Deck CSV does not exist: {deck} (pass --deck PATH)")
    validate_deck(deck)

    label = args.label or f"checkpoint-{checkpoint.key}"
    if label in {"", ".", ".."} or "/" in label or "\\" in label:
        raise SubmissionError("Submission label must be a non-empty path-safe name.")
    submissions_root = _resolve_local_path(
        os.environ.get("KAGGLE_SUBMISSIONS_DIR", "submissions"), root
    )
    staging_dir = (
        _resolve_local_path(args.output_dir, root)
        if args.output_dir
        else submissions_root / label
    )
    archive = (
        _resolve_local_path(args.archive, root)
        if args.archive
        else submissions_root / f"{label}.tar.gz"
    )
    if not str(archive).endswith(".tar.gz"):
        raise SubmissionError(
            "Kaggle competition docs require an archive ending in .tar.gz."
        )
    if staging_dir == submissions_root or not staging_dir.is_relative_to(
        submissions_root
    ):
        raise SubmissionError(
            f"Output directory must be a child of KAGGLE_SUBMISSIONS_DIR ({submissions_root})."
        )
    if not archive.is_relative_to(submissions_root):
        raise SubmissionError(
            f"Archive must be inside KAGGLE_SUBMISSIONS_DIR ({submissions_root})."
        )
    if archive.is_relative_to(staging_dir):
        raise SubmissionError(
            "Archive path must be outside the staged bundle directory."
        )

    competition = args.competition or os.environ.get("KAGGLE_COMPETITION", COMPETITION)
    message = args.message or os.environ.get("KAGGLE_DEFAULT_MESSAGE", label)
    return SubmissionPlan(
        repo_root=root,
        checkpoint=checkpoint,
        deck=deck,
        label=label,
        staging_dir=staging_dir,
        archive=archive,
        competition=competition,
        message=message,
        action_selection=args.action_selection,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parser().parse_args(argv)
    root = repo_root()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        print(f"Loaded .env from {env_path}")
    try:
        plan = make_plan(args, root)
        _print_summary(plan, args.submit)
        if not args.yes:
            try:
                confirmed = input("\nProceed? [y/N]: ").strip().lower()
            except EOFError as error:
                raise SubmissionError(
                    "Confirmation requires interactive input; pass --yes for batch use."
                ) from error
            if confirmed not in {"y", "yes"}:
                print("Cancelled.")
                return 1
        archive_digest = build_submission(plan, force=args.force)
        print(f"\nArchive: {plan.archive}")
        print(f"Archive SHA-256: {archive_digest}")
        if args.submit:
            print("Submitting via Kaggle CLI...")
            submit_archive(plan)
            print("Kaggle submission command completed successfully.")
        else:
            print("To submit:")
            print(shlex.join(kaggle_command(plan)))
        return 0
    except (SubmissionError, subprocess.CalledProcessError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

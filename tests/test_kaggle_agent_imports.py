"""Kaggle-facing import and agent-loader contract tests.

Kaggle's public loader compiles and executes a submitted Python file after
adding the file's directory to ``sys.path``, then selects its last callable:
https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/agent.py
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

SUBMISSION_SOURCE = Path(__file__).parents[1] / "submission"
RUNTIME_FILES = tuple(
    SUBMISSION_SOURCE / name for name in ("main.py", "cg_api.py", "runtime.py")
)
LOCAL_MODULES = {"cg_api", "runtime"}
ALLOWED_EXTERNAL_MODULES = {"torch"}
ALLOWED_STDLIB_MODULES = {
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
UNAVAILABLE_TRAINING_MODULES = {"hydra", "omegaconf", "tensordict", "torchrl"}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def test_submission_source_imports_only_kaggle_runtime_dependencies() -> None:
    """Bundled inference imports only stdlib, local modules, and torch."""
    imports = set().union(*(_import_roots(path) for path in RUNTIME_FILES))
    external = imports - ALLOWED_STDLIB_MODULES - LOCAL_MODULES

    assert external == ALLOWED_EXTERNAL_MODULES
    assert imports.isdisjoint(UNAVAILABLE_TRAINING_MODULES)


def test_kaggle_style_loader_exports_agent_without_training_packages() -> None:
    """Executing main.py like Kaggle does must not resolve training packages."""
    loader = """
import builtins
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
exec(compile(main_path.read_text(encoding="utf-8"), str(main_path), "exec"), namespace)
callables = [name for name, value in namespace.items() if callable(value)]
print(callables[-1])
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(SUBMISSION_SOURCE), str(SUBMISSION_SOURCE.parent))
    )
    result = subprocess.run(
        [sys.executable, "-c", loader, str(SUBMISSION_SOURCE / "main.py")],
        cwd=SUBMISSION_SOURCE,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "agent"


def test_entryfile_cannot_import_shadow_runtime_modules(tmp_path: Path) -> None:
    """Pre-existing generic module names cannot hijack bundled inference."""
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    for module_name in LOCAL_MODULES:
        (shadow / f"{module_name}.py").write_text(
            f'raise RuntimeError("shadow {module_name} imported")\n',
            encoding="utf-8",
        )
    loader = """
import pathlib
import sys

main_path = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, sys.argv[2])
sys.path.append(str(main_path.parent))
namespace = {}
exec(compile(main_path.read_text(encoding="utf-8"), str(main_path), "exec"), namespace)
for module_name in ("cg_api", "runtime"):
    print(pathlib.Path(sys.modules[module_name].__file__).resolve())
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            loader,
            str(SUBMISSION_SOURCE / "main.py"),
            str(shadow),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str((SUBMISSION_SOURCE / "cg_api.py").resolve()),
        str((SUBMISSION_SOURCE / "runtime.py").resolve()),
    ]

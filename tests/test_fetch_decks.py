import hashlib
import os
import subprocess
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_decks.sh"


def _release_assets(tmp_path: Path) -> Path:
    """Build a tiny release with the same top-level layout as production."""
    assets = tmp_path / "assets"
    source = tmp_path / "source"
    deck = source / "deck.csv"
    manifest = source / "manifest.json"
    assets.mkdir()
    source.mkdir()
    deck.write_text("\n".join("1" for _ in range(60)) + "\n", encoding="utf-8")
    manifest.write_text('{"decks": {}}\n', encoding="utf-8")

    archive = assets / "decks.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(
            deck,
            arcname="decks/heuristic-resolved/example-archetype/deck.csv",
        )
        bundle.add(
            manifest,
            arcname="decks/heuristic-resolved/manifest.json",
        )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (assets / "decks.sha256").write_text(f"{digest}  decks.tar.gz\n", encoding="utf-8")
    return assets


def _fake_gh(tmp_path: Path) -> Path:
    """Provide the release-download subset of gh used by fetch_decks.sh."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        """#!/bin/bash
set -euo pipefail
pattern=""
destination=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --pattern) pattern="$2"; shift 2 ;;
    --dir) destination="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cp "$ASSET_DIR/$pattern" "$destination/$pattern"
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return fake_bin


def test_fetch_decks_installs_release_beneath_explicit_root(tmp_path: Path) -> None:
    """--root keeps the corpus and release marker entirely on scratch storage."""
    assets = _release_assets(tmp_path)
    fake_bin = _fake_gh(tmp_path)
    scratch = tmp_path / "scratch" / "slopemon"
    env = {
        **os.environ,
        "ASSET_DIR": str(assets),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    first = subprocess.run(
        ["bash", str(FETCH_SCRIPT), "--root", str(scratch), "decks-v99"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    deck_dir = scratch / "decks"
    assert (
        deck_dir / "heuristic-resolved" / "example-archetype" / "deck.csv"
    ).is_file()
    marker = deck_dir / ".release-sha256"
    assert (
        marker.read_text(encoding="utf-8").strip()
        == hashlib.sha256((assets / "decks.tar.gz").read_bytes()).hexdigest()
    )
    assert f"decks in {deck_dir}" in first.stdout

    second = subprocess.run(
        ["bash", str(FETCH_SCRIPT), "decks-v99", f"--root={scratch}"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "nothing to do" in second.stdout


def test_fetch_decks_rejects_missing_root_value() -> None:
    """A malformed root option fails before gh or the filesystem is touched."""
    result = subprocess.run(
        ["bash", str(FETCH_SCRIPT), "--root"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--root needs a path" in result.stderr

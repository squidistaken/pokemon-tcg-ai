"""Contracts for the two CPU-only Habrok scraper jobs."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_slurm_log_directory_exists_in_a_fresh_checkout():
    assert (ROOT / "slurm-conf" / "logs").is_dir()


def _script(name: str) -> str:
    return (ROOT / "slurm-conf" / name).read_text(encoding="utf-8")


def test_discovery_is_a_true_12_hour_fetch_only_slurm_job():
    script = _script("discover_cards.sh")

    assert script.startswith("#!/bin/bash\n\n#SBATCH")
    assert "#SBATCH --partition=regular" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --gpus" not in script
    assert "python -m scraper.discovery" in script
    assert "--source all" in script
    assert "${SCRAPER_MAX_DECKS:-5000}" in script
    assert "${BULBAPEDIA_MAX_PAGES:-200}" in script
    assert "exec sbatch" not in script


def test_production_is_a_true_uncapped_48_hour_dual_strategy_job():
    script = _script("scrape_all.sh")

    assert script.startswith("#!/bin/bash\n\n#SBATCH")
    assert "#SBATCH --partition=regular" in script
    assert "#SBATCH --time=48:00:00" in script
    assert "#SBATCH --gpus" not in script
    assert "python -m scraper" in script
    assert "--source all" in script
    assert "--card-swap-strategy all" in script
    assert "--max-decks" not in script
    assert "${BULBAPEDIA_MAX_PAGES:-0}" in script
    assert "exec sbatch" not in script

"""Top-level selection of independently resolved deck corpora."""

from pathlib import Path

from hydra import compose, initialize_config_dir

CONF_DIR = Path(__file__).parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="ppo_selfplay_multideck", overrides=list(overrides))


def test_multideck_defaults_to_heuristic_corpus():
    cfg = _compose()

    assert cfg.deck_corpus == "heuristic-resolved"
    assert cfg.env.deck_pool == "decks/heuristic-resolved"


def test_multideck_can_select_mapping_corpus_at_top_level():
    cfg = _compose("deck_corpus=mapping-resolved")

    assert cfg.env.deck_pool == "decks/mapping-resolved"

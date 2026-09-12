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


def _compose_env(env: str, *overrides: str):
    """Compose the base config with an env group selected."""
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=[f"env={env}", *overrides])


def test_v2_pools_resolve_under_the_selected_corpus():
    """
    The per-archetype v2 pools must include the ``deck_corpus`` segment.
    """
    for env in ("multideck_v2", "curriculum_v2"):
        pool = _compose_env(env).env.deck_pool
        assert pool, f"{env} has no deck_pool"
        for entry in pool:
            assert entry.startswith("decks/heuristic-resolved/"), (
                f"{env} entry {entry!r} does not resolve under the corpus"
            )


def test_v2_pools_follow_a_corpus_override():
    for env in ("multideck_v2", "curriculum_v2"):
        pool = _compose_env(env, "deck_corpus=mapping-resolved").env.deck_pool
        for entry in pool:
            assert entry.startswith("decks/mapping-resolved/")

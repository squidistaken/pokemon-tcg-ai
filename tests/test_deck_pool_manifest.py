from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parents[1]
CORPUS_DIR = REPO_ROOT / "decks"
MULTIDECK_V2 = REPO_ROOT / "conf" / "env" / "multideck_v2.yaml"

# The scraped corpus is a pulled release artifact, not committed, so this test
# skips where it is absent (CI checks out only the committed example deck).
_HAS_CORPUS = bool(list(CORPUS_DIR.glob("*/*.csv"))) if CORPUS_DIR.is_dir() else False
requires_corpus = pytest.mark.skipif(
    not _HAS_CORPUS,
    reason="scraped deck corpus not installed; run ./scripts/fetch_decks.sh",
)

#: The rule multideck_v2's hardcoded pool is meant to encode: an archetype
#: earns a place once it has enough lists for its matchups to be scoreable.
MIN_LISTS_PER_ARCHETYPE = 8


def _configured_pool() -> list[str]:
    """
    Read the deck pool pinned in ``conf/env/multideck_v2.yaml``.

    :return: The configured archetype directories, sorted.
    """
    config = OmegaConf.load(MULTIDECK_V2)
    return sorted(str(entry) for entry in config["deck_pool"])


def _archetypes_meeting_rule() -> list[str]:
    """
    Recompute the pool from the corpus using the documented threshold.

    :return: ``decks/<archetype>`` for every archetype with at least
        :data:`MIN_LISTS_PER_ARCHETYPE` lists, sorted.
    """
    return sorted(
        f"decks/{directory.name}"
        for directory in CORPUS_DIR.iterdir()
        if directory.is_dir()
        and len(list(directory.glob("*.csv"))) >= MIN_LISTS_PER_ARCHETYPE
    )


@requires_corpus
def test_multideck_v2_pool_matches_the_archetype_threshold() -> None:
    """
    The pinned pool still equals the rule its header claims to encode.

    The list is hardcoded on purpose: it is the reproducibility artifact that
    keeps A/B arms comparable across corpus updates, which an auto-scan at load
    time would silently break. The cost of pinning is that it can drift out of
    sync with the ">=8 lists" rule as decks are added, so the rule is recomputed
    here instead. A failure means the corpus changed: re-pin the pool
    deliberately (and treat runs before and after as different experiments),
    rather than assuming the old list still means what the header says.
    """
    configured = _configured_pool()
    derived = _archetypes_meeting_rule()

    missing = sorted(set(derived) - set(configured))
    extra = sorted(set(configured) - set(derived))
    assert configured == derived, (
        f"conf/env/multideck_v2.yaml deck_pool has drifted from the "
        f">={MIN_LISTS_PER_ARCHETYPE}-lists rule. "
        f"Newly qualifying and absent from the config: {missing or 'none'}. "
        f"Configured but no longer qualifying: {extra or 'none'}."
    )


@requires_corpus
def test_multideck_v2_pool_is_the_documented_size() -> None:
    """
    The pool is the 19 archetypes the config header and design doc quote.

    Pinned separately from the rule check so a corpus change that happens to
    keep the count while swapping an archetype still fails the test above.
    """
    assert len(_configured_pool()) == 19

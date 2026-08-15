import csv
import re
from pathlib import Path

import torch

_DATA_PATH = Path(__file__).resolve().parents[3] / "scraper" / "EN_Card_Data.csv"

_DRAW_COUNT = re.compile(r"draw (\d+) card")
_DAMAGE_COUNTERS = re.compile(r"(\d+) damage counter")
_SCALING_DAMAGE = re.compile(r"[×x+]\s*$|[×x]\s*\d")


class CardEffectFeatures:
    """
    Per-card effect features parsed from the competition's card text.

    The engine exposes only card IDs (``cg.api.Card``), so a model that sees
    nothing but an ID plus static stats has to learn every card's behaviour
    from reward. This reads ``scraper/EN_Card_Data.csv``, whose
    ``Effect Explanation`` column carries the printed text, and turns it into
    a fixed-width numeric row per card ID: what the card draws, searches,
    discards, heals, moves and inflicts.

    A card can occupy several rows in the source (one per ability and attack).
    All of its rows are merged, taking the maximum of each feature, so a
    Pokemon whose ability draws and whose attack places damage counters scores
    on both.
    """

    FEATURE_NAMES: tuple[str, ...] = (
        "draw_count",
        "draws",
        "searches_deck",
        "puts_to_bench",
        "puts_to_hand",
        "heals",
        "discards_own",
        "discards_opponent",
        "switches_opponent",
        "switches_own",
        "damage_counters",
        "scales_per_each",
        "inflicts_status",
        "attaches_energy",
        "shuffles_hand",
        "coin_flip",
        "prevents_damage",
        "has_ability",
        "damage_scales",
        "mentions_knockout",
        "looks_at_deck",
        "returns_from_discard",
        "text_length",
        "has_text",
    )

    def __init__(self, n_rows: int, data_path: Path | None = None) -> None:
        """
        Parse the card text file into a card-ID-indexed feature table.

        :param n_rows: Number of rows to allocate, matching the card tables it
            is concatenated onto (``max_card_id + 1``).
        :param data_path: Override for the card text CSV, for tests.
        :raises FileNotFoundError: If the card text file is missing.
        """
        path = data_path or _DATA_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"Card effect features need {path}, which does not exist. "
                "Disable model.adapter.card_effect_features to run without it."
            )
        self._features = torch.zeros(n_rows, len(self.FEATURE_NAMES))
        with path.open(encoding="utf-8-sig") as handle:
            for record in csv.DictReader(handle):
                try:
                    card_id = int(record["Card ID"])
                except (TypeError, ValueError):
                    continue
                if not 0 <= card_id < n_rows:
                    continue
                row = self._row(record)
                self._features[card_id] = torch.maximum(self._features[card_id], row)

    @classmethod
    def _row(cls, record: dict[str, str]) -> torch.Tensor:
        """
        Turn one source row into its feature vector.

        :param record: One ``EN_Card_Data.csv`` row.
        :return: Float tensor of shape ``(len(FEATURE_NAMES),)``.
        """
        text = (record.get("Effect Explanation") or "").lower()
        move = (record.get("Move Name") or "").lower()
        damage = record.get("Damage") or ""
        draw_match = _DRAW_COUNT.search(text)
        counter_match = _DAMAGE_COUNTERS.search(text)
        opponent = "opponent" in text
        values = (
            float(draw_match.group(1)) if draw_match else 0.0,
            float("draw" in text),
            float("search your deck" in text),
            float("onto your bench" in text or "on your bench" in text),
            float("into your hand" in text),
            float("heal" in text),
            float("discard" in text and not opponent),
            float("discard" in text and opponent),
            float("switch" in text and opponent),
            float("switch" in text and not opponent),
            float(counter_match.group(1)) if counter_match else 0.0,
            float("for each" in text),
            float(
                any(
                    condition in text
                    for condition in (
                        "asleep",
                        "paralyzed",
                        "confused",
                        "poisoned",
                        "burned",
                    )
                )
            ),
            float("attach" in text and "energy" in text),
            float("shuffle your hand" in text),
            float("flip a coin" in text),
            float("prevent" in text or "less damage" in text),
            float("[ability]" in move),
            float(bool(_SCALING_DAMAGE.search(damage))),
            float("knocked out" in text),
            float("look at the top" in text),
            float("from your discard pile" in text),
            min(len(text) / 200.0, 3.0),
            float(bool(text.strip())),
        )
        return torch.tensor(values, dtype=torch.float32)

    @property
    def features(self) -> torch.Tensor:
        """
        Effect features per card ID, aligned with ``FEATURE_NAMES``.

        :return: Float32 tensor of shape ``(n_rows, len(FEATURE_NAMES))``.
        """
        return self._features

    @property
    def coverage(self) -> float:
        """
        Fraction of allocated rows that carry any parsed text.

        :return: Value in ``[0, 1]``, for logging and sanity checks.
        """
        has_text = self._features[:, self.FEATURE_NAMES.index("has_text")]
        return float(has_text.mean())

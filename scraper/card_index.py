from __future__ import annotations

import csv
import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import NamedTuple

FUZZY_THRESHOLD = 0.90

_POWER_MARKERS = {
    "mega",
    "ex",
    "gx",
    "v",
    "vmax",
    "vstar",
    "tera",
    "prime",
    "radiant",
    "shiny",
}

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CARD_CSV = os.path.join(_PKG_DIR, "EN_Card_Data.csv")

COL_ID = "Card ID"
COL_NAME = "Card Name"
COL_SET = "Expansion"
COL_NUMBER = "Collection No."
COL_STAGE = "Stage (Pokémon)/Type (Energy and Trainer)"
COL_RULE = "Rule"

_ENERGY_WORDS = {
    "G": ["grass"],
    "R": ["fire"],
    "W": ["water"],
    "L": ["lightning"],
    "P": ["psychic"],
    "F": ["fighting"],
    "D": ["darkness", "dark"],
    "M": ["metal"],
}

_GLUED_MARKER_RE = re.compile(r"([a-z])(EX|GX)\b")


def normalize_name(name: str) -> str:
    """Canonicalize a card name for lookup.

    Lowercases, unifies apostrophes, folds accents (é -> e), splits glued EX/GX
    markers, drops trademark/special-character noise, and collapses whitespace.

    :param name: Raw card name as scraped (or None).
    :return: The normalized name, or "" when ``name`` is None.
    """
    if name is None:
        return ""
    s = _GLUED_MARKER_RE.sub(r"\1 \2", name)  # "LugiaEX" -> "Lugia EX"
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'").replace("‘", "'")
    s = s.replace("é", "e")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9'&{}()\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_number(number: str | None) -> str | None:
    """Canonicalize a collection number ("007" -> "7", "TG12" -> "tg12").

    :param number: Raw collection number (or None).
    :return: The normalized number, or None when absent/empty.
    """
    if number is None:
        return None
    n = str(number).strip().lower()
    if not n:
        return None
    # Purely numeric -> drop leading zeros; otherwise keep as-is (promos etc.).
    return str(int(n)) if n.isdigit() else n


def _power_markers(normalized: str) -> frozenset[str]:
    """
    :param normalized: A normalized card name (see :func:`normalize_name`).
    :return: The power-level markers (mega/ex/gx/...) present in the name.
    """
    return frozenset(t for t in normalized.split() if t in _POWER_MARKERS)


@dataclass
class CardInfo:
    """One card's identity and the per-ID flags the deck validator needs."""

    card_id: int
    name: str
    set_code: str
    number: str | None
    is_basic_energy: bool
    is_basic_pokemon: bool
    is_ace_spec: bool


class MatchResult(NamedTuple):
    """Outcome of resolving one scraped card, for auditability."""

    card_id: int | None
    method: str  # "exact" | "energy" | "fuzzy" | "unresolved"
    matched_name: str | None = None
    score: float | None = None  # similarity ratio for fuzzy matches


class CardIndex:
    """Indexed view of ``EN_Card_Data.csv``."""

    def __init__(self, csv_path: str = DEFAULT_CARD_CSV):
        """
        Load and index the card database.

        :param csv_path: Path to ``EN_Card_Data.csv`` (defaults to the repo copy).
        """
        self.csv_path = csv_path
        self.by_id: dict[int, CardInfo] = {}
        self.available_sets: set[str] = set()
        self._by_name_set_no: dict[tuple[str, str, str | None], int] = {}
        self._by_name_set: dict[tuple[str, str], list[int]] = {}
        self._by_name: dict[str, list[int]] = {}
        self._names_by_set: dict[str, set[str]] = {}  # set_lower -> normalized names
        self._energy_alias: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        """
        Read every card row into ``by_id`` and the name/set lookup tables.

        Rows without a numeric ID are skipped; repeated IDs (multi-attack cards)
        keep the first row seen.
        """
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw_id = (row.get(COL_ID) or "").strip()
                if not raw_id.isdigit():
                    continue
                cid = int(raw_id)
                if cid in self.by_id:
                    continue  # multi-attack cards repeat the ID; keep the first row

                stage = (row.get(COL_STAGE) or "").strip()
                rule = (row.get(COL_RULE) or "").strip()
                name = (row.get(COL_NAME) or "").strip()
                set_code = (row.get(COL_SET) or "").strip()
                number = normalize_number(row.get(COL_NUMBER))

                info = CardInfo(
                    card_id=cid,
                    name=name,
                    set_code=set_code,
                    number=number,
                    is_basic_energy=(stage == "Basic Energy"),
                    is_basic_pokemon=(stage == "Basic Pokémon"),
                    is_ace_spec=(rule == "ACE SPEC"),
                )
                self.by_id[cid] = info
                self._index_card(info)

        self._build_energy_aliases()

    def _index_card(self, info: CardInfo) -> None:
        """
        Add one card to the normalized name/(name, set)/(name, set, number)
        lookup tables.

        :param info: The card to index.
        """
        nm = normalize_name(info.name)
        if info.set_code:
            self.available_sets.add(info.set_code)
            skey = info.set_code.lower()
            self._by_name_set_no.setdefault((nm, skey, info.number), info.card_id)
            self._by_name_set.setdefault((nm, skey), []).append(info.card_id)
            self._names_by_set.setdefault(skey, set()).add(nm)
        self._by_name.setdefault(nm, []).append(info.card_id)

    def _build_energy_aliases(self) -> None:
        """Map decklist energy names."""
        for info in self.by_id.values():
            if not info.is_basic_energy:
                continue
            m = re.search(r"\{([A-Z])\}", info.name)
            if not m:
                continue
            letter = m.group(1)
            words = _ENERGY_WORDS.get(letter, [])
            keys = {f"{{{letter.lower()}}}"}
            for w in words:
                keys.add(f"{w} energy")
                keys.add(f"basic {w} energy")
            for key in keys:
                self._energy_alias.setdefault(normalize_name(key), info.card_id)

    def resolve_id(
        self,
        name: str,
        set_code: str | None = None,
        number: str | None = None,
    ) -> int | None:
        """Resolve one card to a Card ID, or ``None`` if absent.

        :param name: Scraped card name.
        :param set_code: Scraped set code, if known.
        :param number: Scraped collection number, if known.
        :return: The matched Card ID, or None if unresolved.
        """
        return self.match(name, set_code, number).card_id

    def match(
        self,
        name: str,
        set_code: str | None = None,
        number: str | None = None,
    ) -> MatchResult:
        """Resolve one scraped card to a Card ID.

        :param name: Scraped card name.
        :param set_code: Scraped set code, if known.
        :param number: Scraped collection number, if known.
        :return: A :class:`MatchResult` recording the Card ID (or None) and how
            it was matched.
        """
        nm = normalize_name(name)

        # Basic energies: match by aliased type name regardless of set/number.
        if nm in self._energy_alias:
            return MatchResult(self._energy_alias[nm], "energy", nm)

        skey = set_code.lower().strip() if set_code else None
        num = normalize_number(number)

        if skey and num is not None:
            hit = self._by_name_set_no.get((nm, skey, num))
            if hit is not None:
                return MatchResult(hit, "exact", nm)
        if skey:
            hits = self._by_name_set.get((nm, skey))
            if hits:
                return MatchResult(hits[0], "exact", nm)
        # Reprint fallback: same name, any set (prefer a same-number printing).
        hits = self._by_name.get(nm)
        if hits:
            if num is not None:
                for cid in hits:
                    if self.by_id[cid].number == num:
                        return MatchResult(cid, "exact", nm)
            return MatchResult(hits[0], "exact", nm)

        return self._fuzzy_match(nm, skey)

    def _fuzzy_match(self, nm: str, skey: str | None) -> MatchResult:
        """Guarded fuzzy fallback. Three guards prevent false positives:

        1. set-availability: Card whose set we don't have is genuinely
          absent, so don't fuzz it;
        2. power markers: both names must carry the identical set of markers, 
           so distinct power levels never collapse together;
        3. similarity: ratio must reach ``FUZZY_THRESHOLD``.

        :param nm: Already-normalized scraped card name.
        :param skey: Lower-cased scraped set code, if known.
        :return: A fuzzy :class:`MatchResult`, or an unresolved one if no
            candidate clears the guards.
        """
        if skey is not None and skey not in {s.lower() for s in self.available_sets}:
            return MatchResult(None, "unresolved")

        # Restrict candidates to cards printed in the scraped set when known.
        candidates = self._names_by_set.get(skey) if skey else self._by_name.keys()
        if not candidates:
            candidates = self._by_name.keys()

        want_markers = _power_markers(nm)
        best_name: str | None = None
        best_score = 0.0
        for cand in candidates:
            if _power_markers(cand) != want_markers:
                continue
            score = SequenceMatcher(None, nm, cand).ratio()
            if score > best_score:
                best_score, best_name = score, cand

        if best_name is not None and best_score >= FUZZY_THRESHOLD:
            return MatchResult(
                self._by_name[best_name][0], "fuzzy", best_name, round(best_score, 3)
            )
        return MatchResult(None, "unresolved")

"""Offline review report for unresolved Limitless card printings.

This command writes only to outputs/. External metadata never enters deck files,
the competition CSV, or training.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from bs4 import BeautifulSoup

from .card_index import CardIndex, CardProfile, normalize_name, normalize_number
from .http import HttpClient
from .models import RawCard
from .sources.limitless import LimitlessSource

DEFAULT_OUTPUT = Path("outputs/card_swap_audit")
CARD_URL = "https://limitlesstcg.com/cards/{set_code}/{number}"


def _category(profile: CardProfile) -> str:
    if "Pokémon" in profile.stage:
        return "pokemon"
    if "Energy" in profile.stage:
        return "energy"
    return "trainer"


def _page_text(html: str) -> str:
    return BeautifulSoup(html, "lxml").get_text(" ", strip=True)


def _pokemon_stage(text: str) -> str | None:
    match = re.search(r"Pok[eé]mon\s+-\s+(Basic|Stage\s+[12])", text, re.IGNORECASE)
    if not match:
        return None
    stage = match.group(1).title()
    return "Basic Pokémon" if stage == "Basic" else f"{stage} Pokémon"


def _rank(card: RawCard, text: str, index: CardIndex) -> list[dict[str, object]]:
    same_name = set(index.candidates_for_name(card.name))
    source_stage = _pokemon_stage(text)
    source_hp = re.search(r"(\d+)\s+HP", text)
    ranked: list[tuple[float, CardProfile]] = []
    for profile in index.profiles.values():
        if same_name and profile.card_id not in same_name:
            continue
        if card.category and _category(profile) != card.category:
            continue
        if source_stage and profile.stage != source_stage:
            continue
        gameplay = SequenceMatcher(
            None,
            normalize_name(text),
            normalize_name(profile.searchable_text),
        ).ratio()
        name = float(normalize_name(card.name) == normalize_name(profile.name))
        hp = 0.5
        if source_hp and profile.hp is not None:
            hp = max(0.0, 1.0 - abs(int(source_hp.group(1)) - profile.hp) / 100)
        ranked.append((0.65 * gameplay + 0.25 * name + 0.10 * hp, profile))
    ranked.sort(key=lambda item: (-item[0], item[1].card_id))
    return [
        {
            "card_id": profile.card_id,
            "name": profile.name,
            "score": round(score, 4),
        }
        for score, profile in ranked[:5]
    ]


def _fetch(card: RawCard, client: HttpClient, cache: Path) -> str:
    if not card.set_code or not card.number:
        raise ValueError("source set and number are required")
    path = cache / (
        f"{card.set_code.upper()}-{normalize_number(card.number) or 'unknown'}.html"
    )
    if path.exists():
        return path.read_text(encoding="utf-8")
    html = client.get_text(
        CARD_URL.format(set_code=card.set_code.upper(), number=card.number)
    )
    path.write_text(html, encoding="utf-8")
    return html


def run_audit(args: argparse.Namespace) -> list[dict[str, object]]:
    """Fetch missing source printings and rank competition-card candidates."""
    output = Path(args.out)
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    client = HttpClient(min_interval=args.card_request_interval)
    index = CardIndex()
    source = LimitlessSource(client=client)
    occurrences: Counter[tuple[str, str, str, str]] = Counter()
    copies: Counter[tuple[str, str, str, str]] = Counter()
    samples: dict[tuple[str, str, str, str], RawCard] = {}

    for deck in source.iter_decks(
        limit=args.limit,
        per_tournament=args.per_tournament,
        page=args.page,
        max_pages=args.max_pages,
        max_decks=args.max_decks,
        since=args.since,
        until=args.until,
        fmt=args.format,
        verbose=args.verbose,
    ):
        for card in deck.cards:
            match = index.match(card.name, card.set_code, card.number)
            if match.method not in {"ambiguous", "unresolved"}:
                continue
            key = (
                normalize_name(card.name),
                (card.set_code or "").upper(),
                normalize_number(card.number) or "",
                card.category or "",
            )
            occurrences[key] += 1
            copies[key] += max(0, card.count)
            samples.setdefault(key, card)

    report: list[dict[str, object]] = []
    for key, frequency in occurrences.most_common():
        card = samples[key]
        entry: dict[str, object] = {
            "source": {
                "name": card.name,
                "set": card.set_code,
                "number": normalize_number(card.number),
                "category": card.category,
            },
            "deck_occurrences": frequency,
            "copies": copies[key],
        }
        try:
            entry["candidates"] = _rank(
                card, _page_text(_fetch(card, client, cache)), index
            )
        except Exception as exc:  # noqa: BLE001 - keep the rest of the audit useful
            entry["candidates"] = []
            entry["error"] = str(exc)
        report.append(entry)

    (output / "report.json").write_text(
        json.dumps({"schema_version": 1, "cards": report}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    with open(output / "report.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["name", "set", "number", "occurrences", "copies", "target_id", "score"]
        )
        for entry in report:
            source_card = entry["source"]
            assert isinstance(source_card, dict)
            candidates = entry["candidates"]
            top = candidates[0] if isinstance(candidates, list) and candidates else {}
            writer.writerow(
                [
                    source_card.get("name"),
                    source_card.get("set"),
                    source_card.get("number"),
                    entry["deck_occurrences"],
                    entry["copies"],
                    top.get("card_id"),
                    top.get("score"),
                ]
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank competition-card substitutes for unresolved printings"
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--per-tournament", type=int, default=8)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--max-decks", type=int)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--format", default="standard")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--card-request-interval", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(args)
    print(f"Wrote {len(report)} unresolved printing(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

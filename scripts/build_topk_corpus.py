"""
Build a small deck corpus from the K most-observed lists of a larger one.

``deck_pool_width`` cuts at the archetype level and keeps every list inside the
archetypes it keeps, so width 8 still leaves 10,756 of 28,670 lists. The waste
is at the list level: lists in one archetype differ by a median of 13 cards out
of 60, and 76.7% of the corpus appears exactly once in tournament data. This
script cuts there instead, writing a corpus that holds K lists in the same
``<archetype>/<list>.csv`` layout plus a ``manifest.json`` carrying their
``observation_count``, so ``env.deck_weighting=observation`` still deals them in
proportion.

Run from the repository root::

    uv run python scripts/build_topk_corpus.py --k 20
    uv run python scripts/build_topk_corpus.py --k 20 --require alakazam-dudunsparce-4
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "decks/heuristic-resolved",
        help="Corpus to select from (default: decks/heuristic-resolved).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output corpus directory (default: decks/top<K>).",
    )
    parser.add_argument("--k", type=int, default=20, help="How many lists to keep.")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="List stem that must be included even if it falls outside the top K. "
        "Repeatable.",
    )
    return parser.parse_args()


def observation_weight(entry: dict) -> float:
    """
    Turn a manifest ``observation_count`` into the sampler's weight.

    Mirrors ``_observation_weight`` in src/training/env_factory.py.

    :param entry: One manifest deck record.
    :return: The weight, at least 1.0.
    """
    count = entry.get("observation_count")
    return float(count) if isinstance(count, int) and count > 0 else 1.0


def main() -> int:
    """
    Write the top-K corpus and report what went into it.

    :return: Process exit status.
    """
    args = parse_args()
    dest = args.dest or REPO_ROOT / f"decks/top{args.k}"
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    decks = source_manifest["decks"]

    ranked = sorted(decks.items(), key=lambda kv: (-observation_weight(kv[1]), kv[0]))
    chosen: dict[str, dict] = dict(ranked[: args.k])
    for stem in args.require:
        if stem in chosen:
            continue
        if stem not in decks:
            print(f"ERROR: --require {stem} is not in {args.source}", file=sys.stderr)
            return 1
        chosen[stem] = decks[stem]
        print(f"forced in: {stem} (weight {observation_weight(decks[stem]):.0f})")

    if dest.exists():
        shutil.rmtree(dest)
    for stem, entry in chosen.items():
        relative = Path(entry["file"])
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / relative, target)

    manifest = dict(source_manifest)
    manifest["decks"] = chosen
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(observation_weight(entry) for entry in chosen.values())
    archetypes = sorted({entry["file"].split("/")[0] for entry in chosen.values()})
    print(f"\n{dest.relative_to(REPO_ROOT)}: {len(chosen)} lists, {len(archetypes)} archetypes")
    print(f"total observation weight {total:.0f}\n")
    print(f"{'list':34s} {'archetype':28s} {'wt':>5s} {'draw share':>11s}")
    for stem, entry in sorted(
        chosen.items(), key=lambda kv: -observation_weight(kv[1])
    ):
        weight = observation_weight(entry)
        print(
            f"{stem:34s} {entry['file'].split('/')[0]:28s} "
            f"{weight:5.0f} {100 * weight / total:10.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

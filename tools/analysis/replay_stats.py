"""
Measure play quality from downloaded Kaggle replays.

Reconstructs per-episode board statistics from the raw replay JSON: prize
progress, energy on board, deck exhaustion and bench development, split by
whether our agent won or lost.
"""
import collections
import glob
import json
import os
import statistics
import sys


def episode_stats(path: str, our_index: int) -> dict:
    data = json.load(open(path))
    steps = data["steps"]
    ours_energy, opp_energy = [], []
    ours_prizes, opp_prizes = None, None
    ours_deck, opp_deck = None, None
    ours_bench = []
    for step in steps:
        obs = step[0]["observation"]
        cur = obs.get("current")
        if not cur or not cur.get("players"):
            continue
        players = cur["players"]
        if len(players) < 2:
            continue
        me, opp = players[our_index], players[1 - our_index]
        def energy_count(p):
            total = 0
            for mon in [m for m in (p.get("active") or []) + (p.get("bench") or []) if m]:
                total += len(mon.get("energies") or [])
            return total
        ours_energy.append(energy_count(me))
        opp_energy.append(energy_count(opp))
        ours_prizes = len(me.get("prize") or [])
        opp_prizes = len(opp.get("prize") or [])
        ours_deck = me.get("deckCount")
        opp_deck = opp.get("deckCount")
        ours_bench.append(len(me.get("bench") or []))
    return {
        "steps": len(steps),
        "our_energy_max": max(ours_energy) if ours_energy else 0,
        "our_energy_final": ours_energy[-1] if ours_energy else 0,
        "opp_energy_max": max(opp_energy) if opp_energy else 0,
        "opp_energy_final": opp_energy[-1] if opp_energy else 0,
        "our_prizes_left": ours_prizes,
        "opp_prizes_left": opp_prizes,
        "our_deck_left": ours_deck,
        "opp_deck_left": opp_deck,
        "our_bench_max": max(ours_bench) if ours_bench else 0,
    }


def main() -> None:
    subs = sys.argv[1:] or sorted(os.listdir("logs/replays"))
    for sub in subs:
        manifest_path = f"logs/replays/{sub}/manifest.json"
        if not os.path.exists(manifest_path):
            continue
        manifest = json.load(open(manifest_path))
        rows = []
        for episode_id, meta in manifest.items():
            path = f"logs/replays/{sub}/episode-{episode_id}-replay.json"
            if not os.path.exists(path):
                continue
            stats = episode_stats(path, meta["our_index"])
            stats["result"] = meta["result"]
            rows.append(stats)
        if not rows:
            continue
        wins = [r for r in rows if r["result"] == "win"]
        losses = [r for r in rows if r["result"] == "loss"]
        def avg(rs, k):
            vals = [r[k] for r in rs if r[k] is not None]
            return statistics.mean(vals) if vals else float("nan")
        print(f"\n=== {sub}  n={len(rows)}  W{len(wins)}/L{len(losses)} ===")
        print(f"{'metric':22} {'win':>8} {'loss':>8} {'all':>8}")
        for key in ("our_energy_max", "opp_energy_max", "our_energy_final",
                    "our_prizes_left", "opp_prizes_left",
                    "our_deck_left", "opp_deck_left", "our_bench_max", "steps"):
            print(f"{key:22} {avg(wins,key):>8.2f} {avg(losses,key):>8.2f} {avg(rows,key):>8.2f}")
        decked = sum(1 for r in rows if (r["our_deck_left"] or 1) == 0)
        print(f"our deck-outs: {decked}/{len(rows)}")


if __name__ == "__main__":
    main()

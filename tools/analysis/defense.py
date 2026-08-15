"""
Measure defensive play for both seats across the downloaded Kaggle replays.

Distinguishes a voluntary switch (the active Pokemon changes while the old one
is still alive and on the bench) from a forced one (the old active is gone, so
it was knocked out). Also tracks how long each seat leaves a badly damaged
Pokemon in the active spot, which is what "never defends" would look like.
"""
import glob
import json
import os
import statistics


def scan(path: str, our_index: int) -> dict:
    """
    Walk one replay and tally switch and damage exposure for both seats.

    :param path: Replay JSON path.
    :param our_index: Seat our agent played.
    :return: Per-seat counters keyed "ours" and "opp".
    """
    data = json.load(open(path))
    seats = {"ours": our_index, "opp": 1 - our_index}
    prev_active = {"ours": None, "opp": None}
    prev_bench_serials = {"ours": set(), "opp": set()}
    stats = {
        side: {
            "voluntary": 0,
            "forced": 0,
            "decisions": 0,
            "low_hp_active": 0,
            "active_seen": 0,
        }
        for side in seats
    }
    for step in data["steps"]:
        current = step[0]["observation"].get("current")
        if not current or len(current.get("players") or []) < 2:
            continue
        for side, seat in seats.items():
            player = current["players"][seat]
            actives = [m for m in (player.get("active") or []) if m]
            bench = [m for m in (player.get("bench") or []) if m]
            if not actives:
                continue
            active = actives[0]
            serial = active.get("serial")
            stats[side]["active_seen"] += 1
            if active.get("maxHp"):
                if active["hp"] <= 0.4 * active["maxHp"]:
                    stats[side]["low_hp_active"] += 1
            previous = prev_active[side]
            if previous is not None and serial != previous:
                # The old active survived if it is now sitting on the bench.
                if previous in {m.get("serial") for m in bench}:
                    stats[side]["voluntary"] += 1
                else:
                    stats[side]["forced"] += 1
            prev_active[side] = serial
            prev_bench_serials[side] = {m.get("serial") for m in bench}
    return stats


def main() -> None:
    totals = {
        side: {"voluntary": 0, "forced": 0, "low_hp_active": 0, "active_seen": 0}
        for side in ("ours", "opp")
    }
    games = 0
    for sub in sorted(os.listdir("logs/replays")):
        manifest_path = f"logs/replays/{sub}/manifest.json"
        if not os.path.exists(manifest_path):
            continue
        for episode_id, meta in json.load(open(manifest_path)).items():
            path = f"logs/replays/{sub}/episode-{episode_id}-replay.json"
            if not os.path.exists(path):
                continue
            games += 1
            result = scan(path, meta["our_index"])
            for side in totals:
                for key in totals[side]:
                    totals[side][key] += result[side][key]

    print(f"games: {games}\n")
    print(f"{'metric':40} {'our agent':>12} {'opponents':>12}")
    for label, key in (
        ("voluntary switches (retreats)", "voluntary"),
        ("forced switches (knockouts taken)", "forced"),
    ):
        print(f"{label:40} {totals['ours'][key]:>12} {totals['opp'][key]:>12}")
    for side in ("ours", "opp"):
        total = totals[side]["voluntary"] + totals[side]["forced"]
        totals[side]["retreat_share"] = (
            totals[side]["voluntary"] / total if total else float("nan")
        )
        totals[side]["low_share"] = (
            totals[side]["low_hp_active"] / max(totals[side]["active_seen"], 1)
        )
    print(f"{'share of switches that were voluntary':40} "
          f"{totals['ours']['retreat_share']:>12.2f} {totals['opp']['retreat_share']:>12.2f}")
    print(f"{'retreats per knockout taken':40} "
          f"{totals['ours']['voluntary']/max(totals['ours']['forced'],1):>12.2f} "
          f"{totals['opp']['voluntary']/max(totals['opp']['forced'],1):>12.2f}")
    print(f"{'time active is under 40% HP':40} "
          f"{totals['ours']['low_share']:>12.2f} {totals['opp']['low_share']:>12.2f}")


if __name__ == "__main__":
    main()

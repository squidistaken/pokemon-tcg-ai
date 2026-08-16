"""
Measure whether the agent assembles the Alakazam deck's win condition.

alakazam-dudunsparce-4 wins with Alakazam (card 743, a Stage 2 reached through
Kadabra 742 or 3x Rare Candy 1079) using Powerful Hand, whose damage scales
with the number of cards held. This walks the downloaded replays and reports
how often Alakazam reaches the board, how large the hand is, and how the
opponent's board compares.
"""
import collections
import glob
import json
import os
import statistics

ALAKAZAM, KADABRA, ABRA = 743, 742, 741


def scan(path: str, our_index: int) -> dict:
    data = json.load(open(path))
    hand_sizes, our_board_hp, opp_board_hp = [], [], []
    alakazam_step = None
    kadabra_step = None
    max_prizes_taken = 0
    for index, step in enumerate(data["steps"]):
        cur = step[0]["observation"].get("current")
        if not cur or len(cur.get("players") or []) < 2:
            continue
        me = cur["players"][our_index]
        opp = cur["players"][1 - our_index]
        mons = [m for m in (me.get("active") or []) + (me.get("bench") or []) if m]
        ids = {m["id"] for m in mons}
        if ALAKAZAM in ids and alakazam_step is None:
            alakazam_step = index
        if KADABRA in ids and kadabra_step is None:
            kadabra_step = index
        if me.get("handCount") is not None:
            hand_sizes.append(me["handCount"])
        our_board_hp.append(sum(m["hp"] for m in mons))
        opp_mons = [m for m in (opp.get("active") or []) + (opp.get("bench") or []) if m]
        opp_board_hp.append(sum(m["hp"] for m in opp_mons))
        max_prizes_taken = max(max_prizes_taken, 6 - len(opp.get("prize") or []))
    return {
        "alakazam": alakazam_step is not None,
        "alakazam_step": alakazam_step,
        "kadabra": kadabra_step is not None,
        "hand_mean": statistics.mean(hand_sizes) if hand_sizes else 0,
        "hand_max": max(hand_sizes) if hand_sizes else 0,
        "prizes_taken": max_prizes_taken,
        "steps": len(data["steps"]),
    }


def main() -> None:
    subs = {"55431769": "76.5M", "55480315": "150M", "55491703": "158M pinned"}
    for sub, label in subs.items():
        manifest = json.load(open(f"logs/replays/{sub}/manifest.json"))
        rows = []
        for episode_id, meta in manifest.items():
            path = f"logs/replays/{sub}/episode-{episode_id}-replay.json"
            if not os.path.exists(path):
                continue
            row = scan(path, meta["our_index"])
            row["result"] = meta["result"]
            rows.append(row)
        n = len(rows)
        ala = sum(r["alakazam"] for r in rows)
        kad = sum(r["kadabra"] for r in rows)
        print(f"\n=== {sub} ({label})  n={n} ===")
        print(f"  games where Alakazam reached the board: {ala}/{n}")
        print(f"  games where Kadabra reached the board:  {kad}/{n}")
        print(f"  mean hand size:  {statistics.mean(r['hand_mean'] for r in rows):.2f}")
        print(f"  max hand size:   {statistics.mean(r['hand_max'] for r in rows):.2f}")
        print(f"  prizes taken:    {statistics.mean(r['prizes_taken'] for r in rows):.2f} / 6")
        for res in ("win", "loss"):
            sel = [r for r in rows if r["result"] == res]
            if not sel:
                continue
            print(f"    {res:5} n={len(sel)}  alakazam {sum(r['alakazam'] for r in sel)}/{len(sel)}"
                  f"  prizes {statistics.mean(r['prizes_taken'] for r in sel):.2f}"
                  f"  hand {statistics.mean(r['hand_mean'] for r in sel):.2f}")


if __name__ == "__main__":
    main()

"""
Compare hand size and prize race across the Alakazam submissions.

Reads the final populated prize arrays rather than a running maximum, so the
prize counts reflect how the game actually ended, and pools wins and losses
across submissions to give the hand-size comparison a usable sample.
"""
import json, os, statistics

SUBS = {"55431769": "76.5M weighted", "55480315": "150M weighted", "55491703": "158M pinned"}


def scan(path: str, our_index: int) -> dict:
    data = json.load(open(path))
    hands, our_prize, opp_prize, turns = [], None, None, 0
    for step in data["steps"]:
        cur = step[0]["observation"].get("current")
        if not cur or len(cur.get("players") or []) < 2:
            continue
        turns += 1
        me, opp = cur["players"][our_index], cur["players"][1 - our_index]
        if me.get("handCount") is not None:
            hands.append(me["handCount"])
        if me.get("prize"):
            our_prize = len(me["prize"])
        if opp.get("prize"):
            opp_prize = len(opp["prize"])
    return {
        "hand_mean": statistics.mean(hands) if hands else 0,
        "hand_final": hands[-1] if hands else 0,
        "hand_max": max(hands) if hands else 0,
        "our_prize_left": our_prize,
        "opp_prize_left": opp_prize,
        "turns": turns,
    }


def main() -> None:
    pooled = {"win": [], "loss": []}
    for sub, label in SUBS.items():
        manifest = json.load(open(f"logs/replays/{sub}/manifest.json"))
        rows = []
        for episode_id, meta in manifest.items():
            path = f"logs/replays/{sub}/episode-{episode_id}-replay.json"
            if not os.path.exists(path):
                continue
            row = scan(path, meta["our_index"])
            row["result"] = meta["result"]
            rows.append(row)
            pooled[meta["result"]].append(row)
        print(f"{sub} {label:16} n={len(rows):>2}  hand_mean {statistics.mean(r['hand_mean'] for r in rows):5.2f}"
              f"  hand_max {statistics.mean(r['hand_max'] for r in rows):5.2f}")

    print("\npooled across the three Alakazam submissions:")
    print(f"{'metric':16} {'win':>8} {'loss':>8}")
    for key in ("hand_mean", "hand_max", "hand_final", "our_prize_left", "opp_prize_left", "turns"):
        def avg(res):
            vals = [r[key] for r in pooled[res] if r[key] is not None]
            return statistics.mean(vals) if vals else float("nan")
        print(f"{key:16} {avg('win'):>8.2f} {avg('loss'):>8.2f}")
    print(f"n: win {len(pooled['win'])}, loss {len(pooled['loss'])}")


if __name__ == "__main__":
    main()

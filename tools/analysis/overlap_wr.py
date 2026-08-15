"""
Test whether the agent wins more when the opponent's deck resembles its corpus.

Bins the 247 replayed Kaggle games by how closely the opponent's 60-card list
matches the nearest list in decks/top20, then reports the win rate per bin.
"""
import json, glob, os, collections, statistics


def corpus():
    out = []
    for path in glob.glob("decks/top20/**/*.csv", recursive=True):
        ids = [int(line) for line in open(path) if line.strip().isdigit()]
        if ids:
            out.append(collections.Counter(ids))
    return out


def overlap(a, b):
    return sum((a & b).values()) / max(sum(a.values()), 1)


def main():
    pool = corpus()
    rows = []
    distinct = set()
    for sub in sorted(os.listdir("logs/replays")):
        mpath = f"logs/replays/{sub}/manifest.json"
        if not os.path.exists(mpath):
            continue
        for episode_id, meta in json.load(open(mpath)).items():
            path = f"logs/replays/{sub}/episode-{episode_id}-replay.json"
            if not os.path.exists(path):
                continue
            data = json.load(open(path))
            vis = data["steps"][0][0].get("visualize")
            if not vis or not vis[0].get("action") or len(vis[0]["action"]) < 2:
                continue
            deck = collections.Counter(vis[0]["action"][1 - meta["our_index"]])
            distinct.add(tuple(sorted(deck.elements())))
            rows.append((max(overlap(deck, c) for c in pool), meta["result"]))

    print(f"games {len(rows)}, distinct opponent decklists {len(distinct)}")
    bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"{'overlap bin':14} {'n':>4} {'wins':>5} {'win rate':>9}")
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel:
            continue
        wins = sum(1 for _, res in sel if res == "win")
        print(f"{lo:.1f}-{hi:.1f}      {len(sel):>4} {wins:>5} {wins/len(sel):>9.2f}")
    wins = sum(1 for _, res in rows if res == "win")
    print(f"{'overall':14} {len(rows):>4} {wins:>5} {wins/len(rows):>9.2f}")


if __name__ == "__main__":
    main()

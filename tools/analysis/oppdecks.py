"""
Compare the decks real Kaggle opponents bring against the training corpus.

Each replay's step-0 visualize block carries both seats' full 60-card lists.
This extracts the opponent's list per episode and scores it against every list
in decks/top20 by card-multiset overlap, which is what decides whether the
agent has ever trained on the matchup.
"""
import json, glob, os, collections, csv, statistics


def corpus() -> dict[str, collections.Counter]:
    out = {}
    for path in glob.glob("decks/top20/**/*.csv", recursive=True):
        ids = [int(line) for line in open(path) if line.strip().isdigit()]
        if ids:
            out[os.path.relpath(path, "decks/top20")] = collections.Counter(ids)
    return out


def overlap(a: collections.Counter, b: collections.Counter) -> float:
    shared = sum((a & b).values())
    return shared / max(sum(a.values()), 1)


def opponent_deck(path: str, our_index: int) -> collections.Counter | None:
    data = json.load(open(path))
    vis = data["steps"][0][0].get("visualize")
    if not vis or not vis[0].get("action"):
        return None
    lists = vis[0]["action"]
    if len(lists) < 2:
        return None
    return collections.Counter(lists[1 - our_index])


def main() -> None:
    pool = corpus()
    print(f"corpus: {len(pool)} lists")
    best_scores, matched = [], collections.Counter()
    total = 0
    for sub in sorted(os.listdir("logs/replays")):
        mpath = f"logs/replays/{sub}/manifest.json"
        if not os.path.exists(mpath):
            continue
        for episode_id, meta in json.load(open(mpath)).items():
            path = f"logs/replays/{sub}/episode-{episode_id}-replay.json"
            if not os.path.exists(path):
                continue
            deck = opponent_deck(path, meta["our_index"])
            if not deck:
                continue
            total += 1
            scored = max(pool.items(), key=lambda kv: overlap(deck, kv[1]))
            score = overlap(deck, scored[1])
            best_scores.append(score)
            matched[scored[0] if score >= 0.8 else "(no corpus match)"] += 1
    print(f"opponent decks parsed: {total}")
    print(f"best-match overlap with the training corpus: mean {statistics.mean(best_scores):.2f}, "
          f"median {statistics.median(best_scores):.2f}, min {min(best_scores):.2f}, max {max(best_scores):.2f}")
    exact = sum(1 for s in best_scores if s >= 0.95)
    close = sum(1 for s in best_scores if 0.8 <= s < 0.95)
    far = sum(1 for s in best_scores if s < 0.8)
    print(f"  >=0.95 overlap (effectively a corpus list): {exact}/{total}")
    print(f"  0.80-0.95 (same archetype, different build): {close}/{total}")
    print(f"  <0.80 (deck the agent never trained against): {far}/{total}")
    print("\nnearest corpus list, by episode count:")
    for name, count in matched.most_common(12):
        print(f"  {count:>3}  {name}")


if __name__ == "__main__":
    main()

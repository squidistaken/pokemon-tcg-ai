"""
Two-way, seat-balanced head-to-head between two checkpoints, with a p-value.

Both directions are played: checkpoint A in the agent seat with B frozen as the
opponent, then B in the agent seat with A frozen. Pooling them cancels the
first-player advantage and any asymmetry between the agent code path (which
decomposes a multi-select into timesteps) and the opponent code path (which
answers a whole selection in one call), neither of which a one-way match can
separate from real skill.

Run from the repository root::

    uv run python scripts/head_to_head.py \\
        --a outputs/kl-coeff-ablation-20260816/kl0.25-local-16w/checkpoints/snapshot_000006094848.pt \\
        --b outputs/bc/bc-v6-submit.pt \\
        --games 2000 --workers 6
"""

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import DictConfig, OmegaConf

from src.training.env_factory import build_probe_specs, make_env_factories
from src.training.self_play import build_eval_opponent_factory
from tools.head_to_head.head_to_head import HeadToHeadMatch, build_match_policy
from tools.head_to_head.match_statistics import EpisodeRecord, Interval, MatchStatistics

logger = logging.getLogger("head_to_head")

#: Seeds are offset by this per shard so no two shards deal the same matchups,
#: while the same offsets are reused in both directions so the two halves see
#: the same deck and seat sequence.
SHARD_SEED_STRIDE = 10_000


def build_config(
    deck_pool: str,
    max_options: int,
    seed: int,
    encoder: str,
    deck_matchup: str = "mirror",
    deck_sampling: str = "round_robin",
) -> DictConfig:
    """
    Build the minimal config the eval environment and specs need.

    Deliberately not the training run's full config: only the environment and
    deck-pool settings affect what is played, and the architecture is read from
    each checkpoint's own embedded config rather than from here.

    :param deck_pool: Directory of decks to draw the matchups from.
    :param max_options: Padded option-space size; must match the checkpoints.
    :param seed: Base seed for deck order and seat draws.
    :param encoder: Observation encoder name.
    :param deck_matchup: ``"mirror"`` gives both seats the same deck, so the
        null is exactly 0.5 and the score is pure policy strength. Set
        ``"independent"`` to deal the seats different decks, which is what
        turns the per-archetype breakdown into a deck-strength probe.
    :return: A config accepted by the env factory and the eval opponent builder.
    """
    return OmegaConf.create(
        {
            "seed": seed,
            "env": {
                "deck_pool": deck_pool,
                "deck_matchup": deck_matchup,
                "eval_deck_matchup": deck_matchup,
                "deck_sampling": deck_sampling,
                "eval_deck_sampling": deck_sampling,
                "deck_holdout_frac": 0.0,
                "deck_split_seed": 0,
                "deck_pool_selection": "observation",
                "deck_switch_steps": 0,
                "max_options": max_options,
                "encoder": encoder,
                "num_workers": 1,
                "parallel": False,
                "serial_for_single": True,
            },
            "train": {},
            "agent": {"device": "cpu"},
            "paths": {"data_dir": "decks"},
        }
    )


def play_shard(task: dict[str, object]) -> list[dict[str, object]]:
    """
    Play one shard of one direction in its own process.

    Each shard builds its own environment, policy and frozen opponent: the
    engine keeps a battle pointer per handle, and a policy is cheap to rebuild
    next to the cost of the games.

    :param task: Shard descriptor with the config container, both checkpoint
        paths, the episode count and the seed offset.
    :return: Episode records as plain dicts, so they survive the process
        boundary without a custom pickler.
    """
    torch.set_num_threads(1)
    cfg = OmegaConf.create(task["config"])
    seed_offset = int(task["seed_offset"])
    cfg.seed = int(cfg.seed) + seed_offset

    obs_spec, action_spec = build_probe_specs(cfg)
    opponent_factory = build_eval_opponent_factory(
        cfg, obs_spec, action_spec, opponent=str(task["opponent"])
    )
    env = make_env_factories(cfg, opponent_factory, deck_split="eval")[0]()
    try:
        policy = build_match_policy(
            str(task["policy"]), cfg, obs_spec, action_spec, device="cpu"
        )
        match = HeadToHeadMatch(env, policy, deterministic=bool(task["deterministic"]))
        records = match.play(int(task["episodes"]))
    finally:
        env.close()
    return [asdict(record) for record in records]


def run_direction(
    cfg: DictConfig,
    policy_checkpoint: Path,
    opponent_checkpoint: Path,
    games: int,
    workers: int,
    deterministic: bool,
) -> list[EpisodeRecord]:
    """
    Play one direction of the match across ``workers`` processes.

    :param cfg: Environment config shared by every shard.
    :param policy_checkpoint: Checkpoint played in the agent seat.
    :param opponent_checkpoint: Checkpoint frozen in the opponent seat.
    :param games: Games to play in this direction.
    :param workers: Processes to spread them over.
    :param deterministic: Take the policy's mode instead of sampling.
    :return: Episode records scored from ``policy_checkpoint``'s seat.
    """
    per_shard = [games // workers] * workers
    for index in range(games % workers):
        per_shard[index] += 1
    container = OmegaConf.to_container(cfg, resolve=True)
    tasks = [
        {
            "config": container,
            "policy": str(policy_checkpoint),
            "opponent": str(opponent_checkpoint),
            "episodes": episodes,
            "seed_offset": shard * SHARD_SEED_STRIDE,
            "deterministic": deterministic,
        }
        for shard, episodes in enumerate(per_shard)
        if episodes > 0
    ]
    context = mp.get_context("fork")
    with context.Pool(processes=len(tasks)) as pool:
        shards = pool.map(play_shard, tasks)
    return [EpisodeRecord(**record) for shard in shards for record in shard]


def report(
    name_a: str,
    name_b: str,
    forward: list[EpisodeRecord],
    reverse: list[EpisodeRecord],
    seed: int,
) -> dict[str, object]:
    """
    Print the comparison and return it as a serializable summary.

    ``reverse`` is recorded from B's seat, so it is flipped before pooling and
    every number below is stated from A's point of view.

    :param name_a: Label for checkpoint A.
    :param name_b: Label for checkpoint B.
    :param forward: Episodes with A in the agent seat.
    :param reverse: Episodes with B in the agent seat.
    :param seed: Bootstrap seed.
    :return: The summary written to the JSON side-car.
    """
    flipped = [
        EpisodeRecord(
            score=1.0 - record.score,
            terminated=record.terminated,
            agent_seat=1 - record.agent_seat,
            deck=record.opponent_deck,
            opponent_deck=record.deck,
            steps=record.steps,
        )
        for record in reverse
    ]
    pooled = MatchStatistics(forward + flipped, seed=seed)
    naive = pooled.naive_interval()
    clustered = pooled.cluster_interval()
    p_value = pooled.binomial_p_value()

    print()
    print(f"{name_a}  vs  {name_b}")
    print(f"  decided games      {pooled.games} ({pooled.unfinished} unfinished)")
    print(
        f"  record             {pooled.wins}W {pooled.draws}D "
        f"{pooled.games - pooled.wins - pooled.draws}L"
    )
    print(f"  score (draw=0.5)   {naive}   naive 95% CI")
    print(f"                     {clustered}   archetype-clustered 95% CI")
    print(f"  Elo                {pooled.elo_interval(clustered)}")
    print(f"  exact binomial p   {p_value:.4g}  (H0: score = 0.5)")
    print(f"  verdict            {_verdict(clustered, p_value)}")

    print()
    print("  by direction")
    for label, records in (
        (f"{name_a} in agent seat", forward),
        (f"{name_b} in agent seat, flipped", flipped),
    ):
        half = MatchStatistics(records, seed=seed)
        print(
            f"    {label:<38} {half.games:>5} games  score {half.score:.4f}"
            f"  {half.naive_interval()}"
        )

    print()
    print("  by seat occupied by " + name_a)
    for seat, (count, mean) in pooled.seat_split().items():
        print(
            f"    seat {seat} ({'first' if seat == 0 else 'second'})"
            f"{'':<20} {count:>5} games  score {mean:.4f}"
        )

    print()
    print("  worst archetypes for " + name_a)
    for deck, (count, mean) in list(pooled.archetype_split().items())[:8]:
        print(f"    {deck:<42} {count:>5} games  score {mean:.4f}")

    print()
    for target in (0.02, 0.01):
        needed = pooled.games_for_precision(target)
        print(
            f"  games for +/-{target:.2f} half-width: {needed}"
            f"  ({max(0, needed - pooled.games)} more)"
        )

    return {
        "a": name_a,
        "b": name_b,
        "games": pooled.games,
        "unfinished": pooled.unfinished,
        "wins": pooled.wins,
        "draws": pooled.draws,
        "score": pooled.score,
        "naive_ci": [naive.low, naive.high],
        "clustered_ci": [clustered.low, clustered.high],
        "binomial_p": p_value,
        "seat_split": {str(k): v for k, v in pooled.seat_split().items()},
        "archetype_split": pooled.archetype_split(),
    }


def _verdict(interval: Interval, p_value: float) -> str:
    """
    State what the interval supports, without overclaiming from the p-value.

    :param interval: The clustered score interval.
    :param p_value: Exact binomial p-value against 0.5.
    :return: A one-line reading.
    """
    if interval.low > 0.5 and p_value < 0.05:
        return "A is stronger; the interval excludes parity"
    if interval.high < 0.5 and p_value < 0.05:
        return "A is weaker; the interval excludes parity"
    return (
        f"no difference resolved; parity sits inside the interval "
        f"(+/-{(interval.high - interval.low) / 2:.3f})"
    )


def parse_args() -> argparse.Namespace:
    """
    :return: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="Checkpoint under test.")
    parser.add_argument("--b", required=True, type=Path, help="Reference checkpoint.")
    parser.add_argument(
        "--games",
        type=int,
        default=2000,
        help="Total games, split evenly across both directions.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Processes per direction. Leave cores for training.",
    )
    parser.add_argument("--deck-pool", default="decks/expert_pool_30")
    parser.add_argument(
        "--deck-matchup",
        choices=("mirror", "independent"),
        default="mirror",
        help="mirror: both seats get the same deck, isolating policy "
        "strength. independent: different decks, so the per-archetype "
        "split reads as deck strength.",
    )
    parser.add_argument(
        "--deck-sampling",
        choices=("round_robin", "uniform"),
        default="round_robin",
        help="uniform is required with --deck-matchup independent; "
        "round_robin pairs only adjacent pool indices there.",
    )
    parser.add_argument("--max-options", type=int, default=128)
    parser.add_argument("--encoder", default="structured")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample actions instead of taking the mode.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the summary and raw records here as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run both directions of the match and report the pooled result.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()
    for checkpoint in (args.a, args.b):
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint}")

    torch.set_num_threads(1)
    cfg = build_config(
        args.deck_pool,
        args.max_options,
        args.seed,
        args.encoder,
        args.deck_matchup,
        args.deck_sampling,
    )
    half = args.games // 2
    deterministic = not args.sample

    started = time.time()
    print(f"Playing {half} games per direction on {args.workers} processes...")
    forward = run_direction(cfg, args.a, args.b, half, args.workers, deterministic)
    print(f"  direction 1 done in {time.time() - started:.0f}s")
    reverse = run_direction(cfg, args.b, args.a, half, args.workers, deterministic)
    print(f"  direction 2 done in {time.time() - started:.0f}s")

    summary = report(args.a.name, args.b.name, forward, reverse, args.seed)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary["deterministic"] = deterministic
    summary["deck_pool"] = args.deck_pool

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "forward": [asdict(record) for record in forward],
                    "reverse": [asdict(record) for record in reverse],
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

"""
Compare two policies against a shared frozen opponent with held-out decks.

Evaluates both against the same (opponent checkpoint + held-out opponent deck)
pairs and reports mean and worst-decile win rate.

Run from the repository root::

    uv run python scripts/compare_policies.py --help
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from src.env.decks.deck import load_deck, resolve_deck_paths
from src.env.tcg_env import TCGEnv
from src.policies.greedy_policy_opponent import load_greedy_opponent
from tests.conftest import MAX_OPTIONS


def _load_policy(checkpoint_path, device="cpu"):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "model": {
                "embed_dim": 128,
                "backbone": {
                    "_target_": "src.models.mlp.MLPBackbone",
                    "num_cells": [256, 256],
                    "activation": "tanh",
                    "in_keys": [
                        ["observation", "globals"],
                        ["observation", "select_cats"],
                        ["observation", "context_card_ids"],
                        ["observation", "stadium_id"],
                        ["observation", "options"],
                        ["observation", "pokemon"],
                        ["observation", "my"],
                        ["observation", "opp"],
                        ["observation", "select_deck"],
                        ["observation", "looking"],
                    ],
                },
                "head": {"_target_": "src.models.heads.LinearPolicyHead"},
                "value_head": {"num_cells": [256]},
            },
            "env": {"max_options": MAX_OPTIONS},
        }
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from src.policies.ppo_actor import build_actor_critic, build_ppo_operator

    ac = build_actor_critic(cfg, checkpoint["obs_spec"], checkpoint["action_spec"])
    ac.load_state_dict(checkpoint["model"])
    return (
        build_ppo_operator(ac, checkpoint["action_spec"])
        .get_policy_operator()
        .to(device)
    )


def _play_episode(env, policy, max_steps=2000, deterministic=True):
    expl = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    with set_exploration_type(expl), torch.no_grad():
        td = env.reset()
        for _ in range(max_steps):
            td = policy(td.to(policy.device)).to("cpu")
            td = env.step(td)
            if td["next", "done"].item():
                return float(td["next", "reward"].item()), bool(
                    td["next", "terminated"].item()
                )
            td = step_mdp(td)
    return 0.0, False


def evaluate(policy, agent_deck, opponent, opponent_decks, episodes_per_pair, seed):
    results = []
    for opp_deck in opponent_decks:
        wins = 0
        decided = 0
        for ep in range(episodes_per_pair):
            env = TransformedEnv(
                TCGEnv(
                    list(agent_deck),
                    list(opp_deck),
                    max_options=MAX_OPTIONS,
                    opponent=opponent,
                    seed=seed + ep,
                ),
                ActionMask(),
            )
            try:
                env.set_spec_lock_(True)
                reward, terminated = _play_episode(env, policy)
                if terminated:
                    decided += 1
                    if reward > 0:
                        wins += 1
            finally:
                env.close()
        results.append(wins / decided if decided else 0.0)
    results.sort()
    n = len(results)
    return {
        "mean": statistics.fmean(results),
        "worst_decile": statistics.fmean(results[: max(1, n // 10)]),
        "per_pair": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uniform", type=Path, required=True)
    p.add_argument("--curriculum", type=Path, required=True)
    p.add_argument("--opponent", type=Path, required=True)
    p.add_argument("--agent-deck", type=Path, required=True)
    p.add_argument("--decks", type=Path, default=Path("decks"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading policies on {device}...")
    uniform_pol = _load_policy(args.uniform, device)
    curriculum_pol = _load_policy(args.curriculum, device)

    print(f"Loading opponent from {args.opponent}...")
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "model": {
                "embed_dim": 128,
                "backbone": {
                    "_target_": "src.models.mlp.MLPBackbone",
                    "num_cells": [256, 256],
                    "activation": "tanh",
                    "in_keys": [
                        ["observation", "globals"],
                        ["observation", "select_cats"],
                        ["observation", "context_card_ids"],
                        ["observation", "stadium_id"],
                        ["observation", "options"],
                        ["observation", "pokemon"],
                        ["observation", "my"],
                        ["observation", "opp"],
                        ["observation", "select_deck"],
                        ["observation", "looking"],
                    ],
                },
                "head": {"_target_": "src.models.heads.LinearPolicyHead"},
                "value_head": {"num_cells": [256]},
            },
            "env": {"max_options": MAX_OPTIONS},
        }
    )
    from src.training.env_factory import make_encoder

    opponent = load_greedy_opponent(
        args.opponent,
        cfg,
        None,
        None,
        make_encoder("structured", MAX_OPTIONS),
        device="cpu",
    )

    all_paths = resolve_deck_paths(str(args.decks))
    arch_name = args.agent_deck.parent.name
    agent_deck = load_deck(str(args.agent_deck))
    held_out = [
        load_deck(p)
        for p in all_paths
        if Path(p).parent.name != arch_name and p != str(args.agent_deck)
    ]
    print(
        f"Agent: {arch_name}  |  Held-out opponent decks: {len(held_out)}/{len(all_paths)}  |  Games per pair: {args.episodes}"
    )

    print("\n--- Uniform ---")
    ur = evaluate(uniform_pol, agent_deck, opponent, held_out, args.episodes, args.seed)
    print(f"mean={ur['mean']:.3f}  worst_decile={ur['worst_decile']:.3f}")

    print("\n--- Curriculum ---")
    cr = evaluate(
        curriculum_pol, agent_deck, opponent, held_out, args.episodes, args.seed + 10000
    )
    print(f"mean={cr['mean']:.3f}  worst_decile={cr['worst_decile']:.3f}")

    summary = {
        "uniform": {"mean": ur["mean"], "worst_decile": ur["worst_decile"]},
        "curriculum": {"mean": cr["mean"], "worst_decile": cr["worst_decile"]},
        "mean_diff": cr["mean"] - ur["mean"],
        "worst_decile_diff": cr["worst_decile"] - ur["worst_decile"],
    }
    print(f"\n{json.dumps(summary, indent=2)}")
    (args.uniform.parent.parent / "comparison.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()

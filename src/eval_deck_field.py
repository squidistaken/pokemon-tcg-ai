from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchrl.data import Categorical

from src.policies.ppo_actor import build_actor_critic, build_ppo_operator
from src.training import Evaluator, build_eval_opponent_factory, make_env_factories

load_dotenv(Path(__file__).parents[1] / ".env", override=False)


@hydra.main(version_base=None, config_path="../conf", config_name="eval_deck_field")
def main(cfg: DictConfig) -> None:
    """
    Score one trained checkpoint's per-deck win rate against a varied field.

    Loads ``train.eval_opponent_checkpoint`` as both the agent under test and
    its opponent, then plays the held-out pool under ``env.deck_matchup=independent``,
    so the resulting per-archetype win rate reflects deck strength against variety
    rather than the mirror-matchup piloting-skill signal ``ppo_selfplay_multideck``'s
    periodic eval reports.

    :param cfg: Hydra configuration, composed from conf/eval_deck_field.yaml.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))

    checkpoint = cfg.train.get("eval_opponent_checkpoint")
    if not checkpoint:
        raise ValueError(
            "eval_deck_field requires train.eval_opponent_checkpoint to point at the "
            "trained agent whose deck performance is being probed; it is loaded as "
            "both the agent under test and its opponent."
        )
    checkpoint_path = Path(to_absolute_path(str(checkpoint)))
    if not checkpoint_path.is_file():
        raise ValueError(f"train.eval_opponent_checkpoint {checkpoint_path} does not exist.")

    probe_env = make_env_factories(cfg)[0]()
    try:
        obs_spec = probe_env.observation_spec
        action_spec = probe_env.action_spec
    finally:
        probe_env.close()
    assert isinstance(action_spec, Categorical), "env action spec must be Categorical"

    actor_critic = build_actor_critic(cfg, obs_spec, action_spec)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor_critic.load_state_dict(state_dict)
    policy = build_ppo_operator(actor_critic, action_spec).get_policy_operator()

    opponent_factory = build_eval_opponent_factory(cfg, obs_spec, action_spec)
    evaluator = Evaluator(
        env_factory=make_env_factories(cfg, opponent_factory=opponent_factory, deck_split="eval")[0],
        n_episodes=int(cfg.train.get("eval_episodes", 100)),
        deterministic=bool(cfg.train.get("eval_deterministic", True)),
        per_archetype=True,
    )
    try:
        metrics = evaluator.evaluate(policy)
    finally:
        evaluator.close()

    archetypes = sorted(
        key.removeprefix("archetype_win_rate/")
        for key in metrics
        if key.startswith("archetype_win_rate/")
    )
    print("\nField win rate per held-out deck (independent matchup):")
    for name in archetypes:
        print(f"  {name}: {metrics[f'archetype_win_rate/{name}']:.3f}")
    print(
        f"\nOverall: mean={metrics.get('archetype_win_rate_mean', float('nan')):.3f} "
        f"worst_quartile={metrics.get('archetype_win_rate_worst_quartile', float('nan')):.3f} "
        f"min={metrics.get('archetype_win_rate_min', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
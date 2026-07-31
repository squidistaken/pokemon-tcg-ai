from pathlib import Path
from typing import Any, cast

import hydra
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.policies.greedy_policy_opponent import load_actor_critic
from src.policies.ppo_actor import build_ppo_operator
from src.training import (
    TrainingCallback,
    WeightsAndBiases,
    build_evaluator,
    build_probe_specs,
)

load_dotenv(Path(__file__).parents[1] / ".env", override=False)


@hydra.main(version_base=None, config_path="../conf", config_name="eval_deck_field")
def main(cfg: DictConfig) -> None:
    """
    Score one trained checkpoint's per-deck win rate against a varied field.

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

    obs_spec, action_spec = build_probe_specs(cfg)

    actor_critic = load_actor_critic(checkpoint_path, cfg, obs_spec, action_spec, device="cpu")
    policy = build_ppo_operator(actor_critic, action_spec).get_policy_operator()

    callbacks: list[TrainingCallback] = [hydra.utils.instantiate(callback) for callback in cfg.callbacks]
    run_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    for callback in callbacks:
        callback.on_train_start(run_config)

    evaluator = build_evaluator(cfg, obs_spec, action_spec)
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

    for callback in callbacks:
        if isinstance(callback, WeightsAndBiases):
            callback.log_table(
                "deck_field/win_rate",
                ["archetype", "win_rate"],
                [(name, metrics[f"archetype_win_rate/{name}"]) for name in archetypes],
            )
        callback.on_train_end(metrics)


if __name__ == "__main__":
    main()

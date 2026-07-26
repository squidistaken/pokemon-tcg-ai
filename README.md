# Pokémon Trading Card Game

Solution for the Kaggle [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) competition.

See [this video](https://www.youtube.com/watch?v=eKC5PlYoboE) for an explanation of the game.

## Setup

### Requirements

After cloning the repository make sure you have the [UV package manager](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
shell_name="$(basename "${SHELL:-}")"

if [ "$shell_name" = "zsh" ]; then
  . "$HOME/.local/bin/env"
elif [ "$shell_name" = "bash" ]; then
  . "$HOME/.local/bin/env"
else
  echo "ERROR: unsupported shell: ${SHELL:-unknown}" >&2
  exit 1
fi

uv --version
```
Install dependencies:
```bash
uv sync
```

Then access the virtual environment with `source .venv/bin/activate`.

To add more libraries:
```bash
uv add <package_name>
```

## Code structure

```
cg/                        Python bindings for the bundled cabt C battle engine (ctypes)
  sim.py                     Shared library loading and low-level ctypes structs
  api.py                     Typed dataclasses for engine observations/options
  game.py                    Thin procedural wrapper around a single global battle
  utils.py                   dict <-> dataclass conversion helpers

ptcg_engine/                C++ source of the cabt battle engine (compiled to the shared library cg/sim.py loads)

src/
  env/                        TorchRL environment
    tcg_env.py                  TCGEnv: EnvBase wrapping the engine, masked action space, opponent-in-env
    battle_handle.py            Per-instance battle pointer, decoupled from cg.game's global one
    observation_encoder.py      ObservationEncoder: abstract base class for observation encoders
    structured_observation_encoder.py  Structured observation contract: card-ID/option/board tensors (docs/torchrl_environment.md)
    option_reference_resolver.py      OptionReferenceResolver: stateless option -> (card, target, attack) ID lookups
    card_database.py            Static card-ID-indexed lookup tables (for model-side embeddings)
    deck.py                     Deck CSV loading
    opponent_pool.py            Self-play opponent pool (samples/holds frozen policy snapshots)
    snapshot_opponent_pool.py   OpponentPool that discovers new learner snapshots from disk (ParallelEnv-safe)
    random_opponent.py          Uniform-random opponent baseline
  models/                     Actor-critic network, independent of the policy/training wiring
    backbone.py                  Backbone ABC + MLPBackbone (DeepSets/SetTransformer/TemporalTransformer/Recurrent planned)
    structured_obs_adapter.py    StructuredObsAdapter: embeds card IDs, normalizes scalars, pools zones for the MLP
    heads.py                     LinearPolicyHead (flat logits over actions) + ValueHead (scalar critic)
    actor_critic.py              ActorCritic: shared trunk feeding both heads, tensordict-in/tensordict-out
    transformer.py                Set-transformer building blocks (placeholder, not yet implemented)
  policies/
    random_masked_policy.py     Uniform random policy over the action mask (stand-in for the future PPO actor)
    greedy_policy_opponent.py   Greedy opponent baseline built on a saved ActorCritic checkpoint
    ppo_actor.py                 build_actor_critic / build_ppo_operator: assemble the ActorValueOperator from Hydra config
  training/
    trainer.py                  Trainer: parallel rollout collection via TorchRL's Collector
    base_trainer.py             BaseTrainer interface
    ppo_trainer.py               PPOTrainer: Trainer subclass running GAE + ClipPPOLoss optimization
    env_factory.py               Builds TransformedEnv instances (deck + opponent + ActionMask) for the collector
    self_play.py                 build_opponent_factory: the picklable self-play league factory handed to each env worker
    evaluator.py                 Evaluator: scores the policy against a fixed opponent (readable curve under self-play)
    callbacks/                  Metric sinks; the trainer emits, these decide where it goes
      base.py                     TrainingCallback hooks + CallbackList (fan-out, isolates failures)
      snapshot_callback.py        SnapshotCallback: freezes the learner into the self-play league at a frame interval
      wandb_callback.py           WeightsAndBiases: the only module that imports wandb
  train.py                    Hydra entry point (python -m src.train)

conf/                        Hydra configs (config.yaml + env/, agent/, model/, train/, collector/, callbacks/, experiment/ groups)
  paths/default.yaml          Scheduler-independent input and output locations
  model/
    default.yaml                Composes one backbone + one head, holds shared dims (embed_dim, value_head)
    backbone/mlp.yaml            MLP baseline trunk (more backbones added as separate config files as they land)
    head/linear.yaml             Flat logits head (more heads added as separate config files as they land)
  train/
    default.yaml                Keys shared by every training variant; the variants below override only what differs
    fixed_opponent.yaml          Default: fixed random opponent, no snapshotting; the control for self-play runs
    ppo_selfplay.yaml            Self-play league: snapshot interval, pool size, periodic fixed-opponent evaluation
  callbacks/
    wandb.yaml                  Default: Weights & Biases run (project/entity/group/tags/mode)
    none.yaml                    Console/Hydra log lines only; for throwaway runs
.env.example                 Template for the untracked .env holding secrets (WANDB_API_KEY)
scripts/                     Standalone dev scripts (not part of the training entry point)
  bench_throughput.py          Collection throughput benchmark (naive vs SerialEnv vs ParallelEnv)
  generate_obs_fixtures.py     Regenerates the committed observation fixtures in tests/fixtures/
  run_selfplay_compile.sh      1M-frame self-play run with torch.compile (caps Inductor's compile workers)
decks/                       Example deck CSVs
docs/                        Design docs (torchrl_environment.md, game.md)
tests/                       Unit tests (+ fixtures/: committed sample observations and card tables)
main.py                      Kaggle submission entry point (fixed format, uses cg.api directly)
slurm-conf/                  Slurm profiles, uv setup, and generic submission/training scripts
```

The **backbone** and **head** are independent Hydra config groups, so any backbone can be paired
with any head from the CLI or a sweep, e.g. `python -m src.train model/backbone=mlp model/head=linear`.

## Usage

Run training from the repo root:
```bash
python -m src.train
```
Config is managed by [Hydra](https://hydra.cc/) (`conf/config.yaml`); override any field on the command line, e.g.:
```bash
python -m src.train collector.total_frames=100000 env.num_workers=4 set_seed=true
```

Metric backends are selected by the `callbacks` group, and the W&B run is
labelled from the top-level `wandb` block:
```bash
python -m src.train agent=ppo wandb.group=ablation-lr wandb.tags=[baseline]
python -m src.train agent=ppo wandb.mode=offline    # record now, `wandb sync` later
python -m src.train agent=ppo callbacks=none        # no metric backend at all
```

For a quick smoke test of the whole loop there is a debug experiment: PPO
self-play at 512 frames on a single in-process env, a small model, no metric
backend and DEBUG logging, finishing in a few seconds:
```bash
python -m src.train +experiment=debug
```

### Self-play

By default (`train=fixed_opponent`) the agent trains against the built-in
uniform-random opponent. `train=ppo_selfplay` instead builds a self-play league:
```bash
python -m src.train agent=ppo train=ppo_selfplay
```
Every `train.snapshot_interval` frames the learner is frozen to disk, and each
environment worker adds that snapshot to the pool it samples an opponent from
per episode, keeping the newest `train.pool_size` of them. The random opponent
stays a permanent league member, so the agent always retains a fixed reference
point instead of drifting against copies of itself. Snapshots go to the Hydra
run directory, so runs never inherit each other's opponents.

Because the league tracks the learner, the collected `win_rate` sits near 0.5
regardless of how strong the policy gets. The readable progress signal is
instead the periodic evaluation against a *fixed* random opponent, logged under
`eval/` every `train.eval_interval` frames:
```bash
python -m src.train agent=ppo train=ppo_selfplay train.eval_interval=100000 train.eval_episodes=200
```
Evaluation is serial and costs collection throughput, so it is a trade between
curve resolution and speed. `train.eval_deterministic` (default `true`) scores
the policy's argmax; set it to `false` to score the sampling behaviour used
during collection.

Online W&B logging is required when selected: configure `WANDB_API_KEY` in an
untracked `.env` or run `wandb login --verify`. Authentication, connection, or
logging failures stop training. Use `wandb.mode=offline` only when local
recording for a later `wandb sync` is intentional.

Slurm support is kept separately in [`slurm-conf/`](slurm-conf/README.md). The
profiles select a normal Hydra config and add scheduler-specific overrides;
they do not participate in local Hydra composition.

## CI

GitHub Actions runs linting, type-checking, and tests for pull requests that
are ready for review. Draft pull requests intentionally skip CI to stay within
the GitHub Actions free-plan budget.

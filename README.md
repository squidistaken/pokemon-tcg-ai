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
    flat_observation_encoder.py       Legacy flat-vector encoder (deprecated, testing only)
    card_database.py            Static card-ID-indexed lookup tables (for model-side embeddings)
    deck.py                     Deck CSV loading
    opponent_pool.py            Self-play opponent pool (samples/holds frozen policy snapshots)
    random_opponent.py          Uniform-random opponent baseline
  models/                     Actor-critic network, independent of the policy/training wiring
    backbone.py                  Backbone ABC + MLPBackbone (DeepSets/SetTransformer/TemporalTransformer/Recurrent planned)
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
    callbacks/                  Metric sinks; the trainer emits, these decide where it goes
      base.py                     TrainingCallback hooks + CallbackList (fan-out, isolates failures)
      wandb_callback.py           WeightsAndBiases: the only module that imports wandb
  train.py                    Hydra entry point (python -m src.train)

conf/                        Hydra configs (config.yaml + env/, agent/, model/, train/, collector/, logging/ groups)
  model/
    default.yaml                Composes one backbone + one head, holds shared dims (embed_dim, value_head)
    backbone/mlp.yaml            MLP baseline trunk (more backbones added as separate config files as they land)
    head/linear.yaml             Flat logits head (more heads added as separate config files as they land)
  logging/
    wandb.yaml                  Default: Weights & Biases run (project/entity/group/tags/mode)
    none.yaml                    Console/Hydra log lines only; for throwaway runs
.env.example                 Template for the untracked .env holding secrets (WANDB_API_KEY)
scripts/                     Standalone dev scripts (not part of the training entry point)
  bench_throughput.py          Collection throughput benchmark (naive vs SerialEnv vs ParallelEnv)
  generate_obs_fixtures.py     Regenerates the committed observation fixtures in tests/fixtures/
decks/                       Example deck CSVs
docs/                        Design docs (torchrl_environment.md, game.md)
tests/                       Unit tests (+ fixtures/: committed sample observations and card tables)
main.py                      Kaggle submission entry point (fixed format, uses cg.api directly)
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

## CI

GitHub Actions runs linting, type-checking, and tests for pull requests that
are ready for review. Draft pull requests intentionally skip CI to stay within
the GitHub Actions free-plan budget.


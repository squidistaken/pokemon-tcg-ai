# Pokémon Trading Card Game

Solution for the Kaggle [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) competition.

See [this video](https://www.youtube.com/watch?v=eKC5PlYoboE) for an explanation of the game.

## Setup

### Requirements

After cloning the repository make sure you have the [UV package manager](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
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
    flat_observation_encoder.py       Legacy flat-vector encoder (deprecated, testing only)
    card_database.py            Static card-ID-indexed lookup tables (for model-side embeddings)
    deck.py                     Deck CSV loading
    opponent_pool.py            Self-play opponent pool (samples/holds frozen policy snapshots)
    random_opponent.py          Uniform-random opponent baseline
  policies/
    random_masked_policy.py     Uniform random policy over the action mask (stand-in for the future PPO actor)
  training/
    trainer.py                  Trainer: parallel rollout collection via TorchRL's Collector
    base_trainer.py             BaseTrainer interface
    env_factory.py               Builds TransformedEnv instances (deck + opponent + ActionMask) for the collector
  train.py                    Hydra entry point (python -m src.train)

conf/                        Hydra configs (config.yaml + env/, agent/, collector/ groups)
scripts/                     Standalone dev scripts (not part of the training entry point)
  bench_throughput.py          Collection throughput benchmark (naive vs SerialEnv vs ParallelEnv)
  generate_obs_fixtures.py     Regenerates the committed observation fixtures in tests/fixtures/
decks/                       Example deck CSVs
docs/                        Design docs (torchrl_environment.md, game.md)
tests/                       Unit tests (+ fixtures/: committed sample observations and card tables)
main.py                      Kaggle submission entry point (fixed format, uses cg.api directly)
```

This section should be kept up to date whenever the code structure changes.

## Usage

Run training from the repo root:
```bash
python -m src.train
```
Config is managed by [Hydra](https://hydra.cc/) (`conf/config.yaml`); override any field on the command line, e.g.:
```bash
python -m src.train collector.total_frames=100000 env.num_workers=4 set_seed=true
```



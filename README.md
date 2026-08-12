# Pokémon Trading Card Game

Solution for the Kaggle [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
competition. [This video](https://www.youtube.com/watch?v=eKC5PlYoboE) explains the game.

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

### Deck corpus

Decks are versioned as GitHub Releases tagged `decks-*`. No version is committed to the repo, so
publishing a new corpus never touches a tracked file. Pull it once per checkout before
multi-deck training:

```bash
./scripts/fetch_decks.sh           # newest decks-* release: verify + unpack into decks/
./scripts/fetch_decks.sh decks-v4  # or pin a specific version for reproducibility
./scripts/fetch_decks.sh --root /path/to/target # install to a specific filepath
```

Publish a new corpus (auto-increments to the next `decks-vN`):

```bash
./scripts/update_decks_release.sh <--notes> "…"
```

`fetch_decks.sh` picks up the highest-numbered release automatically. Its
`--root` option changes only the storage root: the release's standard `decks/`
layout is preserved beneath it, so training can consume it with
`paths.data_dir=/path/to/target`. `build_decks_release.sh` is the
low-level builder it wraps if you only want the local tarball.

Every deck in a release comes from Limitless. The scraper also walks Bulbapedia, but its
archetype pages are historical lists whose cards predate the engine's pool, so all of them drop
as unresolved and none reach the corpus. See the card pool section of `scraper/README.md`.

## Code structure

```
cg/                    ctypes bindings for the cabt battle engine
ptcg_engine/           C++ source of that engine
src/
  env/                 TorchRL environment: TCGEnv, observation encoding, weighted deck
                       samplers, self-play opponent pools
  models/              Backbones (mlp, transformer), obs adapter, policy/value heads, ActorCritic
  policies/            Random, greedy, and Kaggle inference policies + the PPO operator builder
  training/            Trainer, PPO, collectors, env factory, self-play, evaluators, cross-play,
                       and callbacks/ (snapshots, train state, W&B)
  train.py             Hydra entry point (python -m src.train): main() only
  trainer_builder.py   Builds the trainer, callbacks and run paths from the config
  eval_deck_field.py   Per-archetype scoring across a deck field
conf/                  Hydra configs: env/, agent/, model/ (backbone + head), train/,
                       collector/, callbacks/, experiment/, paths/
scripts/               Dev scripts: corpus building, benchmarks, Kaggle packaging, Slurm launchers
slurm-conf/            Slurm profiles, uv setup, job scripts
submission/            Kaggle entryfile, pure-Python obs parser, torch-only runtime
submission_analysis/   `python -m submission_analysis <status|episodes|deck-report|scout>`
checkpoint/            Exported inference assets
decks/                 Deck collections
docs/                  Design docs
tests/                 Unit tests + committed fixtures
```

Backbone and head are independent Hydra groups, so any pair works:
`python -m src.train model/backbone=mlp model/head=linear`.

`pointer` is the default head. It scores each slot from `[state_repr, option_repr_i]`. `linear`
scores slots from the pooled state alone, where the option table arrives as a masked mean. Mean
pooling is permutation-invariant, so `linear` cannot condition on what an action does; at most it
learns a prior over slot indices. No amount of tuning fixes that, which is why `pointer` is the
default. `pointer_dot` replaces the shared MLP with a scaled dot product and is still in
comparison. Keep `linear` as a control only.

## Usage

Examples of what `src/train.py` accepts:

```bash
python -m src.train                                                    # defaults
python -m src.train collector.total_frames=100000 env.num_workers=4    # Hydra overrides
python -m src.train +experiment=debug                                  # 512-frame smoke test
./scripts/train_tf_weighted_field.sh                                   # long run, auto-restart
```

`train_tf_weighted_field.sh` wraps `scripts/train_supervised.sh`, which restarts training after a
crash and resumes from `train_state.pt`. It sets `hydra.run.dir`, `collector.total_frames` and
`train.resume_state` itself, so do not pass those. Any other argument goes through to `src.train`
as a Hydra override. To run the same experiment without the restart wrapper:

```bash
uv run python -m src.train --config-name ppo_selfplay_multideck \
  +experiment=weighted_field collector.total_frames=100000
```

[Hydra](https://hydra.cc/) manages `conf/config.yaml`. The `callbacks` group selects the metric
backend, and the top-level `wandb` block labels the run:

```bash
python -m src.train agent=ppo wandb.group=ablation-lr wandb.tags=[baseline]
python -m src.train agent=ppo wandb.mode=offline    # record now, `wandb sync` later
python -m src.train agent=ppo callbacks=none        # no metric backend
```

Online W&B needs `WANDB_API_KEY` in an untracked `.env`, or `wandb login --verify`. Any
authentication or logging failure stops training. The final checkpoint goes to W&B as a model
artifact unless `wandb.log_checkpoints=false`; self-play snapshots always stay local.

### Reproducing the submitted agent

TODO: the exact config and command that produce the submitted checkpoint.

### Training modes

| Config | What it does |
| --- | --- |
| `train=fixed_opponent` | Default. Trains against the built-in uniform-random opponent. |
| `train=ppo_selfplay` | Self-play league plus PFSP over its members. |
| `env.deck_weighting=observation` | Deals each list in proportion to how often it was played. |
| `train.cross_play=true` | Ranks the run's own checkpoints at the end. |
| `--config-name ppo_best_response` | Measures how exploitable a frozen agent is. |

`env.deck_weighting=observation` (`conf/env/default.yaml`) weights the deck draw by each list's
`observation_count` in `manifest.json`, so training plays the lists people play. Uniform draws
spend most of a run on lists nobody plays: 76.7% of the corpus was observed once, and those lists
hold 43.6% of the observation mass. The other weightings are `winrate` and `placing`, and `null`
is uniform. The weighting applies to both seats, and to training only, because evaluation draws
its own panel. Neither seat is pinned with `env.agent_deck`, because under self-play the opponent
is a snapshot of the same network and a pin leaves it with decks it never practises. Pick the
submission deck afterwards, by probing candidates in the engine with the finished agent.

`train=ppo_selfplay` freezes the learner into `train.checkpoint_dir` every
`train.snapshot_interval` frames. Each worker draws its opponent from the newest
`train.pool_size` snapshots. The uniform-random opponent stays a permanent league member, so the
policy keeps one fixed reference point. Because the league follows the learner, the collected
`win_rate` stays near 0.5 at every strength. Read progress from `eval/` instead, which runs every
`train.eval_interval` frames against `train.eval_opponents`. Evaluation is serial, so each round
stops collection for its `train.eval_episodes` games. `train.eval_deterministic` (default `true`)
scores the argmax. `env.eval_agent_deck` and `env.eval_panel_size` hold the agent's deck and the
opponent panel fixed, so the curve reads like a submission.

`train.cross_play=true`, with self-play on, logs `crossplay/vs_latest_snapshot` during the run.
At the end it plays the checkpoints round-robin and writes `crossplay_matrix.csv` and a
Bradley-Terry Elo in `crossplay_elo.csv` (`src/training/cross_play.py`). Use that Elo to pick the
checkpoint to submit.

`--config-name ppo_best_response` freezes an agent and trains a fresh learner against it. The
learner's `eval/win_rate` is the exploitability of the frozen agent: 0.5 means the best-responder
found no hole, and every point above 0.5 is a hole it found. A self-play `win_rate` near 0.5
cannot show this, because the league moves with the learner.

```bash
python -m src.train --config-name ppo_best_response \
  train.best_response_checkpoint=/path/to/agent.pt
```

`env.deck_pool_width=N` limits training to the `N` most-observed archetypes and leaves the eval
set as it is, so a sweep measures how generalization scales with training diversity. Compare
`eval/archetype_win_rate_{mean,worst_quartile}` across the arms.

```bash
python -m src.train --config-name ppo_selfplay_multideck --multirun env.deck_pool_width=4,8,16,32
```

## Kaggle submissions

Every PPO run appends its final checkpoint to the ignored `logs/checkpoint_keys.csv` and prints a
12-character SHA-256 key. Build from the repo root, with `KAGGLE_API_TOKEN` in `.env` or after
`uv run kaggle auth login`:

```bash
uv run python scripts/make_submission.py --checkpoint latest --label my-agent --yes
uv run python scripts/make_submission.py --checkpoint 5ac45db92f3e --label my-agent \
  --submit --yes --force
```

`latest` is the bottom row of the registry. An explicit hash also resolves from the registry, with
a path or name lookup under `CHECKPOINTS_DIR` for unregistered checkpoints. The builder verifies
the recorded SHA-256 first. The deck defaults to the checkpoint's `env.deck0` and must hold
exactly 60 integer entries; `--deck PATH` overrides it. Without `--submit` the builder only prints
the `kaggle competitions submit ...` command. `--force` replaces the staging directory of that
label. `--config path/to/.hydra/config.yaml` supplies the config for an old bare state-dict
checkpoint.

Inference samples each legal choice from the masked distribution, as in training, and re-encodes
the partial selection before each next choice. `--action-selection greedy` takes the highest score
instead. `submission/runtime.py` supports `MLPBackbone` or `TransformerBackbone` with
`LinearPolicyHead` or `PointerPolicyHead`. Any other architecture fails before the builder writes
an archive.

The archive holds `main.py`, `cg_api.py`, `runtime.py`, `model.pt`, `model_config.json`,
`deck.csv`, and `submission_manifest.json`. Kaggle runs the entryfile with empty globals and calls
its last callable, so `agent` must stay last in `submission/main.py`. Kaggle supplies `torch` but
not TorchRL or `cg`, so the bundle carries a pure-Python observation parser and a Torch-only
runtime, with no TensorDict, Hydra, OmegaConf, or native simulator dependency.

Every build runs a fail-closed preflight: it parses all bundled files as Python 3.11, rejects
dynamic imports, allows only stdlib plus `torch` and bundled modules, reproduces Kaggle's loader,
calls the agent for deck setup and a real selection in an isolated interpreter, then repeats the
checks against the extracted archive. Any failure stops the build before the optional submit.

[`submission_analysis/`](submission_analysis/README.md) covers the rest of the workflow:

```bash
uv run python -m submission_analysis status --most-recent-n 2              # rank and status
uv run python -m submission_analysis episodes --download-replays           # outcomes, replays
uv run python -m submission_analysis deck-report --deck decks/example.csv  # refinement report
uv run python -m submission_analysis scout --deck decks/example.csv        # top teams' decks
```

## Habrok (Slurm)

Habrok is the University of Groningen's high-performance compute cluster. It schedules jobs with
Slurm, and [`slurm-conf/`](slurm-conf/README.md) holds the profiles and job scripts for it. A
profile selects a normal Hydra config and adds scheduler overrides. It takes no part in local
Hydra composition. The launch configs are `baseline`, `ppo`, `ppo_selfplay`, and
`ppo_transformer`.

## CI

GitHub Actions runs linting, type-checking, and tests for pull requests that are ready for review.
Drafts skip CI to stay in the free-plan budget.

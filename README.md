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

### Deck corpus

Decks are versioned as GitHub Releases tagged `decks-*`. No version is committed to the repo, so
publishing a new corpus never touches a tracked file. Pull it once per checkout before
multi-deck training:

```bash
./scripts/fetch_decks.sh           # newest decks-* release: verify + unpack into decks/
./scripts/fetch_decks.sh decks-v3  # or pin a specific version for reproducibility
```

Publish a new corpus (auto-increments to the next `decks-vN`):

```bash
./scripts/update_decks_release.sh <--notes> "…"
```

`fetch_decks.sh` picks up the highest-numbered release automatically. `build_decks_release.sh` is
the low-level builder it wraps if you only want the local tarball.

Every deck in a release comes from Limitless. The scraper also walks Bulbapedia, but its
archetype pages are historical lists whose cards predate the engine's pool, so all of them drop
as unresolved and none reach the corpus — see the card pool section of `scraper/README.md`.

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
    deck_sampler.py             DeckSampler protocol, FixedDeckSampler/PoolDeckSampler, and the seat-aware draw helper
    agent_deck_sampler.py       AgentDeckSampler: pins the agent to one deck, draws only its opponent from the pool
    opponent_pool.py            Self-play opponent pool (samples/holds frozen policy snapshots)
    snapshot_opponent_pool.py   OpponentPool that discovers new learner snapshots from disk (ParallelEnv-safe)
    pfsp_opponent_pool.py       Prioritized Fictitious Self-Play: weights league members by the learner's win rate against them
    archetype_index.py          Groups the deck corpus by archetype folder and numbers the resulting matchups
    level_buffer.py             Prioritized Level Replay buffer over archetype matchups (scores, staleness, win/loss tallies)
    curriculum_handles.py       Shared-memory channel carrying the level distribution from learner to env workers
    curriculum_deck_sampler.py  DeckSampler drawing each episode's matchup from that channel
    random_opponent.py          Uniform-random opponent baseline
  models/                     Actor-critic network, independent of the policy/training wiring
    backbone.py                  Backbone ABC + MLPBackbone (SetTransformer/TemporalTransformer/Recurrent planned)
    structured_obs_adapter.py    StructuredObsAdapter: embeds card IDs, normalizes scalars, set-pools zones
                                   (per seat for the board, see adapter.pokemon_seat_split), and emits the
                                   per-option token table the pointer head scores
    heads.py                     PointerHead (per-option scoring, the default) + LinearPolicyHead
                                   (flat slot-indexed logits, the baseline) + ValueHead (scalar critic)
    actor_critic.py              ActorCritic: shared trunk feeding both heads, tensordict-in/tensordict-out
    transformer.py                TransformerBackbone: self-attention trunk with pooling modes, per-entity token groups, and pointer-head option tokens
  policies/
    random_masked_policy.py     Uniform random policy over the action mask (stand-in for the future PPO actor)
    greedy_policy_opponent.py   Greedy opponent baseline built on a saved ActorCritic checkpoint
    inference.py                 SamplingPolicyAgent: main.py's submission agent, specs/checkpoint loading for Kaggle
    ppo_actor.py                 build_actor_critic / build_ppo_operator: assemble the ActorValueOperator from Hydra config
  training/
    trainer.py                  Trainer: parallel rollout collection through the configured collector
    base_trainer.py             BaseTrainer interface
    collectors.py                build_collector: the three TorchRL collector kinds, normalized to (rows, time) batches
    ppo_trainer.py               PPOTrainer: Trainer subclass running GAE or V-trace + ClipPPOLoss optimization
    env_factory.py               Builds TransformedEnv instances (deck + opponent + ActionMask) for the collector
    self_play.py                 build_opponent_factory: the picklable self-play league factory handed to each env worker
    evaluator.py                 Evaluator: scores the policy against a fixed opponent (readable curve under self-play)
    multi_evaluator.py           MultiEvaluator: runs several Evaluators, namespacing metrics per reference opponent
    curriculum.py                Curriculum: owns the level buffer, scores each collected batch, republishes the distribution
    cuda_memory_guard.py         CudaMemoryGuard: caps the process's GPU reservation so pressure raises OutOfMemoryError instead of killing the CUDA context
    cross_play.py                Round-robins frozen checkpoints into a win-rate matrix + Bradley-Terry Elo ranking
    pipe_timeout.py              apply_pipe_timeout: how fast an unresponsive ParallelEnv worker is detected (torchrl defaults to 2h47m)
    callbacks/                  Metric sinks; the trainer emits, these decide where it goes
      base.py                     TrainingCallback hooks + CallbackList (fan-out, propagates failures)
      snapshot_callback.py        SnapshotCallback: freezes the learner into the self-play league at a frame interval
      curriculum_callback.py      CurriculumStateCallback: dumps the level buffer's win/loss tallies at a frame interval
      cross_play_callback.py      CrossPlayCallback: scores the learner against its own snapshots, ranks them at run end
      wandb_callback.py           WeightsAndBiases: the only module that imports wandb
      wandb_fork_guard.py         WandbForkGuard: detaches W&B's blocking exit hooks in forked ParallelEnv workers
  train.py                    Hydra entry point (python -m src.train)
  eval_deck_field.py          Scores a saved agent per archetype across a deck field (python -m src.eval_deck_field)

conf/                        Hydra configs (config.yaml + env/, agent/, model/, train/, collector/, callbacks/, experiment/ groups)
  paths/default.yaml          Scheduler-independent input and output locations
  model/
    default.yaml                Composes one backbone + one head, holds shared dims (embed_dim, value_head)
    backbone/mlp.yaml            MLP baseline trunk (more backbones added as separate config files as they land)
    head/pointer.yaml            Per-option scoring head (default)
    head/linear.yaml             Flat slot-indexed logits head (the baseline the pointer head replaced)
  train/
    default.yaml                Keys shared by every training variant; the variants below override only what differs
    fixed_opponent.yaml          Default: fixed random opponent, no snapshotting; the control for self-play runs
    ppo_selfplay.yaml            Self-play league: snapshot interval, pool size, periodic fixed-opponent evaluation
  callbacks/
    wandb.yaml                  Default: Weights & Biases run (project/entity/group/tags/mode)
    none.yaml                    Console/Hydra log lines only; for throwaway runs
.env.example                 Template for untracked W&B, Kaggle, and submission defaults
scripts/                     Standalone dev scripts (not part of the training entry point)
  bench_throughput.py          Collection throughput benchmark (naive vs SerialEnv vs ParallelEnv)
  export_inference_checkpoint.py  Export assets for the repository-root inference entry point
  generate_obs_fixtures.py     Regenerates the committed observation fixtures in tests/fixtures/
  make_submission.py           Build a Kaggle .tar.gz and optionally submit it through the Kaggle CLI
  train_supervised.sh          Runs a training job and restarts it from its own train_state.pt on any crash
  train_tf_weighted_field.sh   15M-frame self-play run: transformer trunk, observation-weighted field, fixed eval panel
submission_analysis/          Kaggle submission tooling: `python -m submission_analysis <status|episodes|deck-report|scout>` (see its own README)
submission/
  main.py                      Kaggle entryfile template; `agent` is deliberately its final callable
  cg_api.py                    Pure-Python observation parser (no native simulator dependency)
  runtime.py                   Torch-only structured encoder/model/greedy inference implementation
checkpoint/                  Assets generated for the repository-root inference entry point
decks/                       Deck collections
docs/                        Design docs (torchrl_environment.md, game.md, research/curriculum-design.md)
tests/                       Unit tests (+ fixtures/: committed sample observations and card tables)
main.py                      Alternate inference entry point; not used by make_submission.py
slurm-conf/                  Slurm profiles, uv setup, and generic submission/training scripts
```

The **backbone** and **head** are independent Hydra config groups, so any backbone can be paired
with any head from the CLI or a sweep, e.g. `python -m src.train model/backbone=mlp model/head=linear`.

### Policy head: why `pointer` is the default

`model/head=pointer` scores each action slot from `[state_repr, option_repr_i]` with weights shared
across slots. `model/head=linear` — the original baseline — emits one logit per slot from the pooled
state alone, and the adapter hands it the option table as a *masked mean*. Mean pooling is
permutation-invariant, so under the flat head, shuffling the options leaves the logits bit-identical
while the correct action moves: the policy is structurally unable to condition on what an action
does, and the most it can represent is a prior over slot indices.

`model/head=pointer_dot` is the same idea with a scaled dot product instead of a shared MLP: cheaper
(one product per slot, no hidden widths to tune) and it scores the stop action as a real row of the
option table rather than from a separate branch. `pointer` is the default because it is the arm with
the measured win; `pointer_dot` is the transformer work's formulation, still being compared.

That is not a tuning problem, it is a ceiling, and it was the binding constraint on training — see
[`docs/architecture/pointer-head.md`](docs/architecture/pointer-head.md) for the measurements. `linear` is
kept as the control for that comparison; use `pointer` for real runs.

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

### Level curriculum

`env=curriculum_v2` replaces uniform deck sampling with Prioritized Level Replay over
deck-archetype matchups. The learner scores each matchup by how much systematic error the
critic still carries on it, and republishes a sampling distribution to the environment
workers after every batch, so training concentrates on the matchups it handles worst:

```bash
python -m src.train agent=ppo env=curriculum_v2 train=ppo_selfplay
```

Pair it with `train=ppo_selfplay`, which also enables PFSP over the self-play league. The
two are complementary: a hard matchup played against an obsolete snapshot is still a
trivial win, so prioritizing the deck axis while leaving the opponent axis uniform lets
them cancel.

Requires `env.deck_pool` and `env.mp_start_method=fork` — the distribution reaches the
workers through shared memory, which `spawn` would copy instead of share. Both are checked
at startup rather than silently degrading. Curriculum statistics are logged under
`curriculum/`, and the buffer's win/loss tallies are dumped periodically to the run
directory for later deck selection.

Design rationale is in [`docs/research/curriculum-design.md`](docs/research/curriculum-design.md).

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

### Cross-play and checkpoint selection

`train.cross_play=true` (with self-play on) scores the learner against its own
frozen history through the greedy serving path. During training it logs
`crossplay/vs_latest_snapshot` (am I still improving on my past self); at run end
it round-robins the checkpoints into a win-rate matrix and an order-free
Bradley-Terry **Elo** ranking, written as `crossplay_matrix.csv` /
`crossplay_elo.csv` in the run directory and logged as `crossplay/elo/<ckpt>`.
The Elo ranking is the tool for picking which checkpoint to submit.

### Exploitability (best-response)

```bash
python -m src.train --config-name ppo_best_response \
  train.best_response_checkpoint=/path/to/agent.pt
```
Freezes the given agent and trains a fresh learner to beat it across the corpus;
the learner's `eval/win_rate` is that agent's **exploitability** — how beatable a
dedicated best-responder finds it. A robust, Nash-like agent holds the
best-responder near 0.5; a brittle one is driven far above. This is the read the
ByteRL attack exposed that a self-play win-rate hides.

### Deck-pool-width sweep

`env.deck_pool_width=N` caps training to `N` archetypes while the held-out eval
set stays fixed, so a sweep measures how generalization scales with training
diversity:
```bash
python -m src.train --config-name ppo_selfplay_multideck --multirun \
  env.deck_pool_width=4,8,16,32
```
Compare `eval/archetype_win_rate_{mean,worst_quartile}` across the runs (grouped
by `env.deck_pool_width` in W&B).

Online W&B logging is required when selected: configure `WANDB_API_KEY` in an
untracked `.env` or run `wandb login --verify`. Authentication, connection, or
logging failures stop training. Use `wandb.mode=offline` only when local
recording for a later `wandb sync` is intentional. The final PPO checkpoint is
logged as a run-specific W&B model artifact by default; periodic self-play
snapshots stay local. Set
`wandb.log_checkpoints=false` to keep them local only.

### Kaggle submissions

Every PPO run writes a final checkpoint, even when `train.snapshot_interval=0`.
It appends that checkpoint to the ignored repository-local
`logs/checkpoint_keys.csv` and prints both its path and a 12-character SHA-256
key:

```text
checkpoint: .../checkpoints/snapshot_000000016384.pt
checkpoint-key: 5ac45db92f3e
```

Run submission commands from the repository root. `uv sync` installs the
official Kaggle CLI. Put `KAGGLE_API_TOKEN` in the untracked `.env`, run
`uv run kaggle auth login`, or configure another credential method supported
by the CLI.

Build the newest completed checkpoint (the bottom registry row) without
uploading it:

```bash
uv run python scripts/make_submission.py \
  --checkpoint latest \
  --label my-agent \
  --yes
```

Sampling from the learned masked distribution is the default. To use
deterministic highest-score selection instead, add `--action-selection greedy`.

To rebuild an existing label and upload it through the official CLI in one
command, add `--force --submit`:

```bash
uv run python scripts/make_submission.py \
  --checkpoint 5ac45db92f3e \
  --label my-agent \
  --submit \
  --yes \
  --force
```

**[`submission_analysis/`](submission_analysis/README.md)** covers the rest
of the Kaggle workflow: live submission status and leaderboard rank
(`status`), episode outcomes and replay download (`episodes`), a
replay-driven deck-refinement report with a prioritized "worth looking into"
list (`deck-report`), and scouting the top leaderboard teams' decks
(`scout`).

```bash
uv run python -m submission_analysis status --most-recent-n 2
uv run python -m submission_analysis episodes --download-replays --most-recent-n 2
uv run python -m submission_analysis deck-report --deck decks/example.csv
uv run python -m submission_analysis scout --deck decks/example.csv
```

Building a submission reads `.env`, resolves `latest` from `CHECKPOINT_KEYS_FILE` (default
`logs/checkpoint_keys.csv`), and verifies the recorded SHA-256 before building.
An explicit hash is resolved from the registry first; path/name lookup under
`CHECKPOINTS_DIR` remains available for legacy or unregistered checkpoints. It
defaults the deck to the checkpoint's recorded `env.deck0`, requires exactly
60 integer entries, and accepts `--deck PATH` for an explicit override. Without
`--submit`, it prints the exact `kaggle competitions submit ...` command but
does not contact Kaggle. `--force` replaces only the selected label's staging
directory and archive.

The uploaded archive is deliberately small and self-contained:

```text
main.py
cg_api.py
runtime.py
model.pt
model_config.json
deck.csv
submission_manifest.json
```

The builder copies `submission/main.py`, not the repository-root TorchRL-based
inference entrypoint. Kaggle executes the bundled entryfile with an empty
globals mapping and selects its final callable, so `agent` must remain the last
callable defined in the file. The validator supplies `torch`, but neither
TorchRL nor the repository's `cg` Python package. The bundle therefore carries
a pure-Python observation parser and Torch-only inference implementation; it
contains no TensorDict, Hydra, OmegaConf, or native simulator dependency.

Submission inference samples each legal choice from the learned masked
distribution by default, matching training behavior. Pass
`--action-selection greedy` for deterministic highest-score selection. Both
modes re-encode the partial selection before each subsequent choice.

The current portable runtime supports structured-observation checkpoints using
`MLPBackbone` or `TransformerBackbone` (every pooling mode, per-entity token
groups, learned segment embeddings, and the optional per-option tokens a
pointer head needs) paired with `LinearPolicyHead` or `PointerPolicyHead`. The
builder strictly reconstructs both the training model and portable model, so
an unsupported architecture or mismatched config fails before an archive is
produced. For an old bare state-dict checkpoint without embedded config, pass
its Hydra config with `--config path/to/.hydra/config.yaml`.

Every build performs a fail-closed Kaggle preflight before producing output. It
checks all bundled Python as Python 3.11, rejects dynamic imports, permits only
an explicit standard-library allowlist plus `torch` and bundled modules,
prepends the resolved bundle path and verifies that local imports resolve
there, reproduces Kaggle's empty-globals compile/execute/last-callable loader,
and calls the agent for both deck setup and a real model selection in an
isolated interpreter. It then extracts the finished archive, checks its exact
root file set and checkpoint/deck hashes, and repeats the same preflight against
the extracted upload bytes. Any failure aborts before the optional CLI submit.

Slurm support is kept separately in [`slurm-conf/`](slurm-conf/README.md). The
profiles select a normal Hydra config and add scheduler-specific overrides;
they do not participate in local Hydra composition. Complete launch configs are
available as `baseline`, `ppo`, `ppo_selfplay`, and `ppo_transformer` (the
Issue-45 self-attention backbone).

The first `ppo_transformer` run underperformed, so the candidate causes are
split into one `conf/experiment/tf_*.yaml` overlay each — optimizer steps,
trunk capacity, LR/entropy schedule, pre-LN, CLS readout, per-entity tokens,
pointer head — plus `tf_combined` stacking them all.
[`scripts/run_tf_diagnosis.sh`](scripts/run_tf_diagnosis.sh) submits the sweep,
one A100 job per arm sharing a W&B group; `--dry-run` prints without
submitting, and naming an arm submits just that one.

## CI

GitHub Actions runs linting, type-checking, and tests for pull requests that
are ready for review. Draft pull requests intentionally skip CI to stay within
the GitHub Actions free-plan budget.

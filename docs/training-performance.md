# Training throughput profile

Measured 2026-07-31, after the fix in
[`research/curriculum-transport-fix.md`](research/curriculum-transport-fix.md)
made the environment workers actually fork.

**Headline results**

- `env.num_workers: 16` leaves ~40% of achievable throughput unused; **32 is the
  recommended setting**.
- `agent.device=cuda` is worth **7.8×**. Never run PPO training on CPU.
- The self-play league costs **46%** of throughput — the single largest expense.
- The level curriculum costs **nothing measurable**.
- Forking rather than spawning workers cut startup from **52s to 4s**.

Added 2026-08-09 for issue #86 (§6), measured on a 7800X3D (8 cores, 16
threads) with a 24 GiB CUDA device, against the `tf-ptr-weighted-15m-s42`
configuration with its self-play league populated:

- The pegged main process is running the **policy forward**, not stacking
  tensordicts or waiting on IPC. The suspected cause was wrong.
- **`collector.type: multi_sync` with `agent.collector_device: cpu` is worth
  1.6x end to end**, 251 to ~400 fps, with the PPO update included and the
  learning diagnostics unchanged.
- The genuinely asynchronous collectors buy nothing. `multi_async` ties
  `multi_sync` and costs on-policy batches; `async_batched` cannot collect a
  batch on this observation at all (§6.6).
- After the change the **PPO update is the bottleneck**, 56% of the loop.
  Further collector work has little left to win.

---

## Setup

| | |
|---|---|
| CPU | AMD Ryzen 7 7800X3D — **8 cores / 16 threads**, 96 MB L3 (3D V-Cache) |
| RAM | 32 GB on the host; **~23 GB visible to WSL2** |
| GPU | NVIDIA RTX 4090 — 24 GB **GDDR6X**, 384-bit bus (~1 TB/s) |
| OS | Linux 5.15 (WSL2) |
| Python | 3.13 |

Note the core count: `nproc` reports 16, but those are SMT threads over 8
physical cores. The worker-scaling numbers below should be read against 8 cores,
not 16.

Environment: `env=curriculum_v2` — merged corpus (`decks/` + `decks_v2/`)
restricted to archetypes with ≥15 lists, giving 37 archetypes, 1,759 decks and
1,369 curriculum levels.

Training: `agent=ppo train=ppo_selfplay`, `frames_per_batch=4096`,
`num_epochs=4`, `sub_batch_size=4096`.

Every run used `callbacks=none`, `seed=0`, `set_seed=true`, and unless the
measurement was specifically about them, `train.eval_interval=0` and
`train.snapshot_interval=0` so neither evaluation nor snapshot writing
contaminated the timing.

Baseline command:

```bash
python -m src.train agent=ppo env=curriculum_v2 train=ppo_selfplay callbacks=none \
  agent.device=cuda collector.total_frames=100000 \
  train.eval_interval=0 train.snapshot_interval=0 \
  env.num_workers=<N> seed=0 set_seed=true hydra.run.dir=<dir>
```

`fps` throughout is the figure the trainer reports at exit: total frames divided
by total training wall time, so it includes the PPO update, not collection alone.

---

## 1. Worker scaling

100,000 frames, `agent.device=cuda`, league disabled.

| workers | fps | vs default | steps/worker/batch |
|---|---|---|---|
| **16** (current default) | 1,498 | — | 256 |
| 24 | 1,814 | +21% | 171 |
| **32** (recommended) | **2,126** | **+42%** | 128 |
| 48 | 2,324 | +55% | 85 |
| 64 | 2,500 | +67% | 64 |

Throughput was still climbing at 64 workers on **8 physical cores** — 8×
oversubscription, and still gaining. Workers are therefore **latency-bound, not
CPU-bound**: each spends most of its time inside the engine and waiting on
`ParallelEnv`'s lockstep synchronisation, so adding processes keeps paying well
past the core count.

The 7800X3D's 96 MB L3 likely flatters these numbers relative to a
cluster CPU: the engine's per-battle state and the card tables are small, so many
concurrent workers can stay resident in cache where a conventional CPU would be
going to DRAM. Expect the scaling curve to flatten earlier on a cluster node.

Memory is not the constraint at these worker counts: 9 GB in use at 64 workers.
The ceiling to watch is WSL2's ~23 GB cap, not the host's 32 GB — a forked worker
adds little (copy-on-write shares the parent's card tables and deck pool), but
the cap is what would bite first if worker counts were pushed much further, and
it is raisable via `.wslconfig` if needed.

### Why 32 and not 64

`frames_per_batch` is fixed at 4,096, so steps collected per worker per batch
falls as workers rise (last column above). Episodes run roughly 130 steps, so:

- at **16–32** workers each worker completes about one episode per batch;
- at **64** workers no episode finishes inside a batch, and GAE leans much
  harder on value bootstrapping at fragment boundaries.

32 captures most of the speedup while keeping the fragment length comparable to
an episode. Going beyond it would mean raising `frames_per_batch` in step, which
changes the PPO batch size and is a separate decision.

## 2. Policy device

100,000 frames, 32 workers.

| `agent.device` | fps |
|---|---|
| `cuda` | **2,091** |
| `cpu` | 267 |

**7.8×.** The environments always step on CPU regardless; this is purely the
policy forward and the PPO update.

GPU utilisation sampled once per second during a 64-worker run:

```
66%   7%   7%   95%   8%
```

Spiky and mostly idle — the GPU bursts during each update and then waits for the
next batch. Consistent with the CPU-bound picture above, and it means there is
headroom for a larger network before the GPU becomes the constraint.

Neither VRAM capacity nor bandwidth is anywhere near being exercised: peak usage
was ~6.5 GB of 24 GB, and the current MLP over an 824-dimensional observation is
far too small to stress GDDR6X. That headroom is the relevant number for the
planned Transformer backbone — the constraint on scaling the network here is
CPU-side environment stepping, not the GPU.

## 3. Self-play league

100,000 frames, 32 workers. Compared `train.snapshot_interval=0` (workers face
the built-in random opponent, no network forward) against `=20000` (workers load
snapshots and run them).

| league | fps | cost |
|---|---|---|
| off | 2,171 | — |
| **on** | **1,165** | **−46%** |

**The largest single cost in the pipeline.** Every opponent move is a full
forward pass through the actor-critic, executed on CPU inside the worker
process.

This is inherent to the opponent-in-env design, and the CPU placement is
deliberate — 32 per-worker CUDA contexts would cost far more than the small MLP
forward they would accelerate. It is not something to change now, but it is the
obvious lever if throughput ever becomes the binding constraint: a distilled or
deliberately smaller opponent network would buy most of that 46% back.

Caveat: measured over 100k frames with a 20k snapshot interval, so the league
held only a handful of members. A fully populated `pool_size=5` league in a long
run may differ somewhat, though the per-move cost should not.

## 4. Curriculum overhead

100,000 frames, 16 workers.

| | fps |
|---|---|
| `env=multideck_v2` (uniform) | 1,496 |
| `env=curriculum_v2` (PLR) | 1,522 |

**No measurable cost** — the difference is within run-to-run noise. Worth
recording explicitly because the curriculum recomputes and publishes a
1,369-element distribution every batch, and every worker reads it at every
episode reset. The shared-memory channel makes both effectively free.

## 5. Startup: fork vs spawn

20,000 frames, 32 workers. `startup` is wall time minus collection time derived
from the reported fps.

| start method | wall | collect | startup |
|---|---|---|---|
| **fork** (current) | **14s** | 10s | **4s** |
| spawn | 65s | 13s | 52s |

**13× faster startup.** Under `spawn` every worker re-imports torch and torchrl
from scratch; under `fork` they inherit the parent's memory.

This was a side benefit of the curriculum transport fix, which forces the
process-wide start method before `ParallelEnv` construction. Before it, workers
were being spawned despite `env.mp_start_method: fork`. At 32 workers that was
~50s of dead time on every run, which made short debug runs disproportionately
painful.

---

## 6. Which collector to use (issue #86)

Everything above this section was measured with fps alone, which cannot say
*why* a configuration is slow. `scripts/bench_throughput.py --mode profile` can:
it splits the collection wall clock into policy forward, environment step (the
IPC round trip plus the workers' own work) and the collector's own tensordict
bookkeeping, and reports CPU for the main process and the worker pool
separately.

The starting point was `tf-ptr-weighted-15m-s42`: 480 fps at the collector,
**main process pegged at 99.5% of one core**, workers 13-15% each, **300% total
CPU out of 1600%**, GPU bursty at 40% mean. Collection, not the update, was the
constraint, and one process was doing all of it.

### 6.1 How these numbers were produced

All of §6 is one machine: a 7800X3D (8 physical cores, 16 threads) with a
24 GiB CUDA device, under WSL2. Numbers on another box will differ; §6.7 is how
to regenerate them.

Two scenarios appear below and they do not agree, which is the main lesson of
this section:

- **collection only**, driving the base `Trainer` whose `_update` is a no-op.
  Isolates the collector but excludes the PPO update.
- **end to end**, through `src/train.py` with the update, the deck pool and a
  populated self-play league. This is what a run actually costs.

Both use the `tf-ptr-weighted-15m-s42` architecture: transformer trunk with
`token_groups=[pokemon]`, `pointer` head, `embed_dim=256`, `entity_dim=128`,
`option_target_state`, `pokemon_seat_split`, over the observation-weighted
136-archetype pool.

An earlier revision of this section reported collection-only numbers taken
without the league. Three of its conclusions were wrong, and each is corrected
below. **Do not draw a collector conclusion from a league-free, update-free
measurement.**

### 6.2 The suspected cause was wrong

The hypothesis in the issue was tensordict stacking and IPC of the large nested
observation. It is neither. Under `collector.type=sync` with the league, 16
workers:

| bucket | share of collection wall clock |
|---|---|
| policy forward | 25% (cuda) / 38% (cpu) |
| env step, i.e. the IPC round trip the barrier costs | 66% / 57% |
| collector tensordict bookkeeping, the suspected culprit | 9% / 5% |

**The main process is pegged running the policy and waiting on workers, not
marshalling data.** The bookkeeping the issue suspected is the smallest bucket
in every configuration measured.

This is also why removing the barrier was never the answer. It leaves the same
serialized forward pass in the same single process.

### 6.3 What each collector kind is worth

End to end, 49152 frames, `agent.device=cuda`, league populated, eval off:

| collector | collection device | workers | fps | vs baseline |
|---|---|---|---|---|
| `sync` | cuda | 16 | 251 | 1.00x |
| `multi_sync` | cpu | 16 | 382 | 1.52x |
| `multi_sync` | cpu | 12 | 402 | 1.60x |
| `multi_async` (GAE) | cpu | 16 | 404 | 1.61x |
| `multi_async` (V-trace) | cpu | 16 | 402 | 1.60x |
| `async_batched` | cpu | 16 | does not collect a batch (§6.6) | removed |

**`multi_sync` is the answer.** Giving every worker its own copy of the policy
parallelises the one thing that was serialized. The main process drops from 99%
of a core to 6%, and total CPU rises from 307% to ~1250% of 1600%. It is also
the only option here that keeps batches on-policy: its workers sit idle between
handing a batch over and being told to continue, so a weight push in that gap
reaches all of them before the next batch starts. No V-trace required.

**`multi_async` buys nothing.** It ties `multi_sync` within noise while costing
the on-policy guarantee, requiring V-trace, and being incompatible with the
level curriculum (§6.5). Its one structural advantage, that workers keep
collecting through the update, does not show up in the measurement even though
the update is 56% of the loop.

**V-trace is correct but idle here.** 402 fps against GAE's 404 on the same
collector, i.e. it costs nothing and corrects a drift that is not large enough
to matter. It exists for the asynchronous path; nothing currently recommends
taking that path.

### 6.4 After this change, the update is the bottleneck

The per-frame decomposition is consistent across both collectors:

| | collection | update | total | fps |
|---|---|---|---|---|
| `sync` | 2.52 ms | 1.46 ms | 3.98 ms | 251 |
| `multi_sync` | 1.16 ms | 1.46 ms | 2.62 ms | 382 |

The update is unchanged by the collector, as it must be, and its share rises
from 37% to 56%. **Collection is no longer where the time goes.** The next
lever is the update itself (`sub_batch_size`, `num_epochs`, AMP), not the
collector.

This also explains why the collection-only speedup (2.15x, 397 to 861 fps) is
larger than the end-to-end one (1.52x). Quoting the collection-only figure as
the run's speedup would be wrong by 40%.

### 6.5 What `multi_sync` costs and what it needs

It puts a policy copy in every worker process. Two consequences:

- **Collection must run on the CPU.** Not an optimisation, a requirement: the
  workers are forked, and CUDA refuses to initialize in a forked child, so a
  CUDA collection policy dies with `Cannot re-initialize CUDA in forked
  subprocess`. `build_collector` rejects the combination with a message naming
  the setting (`tests/test_collectors.py` pins it).

  Do *not* fix this with `agent.device=cpu`, which drags the PPO update onto the
  CPU and gives back the 7.8x from section 2. The two devices are separately
  configurable: set **`agent.device=cuda` with `agent.collector_device=cpu`**,
  so the update keeps the GPU while collection runs one CPU policy copy per
  worker.
- **Weights must be pushed to the workers after every update.** `Trainer._collect`
  does this and `tests/test_collectors.py` pins it, including across the
  cuda/cpu boundary. Without it a run collects at full speed against the weights
  the workers forked with and looks healthy while learning nothing.
`multi_sync` is compatible with the level curriculum; `multi_async` is not. The
curriculum accumulates a residual per collector row and commits it when that row
reports `done`, which assumes row `r` of the next batch continues the same
environment as row `r` of this one. `sync` and `multi_sync` both honour that.
`multi_async` yields rollouts in completion order and does not, so
`PPOTrainer` rejects that pairing at construction rather than silently scoring
one matchup with another matchup's evidence.

### 6.6 `async_batched` cannot carry this observation

It does not run. `AsyncEnvPool` ships every transition through a
`multiprocessing.Queue`, and torch moves each leaf tensor by allocating a fresh
shared-memory segment that the receiver maps. This environment's observation is
~288 leaf tensors (9 phase groups of 32), so one transition costs several
hundred mappings, and the receiving process exhausts Linux's `vm.max_map_count`
(65530 by default) within a few hundred steps:

```
RuntimeError: unable to mmap 124 bytes from file </torch_...>:
Cannot allocate memory (12)
```

That is the mapping limit, not memory: RAM, `/dev/shm` and the file-descriptor
limit were all far from exhausted, and it fails identically under both
`file_descriptor` and `file_system` sharing. At 16 workers it hangs rather than
crashing, which is what thrashing at the limit looks like.

The batched environments every other kind uses allocate one shared tensordict
at startup and write into it in place, so they never map per step and never hit
this. It is a property of that transport, not of asynchrony.

**The kind was removed after this measurement.** `CollectorKind.ASYNC_BATCHED`,
its stream assembler, its four `conf/collector/default.yaml` settings
(`max_batch_size`, `min_batch_size`, `server_timeout`, `env_backend`) and the
mmap-failure recognizer are gone, since a kind that cannot collect a batch is
not a configuration anyone should be able to select. This section is the record
of why. Raising `vm.max_map_count` may be enough to make `AsyncBatchedCollector`
run; re-adding it means reverting that removal and re-measuring.

### 6.7 Worker count: measure it, do not carry a number over

Section 1 found throughput still climbing at 8x oversubscription and recommended
pushing `env.num_workers` well past the core count. That holds only while
workers are latency-bound. Under `multi_sync` they are not, since each one now
runs a policy forward per step.

Collection-only, with the league:

| workers | 6 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|
| fps | 420 | 665 | 830 | 861 | 745 |

The peak is at 16, the **thread** count, not 8, the physical core count. An
earlier revision of this section recommended sizing to physical cores; on this
machine that costs 23%.

End to end the curve is flat and noisy: 10 workers 396, 12 workers 402/471/415
over three runs, 14 workers 400, 16 workers 382/392. **The spread at one
setting is as wide as the spread between settings**, so the worker count is not
resolvable at this run length, and short runs additionally penalise larger
counts because forking N workers that each load the deck pool and the league is
a fixed cost inside a two-minute window. Over 15M frames that cost vanishes,
which is why the collection-only ordering is the better guide.

**Keep `env.num_workers: 16` and do not tune it further.** The collector choice
is a 60% gain against a 15% noise floor; the worker count is inside the noise.

### 6.8 Reproducing this

The synthetic scenario (no deck pool, no league) is the default and is the one
that misled the earlier revision. Pass `--config-name` to profile a real run:

```bash
uv run python scripts/bench_throughput.py --mode profile --policy ppo \
  --collector multi_sync --device cpu --num-workers 16 \
  --config-name ppo_selfplay_multideck \
  --override paths.data_dir=decks \
  --override deck_corpus=heuristic-resolved \
  --override env.agent_deck=null \
  --override env.deck_weighting=observation \
  --override model/backbone=transformer \
  --override model/head=pointer \
  --override "model.backbone.token_groups=[pokemon]" \
  --override model.backbone.option_tokens=true \
  --override train.pool_size=5 \
  --override train.snapshot_interval=200000 \
  --checkpoint-dir outputs/<group>/<run>/checkpoints
```

`--checkpoint-dir` matters: without it the league is empty and every worker
faces the random warmup opponent, which is not what a run in progress costs.
`--policy ppo` matters too; under the default random policy the policy-forward
bucket measures a `multinomial` call rather than the network the real runs
collect with.

For end-to-end fps, run `src/train.py` itself with a small
`collector.total_frames` and `train.eval_interval=0`, pointing
`train.checkpoint_dir` at an absolute path holding a few snapshots. An absolute
path is used as given, so the league can be pre-populated.

---

## Recommendations

1. **Set `collector.type: multi_sync` with `agent.device=cuda` and
   `agent.collector_device=cpu`.** 1.6x end to end on the reference machine,
   251 to ~400 fps, with ESS and `clip_fraction` unchanged. Batches stay
   on-policy, so nothing else about the run has to change (§6.3).
2. **Leave `env.num_workers` at 16** under `multi_sync`, and do not tune it: the
   differences between 10 and 16 are inside run-to-run noise (§6.7).
3. **Always `agent.device=cuda`** for the update, under every collector.
4. **Leave the curriculum on** when wanted; it is free. It cannot be combined
   with `multi_async`, which is rejected at construction (§6.5).
5. **Do not use `multi_async`.** It ties `multi_sync` while costing on-policy
   batches. `async_batched` does not run and has been removed (§6.6).
6. If throughput is still binding, attack **the PPO update**, which is 56% of
   the loop after this change, and the **league opponent forward**, which
   section 3 measured at 46% of collection.

## Methodology caveats

- **Single measurement per configuration.** No repeats, so run-to-run noise is
  unquantified. Differences under ~5% should not be read as real — which is
  exactly why the curriculum overhead is reported as "no measurable cost" rather
  than as a specific number.
- **`startup` is derived**, not instrumented: wall time minus `frames / fps`.
  Since the reported fps already excludes setup, this is a reasonable estimate
  but not a direct measurement.
- **fps includes the PPO update.** Collection-only throughput was not isolated;
  an attempt to measure it with `agent=random` produced no comparable figure and
  was dropped rather than reported.
- **§6 was measured on different hardware from §§1–5**, and reports no absolute
  figures for that reason; run §6.5 on the machine you care about.
- **§6 ran without the self-play league or evaluation**, so it profiles a clean
  collection-and-update pipeline, not what a real training run sees.
- Measurements come from a WSL2 environment on a 7800X3D; absolute numbers on the
  cluster will differ, though the relative ordering should hold. The worker
  scaling curve in particular is the least portable result — it depends on core
  count and on the unusually large L3, so the optimal `num_workers` should be
  re-measured on the cluster rather than carried over.

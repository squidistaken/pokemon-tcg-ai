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

Added 2026-08-08 for issue #86 (§6). Absolute figures there are left out on
purpose — they vary widely across the machines this is run on, and §6.5 is how
each one regenerates its own:

- The pegged main process is running the **policy forward**, not stacking
  tensordicts or waiting on IPC. The suspected cause was wrong.
- **`collector.type: multi_sync` is the win.** It unpegs the main process and
  costs no on-policy guarantees.
- The genuinely asynchronous collectors are *not*: `multi_async` gains less and
  costs on-policy batches, and `async_batched` is an order of magnitude slower.

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
going to DRAM. Expect the scaling curve to flatten earlier on Habrok.

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

## 6. Where collection time actually goes (issue #86)

Everything above this section was measured with fps alone, which cannot say
*why* a configuration is slow. `scripts/bench_throughput.py --mode profile` can:
it splits the collection wall clock into policy forward, environment step (the
IPC round trip plus the workers' own work) and the collector's own tensordict
bookkeeping, and reports CPU for the main process and the worker pool
separately.

The starting point was `tf-ptr-weighted-15m-s42`: 480 fps at the collector,
**main process pegged at 99.5% of one core**, workers 13-15% each, **300% total
CPU out of 1600%**, GPU bursty at 40% mean. Collection, not the update, is the
constraint, and one process is doing all of it.

Absolute numbers below are deliberately omitted: they differ by an order of
magnitude across the machines this is run on, and the point of the profiler is
that each one regenerates its own. What follows is what the *shape* of the
result was on every machine tried.

### 6.1 The suspected cause was wrong

The hypothesis in the issue was tensordict stacking and IPC of the large nested
observation. It is neither. Under `collector.type=sync` with the transformer +
pointer architecture:

- **policy forward: the majority of the loop**, whether the policy is on CPU or
  on CUDA;
- environment step, i.e. the IPC round trip the barrier costs: roughly a
  quarter;
- the collector's tensordict bookkeeping, the suspected culprit: the smallest of
  the three.

**The main process is pegged running the policy, not marshalling data.** Moving
the policy to CUDA helps but does not change the shape of the problem: at a
batch size equal to the worker count the forward is dominated by per-step Python
and host/device transfers rather than by GPU arithmetic.

This is why the barrier was never the real issue. Removing it leaves the same
serialized forward pass in the same single process.

### 6.2 What each collector kind does about it

**`multi_sync` is the answer.** Giving every worker its own copy of the policy
parallelises the one thing that was serialized: the main process drops from
~100% of a core to under 15%, and total CPU rises from a small fraction of the
machine to near saturation. It was the fastest of the four on every machine
tried. It is also the *only* asynchronous-collection option that keeps batches
on-policy: its workers sit idle between handing a batch over and being told to
continue, so a weight push in that gap reaches all of them before the next batch
starts. No V-trace required.

**`multi_async` costs learning guarantees for less speed.** It gains over
`sync`, but by less than `multi_sync` does, and its batches straddle optimizer
steps. It is wired up, and V-trace with it, but no measurement has yet given a
reason to prefer it.

**`async_batched` is an order of magnitude slower, not faster.** It removes the
barrier and then adds a second IPC hop: every step goes environment ->
coordinator thread -> inference server -> back, unbatched, and its main process
burns *more* than one core spinning on transport. The barrier was never what
cost the time.

### 6.3 `multi_sync` inverts the worker-count advice

Section 1 found throughput still climbing at 8x oversubscription and recommended
pushing `env.num_workers` well past the core count. That holds only while
workers are latency-bound. Under `multi_sync` they are not -- each one now runs
a policy forward per step -- and oversubscription starts costing throughput
rather than buying it. Under `sync` the same sweep is flat, because the parent
is the bottleneck either way.

**Under `multi_sync`, size `env.num_workers` to the physical core count**, and
measure it per machine rather than carrying a number over.

### 6.4 What this costs and what it needs

`multi_sync` puts a policy copy in every worker process. Two consequences:

- **Keep the collection policy on CPU.** On CUDA it opens one context per
  worker for batch-size-1 forwards, which is the case a GPU is worst at. The
  trainer logs a warning if you do it anyway.

  Do *not* do this by setting `agent.device=cpu`, which would drag the PPO
  update onto the CPU as well and give back the 7.8x section 2 measured. The two
  devices are separately configurable: set **`agent.device=cuda` with
  `agent.collector_device=cpu`**, so the update keeps the GPU while collection
  runs one CPU policy copy per worker. Weight pushes move across that boundary
  (pinned by `tests/test_collectors.py`).
- **Weights must be pushed to the workers after every update.** The trainer does
  this (`Trainer._collect`), and `tests/test_collectors.py` pins it. Without it
  a run collects at full speed against the weights the workers forked with and
  looks perfectly healthy while learning nothing.

Not measured, and worth measuring before a long run: the profiler runs without
a self-play league, which section 3 found costs 46% of throughput by running an
opponent forward inside every worker. `multi_sync` adds a *second* per-worker
forward on top of that, so on a core-bound node the two may contend in a way
idle environments do not show.

### 6.5 Reproducing this

```bash
# Where the time goes, for one configuration.
uv run python scripts/bench_throughput.py --mode profile --policy ppo \
  --collector sync --device cuda --num-workers <cores> \
  --model-override model/backbone=transformer \
  --model-override model/head=pointer_dot \
  --model-override model.backbone.option_tokens=true

# The same, per collector kind.
... --collector multi_sync --device cpu
```

`--policy ppo` matters: under the default random policy the policy-forward
bucket measures a `multinomial` call rather than the network the real runs
collect with, and the conclusion above inverts.

---


## Recommendations

1. **Set `collector.type: multi_sync`** with **`agent.device=cuda`** and
   **`agent.collector_device=cpu`** — the change that unpegs the main process,
   and the fastest option on every machine tried, while the update keeps the
   GPU. Batches stay on-policy, so nothing else about the run has to change
   (§6).
2. **Size `env.num_workers` to the physical core count under `multi_sync`.** The
   §1 advice to oversubscribe applies to `sync` only (§6.3).
3. **Always `agent.device=cuda`** for PPO training, under every collector.
4. **Leave the curriculum on** when wanted; it is free — but note it cannot be
   combined with `multi_async`, which is rejected at construction (§6.4).
5. If throughput is still binding, attack the **league opponent forward** — it is
   46% of the budget under `sync` and nothing else comes close.

At 32 workers with the league active under `sync`, a 2M-frame arm takes roughly
**16 minutes** of training (plus evaluation), against ~22 minutes at the old
default.

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
  re-measured on Habrok rather than carried over (see
  [`habrok_guide.md`](habrok_guide.md)).

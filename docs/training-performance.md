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

## Recommendations

1. **Set `env.num_workers: 32`** (from 16) — +42% for no change in learning
   dynamics beyond a shorter per-worker fragment.
2. **Always `agent.device=cuda`** for PPO training.
3. **Leave the curriculum on** when wanted; it is free.
4. If throughput becomes binding, attack the **league opponent forward** first —
   it is 46% of the budget and nothing else comes close.

At 32 workers with the league active, a 2M-frame arm takes roughly **16 minutes**
of training (plus evaluation), against ~22 minutes at the old default.

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
- Measurements come from a WSL2 environment on a 7800X3D; absolute numbers on the
  cluster will differ, though the relative ordering should hold. The worker
  scaling curve in particular is the least portable result — it depends on core
  count and on the unusually large L3, so the optimal `num_workers` should be
  re-measured on Habrok rather than carried over (see
  [`habrok_guide.md`](habrok_guide.md)).

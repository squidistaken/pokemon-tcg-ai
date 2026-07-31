# Curriculum fix: workers were spawned, not forked

## The fix

One line in `Trainer._make_vec_env` (`src/training/trainer.py`):

```python
if self._use_parallel_env:
    torch_mp.set_start_method(self._mp_start_method, force=True)   # <-- added
    return ParallelEnv(
        num_workers=len(self._env_factories),
        create_env_fn=self._env_factories,
        mp_start_method=self._mp_start_method,
        serial_for_single=self._serial_for_single,
    )
```

Plus `import torch.multiprocessing as torch_mp`.

## Why it matters

The level curriculum hands its sampling distribution to the environment workers
through **shared-memory tensors** (`CurriculumHandles`). That only works if the
workers are **forked**: a forked child inherits the parent's memory mapping, so
when the learner republishes after each batch, every worker sees the new values
immediately.

If the workers are **spawned**, the tensors are pickled instead and each worker
receives a **private copy**. Nothing errors. Collection runs at full speed. The
learner keeps updating its copy every batch; the workers keep reading theirs,
frozen at whatever it held the moment they started.

In the real training run the workers were being spawned — confirmed by 16 child
processes running `multiprocessing.spawn.spawn_main` — despite
`env.mp_start_method: fork` in the config and `ParallelEnv` being handed
`mp_start_method="fork"` explicitly. Forcing the process-wide start method
before construction makes them fork.

## What the workers were frozen on

The stale copy held the very first distribution, published by
`Curriculum.__init__` before any level had been visited.

At that point every level scores `inf`, so all scores tie. `LevelBuffer.distribution()`
ranks with `np.argsort(-scores, kind="stable")`, and a stable sort breaks ties by
**array position**. Ranks therefore became buffer index, and `(1/rank)^(1/0.9)`
turned that into a steep Zipf decay over `pair_id`.

Because `pair_id = agent * K + opponent`, slot 0 is (archetype 0, archetype 0).
Archetype 0 is `alakazam-dudunsparce` (alphabetically first), which is why it
owned the entire top of the visit table with its opponents in alphabetical order
and monotonically decaying counts.

The result was not a neutral failure. Sampling ended up *anti*-correlated with
the curriculum's own scores:

- the 20 most-visited levels averaged `|score|` **0.0170**
- the 200 least-visited averaged **0.0936**
- the single highest-scoring level (0.5618) received **3 episodes**
- one matchup absorbed **12.8%** of all training

## Evidence

Identical 250k-frame runs (`env=curriculum_v2`, 1369 levels), before and after:

| measurement | before | after |
|---|---|---|
| r(realized draws, index-Zipf) | +0.888 | **+0.031** |
| r(realized draws, published distribution) | −0.003 | +0.004 |
| max visits on a single level | 300 (slot 0) | **24** (slot 66) |
| levels matured (≥ `min_visits`) | 173 / 1369 | **308 / 1369** |

The index-ordered sweep is gone, no single matchup dominates, and coverage
nearly doubled at equal budget.

The learner side was **correct throughout** and needed no change — reconstructing
`distribution()` from the state dumps gives `r(dist, |score|) = 0.63` by end of
run, no index dependence, and an argmax that migrates as levels mature. Only the
delivery to the workers was broken.

## What is not settled

- **The mechanism is not fully isolated.** It is confirmed empirically that
  forcing the start method fixes the real path, but not *why* `ParallelEnv`'s own
  `mp_start_method` argument was insufficient in that process. Some interaction of
  CUDA initialization, the `Collector`, and `ParallelEnv`'s lazy worker start.
- **The regression test does not reproduce the failure.**
  `test_workers_track_a_republished_distribution_through_the_trainer_path` passes
  both with and without the fix: in a bare pytest process `ParallelEnv` does
  honour its explicit `fork` argument, even with the global method forced to
  `spawn` first. The assertion is correct and worth keeping, but it would not have
  caught this bug and should not be treated as the safety net.

## Recommended follow-up

Log `r(realized_visits, published_distribution)` as a training metric. Every
existing diagnostic — `curriculum/matured`, `visits_mean`, `score_mean` — looked
healthy for the entire duration of this failure. That one number would have made
it obvious within the first batch, and it is the only cheap guard that does not
depend on reproducing the multiprocessing conditions in a test.

## Status of prior results

The completed 2M-frame `curriculum-s0` run predates this fix and does not measure
a curriculum. It is not a usable baseline either, since index-Zipf sampling is not
uniform. The three-arm experiment needs re-running.

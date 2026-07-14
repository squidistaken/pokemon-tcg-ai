# Collectors & replay buffers

Source: <https://docs.pytorch.org/rl/stable/reference/collectors.html>
Buffers: <https://docs.pytorch.org/rl/stable/reference/data.html>

## Collectors

Run a policy against an env and **yield batches of transitions as TensorDicts**. Iterate
over the collector in the training loop.

```python
from torchrl.collectors import Collector

collector = Collector(
    env,
    policy=policy_module,
    frames_per_batch=1000,   # transitions yielded per iteration
    total_frames=100_000,    # total collection budget (loop stops after)
    split_trajs=False,
    device=device,
)
for tensordict_data in collector:
    ...  # tensordict_data shape ~ [frames_per_batch] (or [n_envs, steps] if batched)
```

> As of TorchRL v0.13 the old `SyncDataCollector` / `MultiSyncDataCollector` /
> `MultiaSyncDataCollector` aliases were **removed**. `Collector` and `MultiCollector`
> are the canonical classes now — the generic constructor arg is `policy=` (positional
> `policy_module` still works on `Collector`, but keyword is preferred since `MultiCollector`
> requires it).

Types:
- **`Collector`** — single-process, collect on the training worker (was `SyncDataCollector`).
- **`MultiCollector(..., sync=True|False)`** — parallel workers; `sync` selects the delivery mode:
  - `sync=True` → equivalent to **`MultiSyncCollector`** (on-policy: PPO/A2C — all workers
    finish before a batch is delivered, so data matches the current policy). Replaces
    `MultiSyncDataCollector`.
  - `sync=False` → equivalent to **`MultiAsyncCollector`** (off-policy: DQN/SAC — batches are
    delivered first-come-first-served; the policy may lag slightly). Replaces
    `MultiaSyncDataCollector`.

  `MultiSyncCollector`/`MultiAsyncCollector` can also be imported and used directly instead of
  going through `MultiCollector(sync=...)` — they're the same classes.
- **`AsyncCollector`** — runs a single `Collector` on a separate process (new in the current API).
- **`AsyncBatchedCollector`** — pairs per-env threads with an `AsyncEnvPool` and an
  `InferenceServer` to auto-batch policy inference across many async envs, overlapping env
  stepping with GPU inference for higher throughput (new, no equivalent in the old API).
- **`BaseCollector`** — abstract base class shared by all of the above.

```python
from torchrl.collectors import MultiCollector

def make_env():
    return env_fn()

collector = MultiCollector(
    create_env_fn=[make_env] * 4,   # 4 parallel workers
    policy=policy_module,
    frames_per_batch=1000,
    total_frames=100_000,
    sync=True,       # PPO/A2C-style: wait for every worker each batch
    cat_results="stack",  # keep the worker dim separate (default since v0.5)
)
for tensordict_data in collector:
    ...  # [num_workers, *env_batch_dims, frames] with cat_results="stack"
collector.shutdown()
```

Keep inference policy weights fresh: call `collector.update_policy_weights_()` each iteration,
or pass `update_at_each_batch=True` to `MultiCollector`/`MultiSyncCollector`/`MultiAsyncCollector`
to do this automatically before every batch. For multi-process/distributed setups (shared
memory, `torch.distributed`, RPC, Ray), pass `weight_sync_schemes` to configure the transport —
see the [Weight Synchronization docs](https://docs.pytorch.org/rl/stable/reference/collectors_weightsync.html).

## Replay buffers

```python
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

# PPO: sample minibatches WITHOUT replacement from the just-collected batch
rb = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)
rb.extend(tensordict_data.reshape(-1).cpu())
minibatch = rb.sample(sub_batch_size)

# Off-policy (DQN/SAC): large persistent buffer, random sampling
rb = TensorDictReplayBuffer(storage=LazyTensorStorage(1_000_000), batch_size=256)
rb.extend(tensordict_data)
batch = rb.sample()
```

Storages: `LazyTensorStorage` (RAM, lazy-inits on first extend), `LazyMemmapStorage`
(memory-mapped, for large off-policy buffers). Samplers: `SamplerWithoutReplacement`,
`PrioritizedSampler`, default random.

## On-policy vs off-policy wiring
- **On-policy (PPO/A2C):** `Collector` or `MultiCollector(sync=True)`, buffer sized to one
  collected batch, `SamplerWithoutReplacement`, re-filled each collector iteration.
- **Off-policy (DQN/SAC):** `MultiCollector(sync=False)`, big buffer persisted across
  iterations, random sampling, many gradient steps per collected batch.

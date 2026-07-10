# Collectors & replay buffers

Source: <https://docs.pytorch.org/rl/stable/reference/collectors.html>
Buffers: <https://docs.pytorch.org/rl/stable/reference/data.html>

## Collectors

Run a policy against an env and **yield batches of transitions as TensorDicts**. Iterate
over the collector in the training loop.

```python
from torchrl.collectors import SyncDataCollector

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=1000,   # transitions yielded per iteration
    total_frames=100_000,    # total collection budget (loop stops after)
    split_trajs=False,
    device=device,
)
for tensordict_data in collector:
    ...  # tensordict_data shape ~ [frames_per_batch] (or [n_envs, steps] if batched)
```

Types:
- **`SyncDataCollector`** — single-process, collect on the training worker.
- **`MultiSyncDataCollector`** — parallel workers, synchronized (on-policy: PPO/A2C — data
  matches the current policy).
- **`MultiaSyncDataCollector`** — parallel, async (off-policy: DQN/SAC — slight policy lag OK).

Keep inference policy weights fresh (multi-process): `collector.update_policy_weights_()`.

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
- **On-policy (PPO/A2C):** buffer sized to one collected batch, `SamplerWithoutReplacement`,
  re-filled each collector iteration.
- **Off-policy (DQN/SAC):** big buffer persisted across iterations, random sampling, many
  gradient steps per collected batch.

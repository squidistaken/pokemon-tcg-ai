# TorchRL — overview & docs map

Source: <https://docs.pytorch.org/rl/stable/index.html>

TorchRL is a PyTorch-first, open-source RL library. It offers both low- and high-level
abstractions and is built around **TensorDict** — a dict-like tensor container that flows
through every component so modules, envs, collectors, and losses share one data contract.

## Core building blocks

- **TensorDict** (`tensordict` package): the universal data carrier. Supports nested keys,
  batch dimensions, device moves, and `reshape`. Everything reads `in_keys` and writes
  `out_keys`.
- **`EnvBase`** (`torchrl.envs`): environment interface. `reset`/`step` take and return
  TensorDicts. Behavior described by **specs** (observation/action/reward/done).
- **`TensorDictModule`** (`tensordict.nn`): wraps an `nn.Module`, routing named tensordict
  fields in and out. Composed with `TensorDictSequential`.
- **Collectors** (`torchrl.collectors`): run a policy against an env and yield batches of
  transitions as TensorDicts (`SyncDataCollector`, multi-process variants).
- **Replay buffers** (`torchrl.data`): storage + sampler for transitions.
- **Objectives / losses** (`torchrl.objectives`): stateful modules (PPO, DQN, SAC, ...)
  that consume a TensorDict and emit `loss_*` components, using value estimators (GAE,
  TD(0), TD(λ)) internally.

## Documentation TOC (with paths)

### Getting Started
- Environments, TED and transforms
- TorchRL's modules
- Model optimization
- Data collection and storage
- Logging
- First training loop

### Tutorials
- Basics: PPO with TorchRL; Pendulum (writing envs & transforms); Intro to TorchRL
- Intermediate: Multi-Agent PPO; TorchRL environments; Pretrained models; Recurrent DQN;
  MuJoCo scripted manipulation; Collectors deep dive; Evaluator usage; Replay Buffers;
  Memory-efficient RL training; Exporting modules
- Advanced: Competitive Multi-Agent (DDPG); Multi-task policies; DDPG loss implementation;
  DQN trainer example

### API references (`reference/<name>.html`)
- `torchrl.collectors`, `torchrl.data`, Data layout (contiguous trajectories),
  `torchrl.envs`, LLM Interface, `torchrl.modules`, `torchrl.objectives`,
  Service Registry, `torchrl.trainers`, `torchrl._utils`, Configuration System,
  Profiling, Glossary

### Knowledge Base
- Contributing, Debugging RL, Installation guides (dm_control, MuJoCo, habitat, IsaacLab),
  Gym integration, PyTorch troubleshooting, Resources/versioning, Video rendering

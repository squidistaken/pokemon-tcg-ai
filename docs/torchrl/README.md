# TorchRL reference cache

Local, curated notes on [TorchRL](https://docs.pytorch.org/rl/stable/index.html) for
this project. **Consult these before doing any TorchRL/TensorDict work** (the routine is
defined in the repo-root [CLAUDE.md](../../CLAUDE.md)).

> Pinned version: `torchrl>=0.13.2`, `tensordict>=0.13`. These notes were distilled from
> the `stable` docs. If an API doesn't match the installed version, trust the installed
> version and the live docs, then update the relevant file here.

## Index

| File | Topic |
|------|-------|
| [00-overview.md](00-overview.md) | What TorchRL is, core building blocks, full docs TOC + URLs |
| [01-custom-env.md](01-custom-env.md) | Writing a custom `EnvBase` + specs (relevant: wrapping the PTCG engine) |
| [02-modules.md](02-modules.md) | `TensorDictModule`, actors, critics, discrete-action policies |
| [03-objectives-losses.md](03-objectives-losses.md) | Loss modules (PPO, DQN, A2C, SAC), advantage/value estimators |
| [04-collectors-buffers.md](04-collectors-buffers.md) | Data collectors and replay buffers |
| [05-ppo-recipe.md](05-ppo-recipe.md) | End-to-end PPO training loop recipe |

## Live docs quick links

- Home / getting started: <https://docs.pytorch.org/rl/stable/index.html>
- `torchrl.envs`: <https://docs.pytorch.org/rl/stable/reference/envs.html>
- `torchrl.modules`: <https://docs.pytorch.org/rl/stable/reference/modules.html>
- `torchrl.objectives`: <https://docs.pytorch.org/rl/stable/reference/objectives.html>
- `torchrl.collectors`: <https://docs.pytorch.org/rl/stable/reference/collectors.html>
- `torchrl.data` (buffers/specs): <https://docs.pytorch.org/rl/stable/reference/data.html>
- PPO tutorial: <https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html>
- Custom env tutorial (Pendulum): <https://docs.pytorch.org/rl/stable/tutorials/pendulum.html>
- Debugging RL: knowledge base section on the home page TOC

## How to extend this cache

When you fetch a docs page the cache doesn't cover, add or update the matching file
above with the distilled, code-first notes and cite the source URL at the top. Keep it
practical — recipes and snippets over prose.

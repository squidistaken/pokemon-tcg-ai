# Custom environments (`EnvBase`)

Source: <https://docs.pytorch.org/rl/stable/reference/envs.html>
See also the Pendulum tutorial: <https://docs.pytorch.org/rl/stable/tutorials/pendulum.html>

Relevant to this project: the PTCG engine (`ptcg_engine/`, C++ via a JSON API) will be
wrapped as a custom `EnvBase` so TorchRL collectors/losses can drive it.

## What to implement

Subclass `torchrl.envs.EnvBase` and implement:

1. **`_reset(self, tensordict)`** — initialize state, return a TensorDict with the initial
   observation (and `done`/`terminated` flags as needed).
2. **`_step(self, tensordict)`** — read the chosen action from the input TensorDict, advance
   the engine one step, return a TensorDict with `observation`, `reward`, `done`/`terminated`
   (and `truncated` if used).
3. **`_set_seed(self, seed)`** — seed any RNG the env uses.

All data flows as `tensordict.TensorDict`, so nested/batched observations are natural.

## Specs (the env's contract)

Define these in `__init__` (usually wrapped in a `Composite`):

- `observation_spec` — structure & bounds of observations
- `action_spec` — the action space
- `reward_spec` — reward shape/dtype
- `done_spec` — termination signals

Spec classes (from `torchrl.data`): `Composite`, `Bounded`, `Unbounded`, `Categorical`
(discrete). For the PTCG agent the action is **choosing an option index** among
`obs.select.option`, i.e. a discrete choice → use `Categorical`. Note the number of legal
options is *dynamic per step*; TorchRL specs are static, so a common pattern is to set
`action_spec` to the max option count and carry an **action mask** in the observation
(e.g. an `action_mask` key) that the policy/`QValueModule` uses to mask illegal actions.

## Sketch

```python
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, Categorical, Unbounded, Bounded

class PTCGEnv(EnvBase):
    def __init__(self, max_options: int, obs_dim: int, device="cpu"):
        super().__init__(device=device)
        self.observation_spec = Composite(
            observation=Unbounded(shape=(obs_dim,), dtype=torch.float32),
            action_mask=Categorical(2, shape=(max_options,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Categorical(max_options, shape=(), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Categorical(2, shape=(1,), dtype=torch.bool)

    def _reset(self, tensordict=None):
        obs, mask = self._engine_reset()
        return TensorDict({
            "observation": obs,
            "action_mask": mask,
            "done": torch.zeros(1, dtype=torch.bool),
        }, batch_size=[])

    def _step(self, tensordict):
        action = tensordict["action"]
        obs, mask, reward, done = self._engine_step(int(action))
        return TensorDict({
            "observation": obs,
            "action_mask": mask,
            "reward": torch.tensor([reward], dtype=torch.float32),
            "done": torch.tensor([done], dtype=torch.bool),
        }, batch_size=[])

    def _set_seed(self, seed):
        self._rng = torch.manual_seed(seed)
```

Validate a custom env with `torchrl.envs.utils.check_env_specs(env)` before training.

> Verify exact spec constructor names/signatures against the installed version — some spec
> classes have historical aliases (`DiscreteTensorSpec` == `Categorical`,
> `UnboundedContinuousTensorSpec` == `Unbounded`, etc.).

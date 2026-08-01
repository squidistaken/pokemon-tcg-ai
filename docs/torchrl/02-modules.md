# Modules: policies & critics

Source: <https://docs.pytorch.org/rl/stable/reference/modules.html>

TorchRL networks are `tensordict.nn.TensorDictModule`s: they route named tensordict fields
through an `nn.Module` via `in_keys` / `out_keys`. This is the core declarative pattern —
you say *which keys* feed the net and *which keys* it writes.

## Helpers
- **`MLP`** (`torchrl.modules`) — configurable fully-connected net.
- **`ConvNet`** — conv stack for image observations.
- Both can be `LazyLinear`-style (infer input dim on first forward).

## Discrete-action policies (relevant for PTCG option selection)

For value-based (DQN-style) discrete control:

```python
from torchrl.modules import MLP, QValueActor

value_mlp = MLP(out_features=env.action_spec.space.n, num_cells=[128, 128])
policy = QValueActor(value_mlp, in_keys=["observation"], spec=env.action_spec)
# policy writes "action", "action_value", "chosen_action_value"
```

`QValueModule` / `QValueActor` pick the argmax Q action. When actions must be masked
(illegal options), pass an `action_mask_key` (e.g. `"action_mask"`) so masked actions are
never selected. Wrap with `EGreedyModule` for exploration:

```python
from torchrl.modules import EGreedyModule
from tensordict.nn import TensorDictSequential
explore_policy = TensorDictSequential(policy, EGreedyModule(env.action_spec, eps_init=1.0, eps_end=0.05))
```

For policy-gradient (PPO/A2C) discrete control, output logits and use a categorical dist:

```python
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import MaskedCategorical  # honors action_mask

policy_net = TensorDictModule(
    MLP(out_features=n_actions, num_cells=[256, 256]),
    in_keys=["observation"], out_keys=["logits"],
)
policy = ProbabilisticActor(
    module=policy_net,
    in_keys={"logits": "logits", "mask": "action_mask"},  # MaskedCategorical kwargs
    out_keys=["action"],
    distribution_class=MaskedCategorical,
    return_log_prob=True,
    spec=env.action_spec,
)
```

(For continuous control the docs' canonical example emits `loc`/`scale` via
`NormalParamExtractor` and uses `TanhNormal` — see [05-ppo-recipe.md](05-ppo-recipe.md).)

## Critic / value network

```python
from torchrl.modules import ValueOperator
value_net = MLP(out_features=1, num_cells=[256, 256])
value_module = ValueOperator(value_net, in_keys=["observation"])  # writes "state_value"
```

## Key points
- **Spec-based construction**: many modules read `env.action_spec` to size the output layer.
- **Exploration wrappers**: `EGreedyModule`, `AdditiveGaussianModule`,
  `OrnsteinUhlenbeckProcessModule`.
- **Safe modules** project actions into the valid spec range.
- Compose multiple `TensorDictModule`s with `TensorDictSequential`.

> Verify `MaskedCategorical`'s exact in_keys wiring against the installed version; masking
> APIs have evolved. If unsure, check `torchrl.modules.distributions`.

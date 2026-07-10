# End-to-end PPO recipe

Source: <https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html>

Canonical TorchRL PPO loop. The tutorial targets continuous control (InvertedDoublePendulum)
— for PTCG's **discrete** option selection, swap the continuous policy head for logits +
`MaskedCategorical` (see [02-modules.md](02-modules.md)) and keep the rest.

## 1. Env + transforms
```python
from torchrl.envs import TransformedEnv, Compose, ObservationNorm, DoubleToFloat, StepCounter
env = TransformedEnv(base_env, Compose(
    ObservationNorm(in_keys=["observation"]),
    DoubleToFloat(),
    StepCounter(),
))
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
```

## 2. Policy (stochastic actor) + value network
```python
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator, NormalParamExtractor, TanhNormal

actor_net = nn.Sequential(
    nn.LazyLinear(256), nn.Tanh(),
    nn.LazyLinear(256), nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1]), NormalParamExtractor(),
)
policy_module = ProbabilisticActor(
    module=TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"]),
    spec=env.action_spec, in_keys=["loc", "scale"],
    distribution_class=TanhNormal, return_log_prob=True,
)
value_module = ValueOperator(
    module=nn.Sequential(nn.LazyLinear(256), nn.Tanh(), nn.LazyLinear(256), nn.Tanh(), nn.LazyLinear(1)),
    in_keys=["observation"],
)
```

## 3–6. Collector, buffer, advantage, loss, optim
```python
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
import torch

collector = SyncDataCollector(env, policy_module, frames_per_batch=1000,
                              total_frames=10_000, split_trajs=False, device=device)
rb = ReplayBuffer(storage=LazyTensorStorage(1000), sampler=SamplerWithoutReplacement())
advantage_module = GAE(gamma=0.99, lmbda=0.95, value_network=value_module, average_gae=True)
loss_module = ClipPPOLoss(actor_network=policy_module, critic_network=value_module,
                          clip_epsilon=0.2, entropy_bonus=True, entropy_coeff=1e-4)
optim = torch.optim.Adam(loss_module.parameters(), lr=3e-4)
```

## 7. Nested training loop
```python
for tensordict_data in collector:                 # outer: collect a batch
    for _ in range(num_epochs):                    # middle: reuse the batch
        advantage_module(tensordict_data)          # recompute GAE on fresh values
        rb.extend(tensordict_data.reshape(-1).cpu())
        for _ in range(frames_per_batch // sub_batch_size):   # inner: minibatches
            subdata = rb.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss = (loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
```

**Order that matters:** compute advantages → (re)fill buffer → sample minibatches → loss →
backward → clip grads → step. GAE is recomputed each epoch because value estimates change.

## PTCG adaptation checklist
- Custom `EnvBase` wrapping the engine ([01-custom-env.md](01-custom-env.md)).
- Discrete masked policy: logits head + `MaskedCategorical` + `action_mask` observation.
- ObservationNorm may not apply to structured card-game observations — consider a custom
  featurizer transform instead.
- Log to W&B in the outer loop (mean reward, losses, episode length from `StepCounter`).
- Consider DQN (off-policy) as a simpler first baseline — see [03-objectives-losses.md](03-objectives-losses.md).

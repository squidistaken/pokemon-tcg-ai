# Objectives / loss modules

Source: <https://docs.pytorch.org/rl/stable/reference/objectives.html>

Loss modules are **stateful** `nn.Module`s: instantiate with the actor/critic networks,
call on a TensorDict of rollout data, and get back a TensorDict of `loss_*` components.

## General pattern

```python
from torchrl.objectives import DDPGLoss

loss = DDPGLoss(actor_network=actor, value_network=value, gamma=0.99)
td = collector.rollout()
loss_vals = loss(td)
total_loss = sum(v for k, v in loss_vals.items() if k.startswith("loss_"))
total_loss.backward()
```

`loss.parameters()` exposes the trainable params → feed to your optimizer.

## Key losses
- **`PPOLoss` / `ClipPPOLoss`** — on-policy policy gradient; `ClipPPOLoss` adds ratio
  clipping. Returns `loss_objective`, `loss_critic`, `loss_entropy`. Needs advantages
  (compute with a `GAE` module before the update — see [05-ppo-recipe.md](05-ppo-recipe.md)).
- **`DQNLoss`** — value-based, discrete actions; TD targets. Pairs with a target-net updater
  (`SoftUpdate` / `HardUpdate` from `torchrl.objectives`). Good baseline for PTCG discrete
  option selection.
- **`A2CLoss`** — actor-critic with baseline-subtracted advantages.
- **`SACLoss`** — soft actor-critic (continuous; entropy-regularized).

## Value estimators / advantages

Estimators: **TD(0)**, **TD(λ)**, **GAE**. Set the estimator on a loss via
`loss.make_value_estimator(ValueEstimators.GAE, gamma=..., lmbda=...)`, or run a standalone
advantage module before the loss:

```python
from torchrl.objectives.value import GAE
advantage = GAE(gamma=0.99, lmbda=0.95, value_network=value_module, average_gae=True)
advantage(tensordict_data)   # writes "advantage" and "value_target" into the td
```

Loss modules can also compute advantages internally, removing manual preprocessing — but
for PPO the tutorial recomputes GAE each epoch on fresh value estimates (recommended).

## DQN skeleton (discrete)

```python
from torchrl.objectives import DQNLoss, SoftUpdate
loss = DQNLoss(policy, action_space=env.action_spec)
loss.make_value_estimator(gamma=0.99)
target_updater = SoftUpdate(loss, eps=0.995)
# per update: loss_vals = loss(sampled_td); backward; optim.step(); target_updater.step()
```

> Confirm constructor arg names per installed version (e.g. `actor_network` vs
> `actor_critic`, `critic_network`, `action_space`); they vary across losses/versions.

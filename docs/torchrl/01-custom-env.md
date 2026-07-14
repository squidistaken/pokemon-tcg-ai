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
`obs.select.option`, i.e. a discrete choice → use `Categorical`.

### Handling a dynamic number of legal options

The number of legal options is *dynamic per step*. TorchRL does support dynamic specs
(variable-size dims via `-1` on a spec's shape, combined with `return_contiguous=False`
on rollouts), but for a per-step-varying *discrete action count* the simpler, standard
pattern is still: fix `action_spec` to the max option count and carry an **action mask**
in the observation (e.g. an `action_mask` key). Prefer TorchRL's built-in consumers of
that mask over hand-rolling the logic:

- **`torchrl.envs.transforms.ActionMask`** — an env transform that reads the mask key and
  keeps `action_spec` in sync with the currently-legal actions.
- **`torchrl.modules.distributions.MaskedCategorical`** — used as the `distribution_class`
  of a `ProbabilisticActor` for policy-gradient/PPO actor-critic policies (this project's
  architecture; see [02-modules.md](02-modules.md)), wired via
  `in_keys={"logits": "logits", "mask": "action_mask"}`.

`QValueModule`/`QValueActor`'s own `action_mask_key` is the equivalent mechanism for
value-based (DQN-style) policies — not what this project's PPO actor-critic uses, but
relevant if a DQN baseline is ever tried (see [03-objectives-losses.md](03-objectives-losses.md)).
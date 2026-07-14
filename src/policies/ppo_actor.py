import math

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from tensordict.nn import TensorDictModule
from torchrl.data import Categorical, Composite, TensorSpec
from torchrl.modules import ActorValueOperator, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical

from src.models.actor_critic import ActorCritic
from src.models.backbone import Backbone
from src.models.heads import ValueHead

HIDDEN_KEY = "hidden"
OPTION_REPR_KEY = "option_repr"
LOGITS_KEY = "logits"
VALUE_KEY = "state_value"
ACTION_MASK_KEY = "action_mask"
ACTION_KEY = "action"
# The environment nests the encoder's output under "observation". These are
# the structured encoder's top-level groups (see StructuredObservationEncoder
# .spec()); MLPBackbone flattens each and concatenates them.
DEFAULT_IN_KEYS = [
    ("observation", "globals"),
    ("observation", "select_cats"),
    ("observation", "context_card_ids"),
    ("observation", "stadium_id"),
    ("observation", "options"),
    ("observation", "pokemon"),
    ("observation", "my"),
    ("observation", "opp"),
    ("observation", "select_deck"),
    ("observation", "looking"),
]


def _normalize_keys(raw_keys) -> list:
    """
    Coerce config-declared keys to strings or nested-key tuples.

    :param raw_keys: Keys as read from config (strings or lists for nesting).
    :return: List of ``str`` or ``tuple[str, ...]`` tensordict keys.
    """
    return [tuple(key) if isinstance(key, (list, ListConfig)) else key for key in raw_keys]


def _feature_width(spec: TensorSpec | Composite) -> int:
    """
    Flattened feature width of a (batchless) observation spec entry.

    Recurses into a :class:`~torchrl.data.Composite` (a group key, e.g. the
    structured encoder's ``options``/``pokemon`` tables) by summing the width
    of every leaf beneath it.

    :param spec: Spec of one observation key: a leaf tensor spec, or a
        composite group of them.
    :return: Product of its shape (its length for a vector entry), or the
        summed width of all leaves for a composite group.
    """
    if isinstance(spec, Composite):
        return sum(_feature_width(spec[leaf]) for leaf in spec.keys(True, True))
    return int(math.prod(spec.shape)) if len(spec.shape) > 0 else 1


def _input_dim(obs_spec: Composite, in_keys: list[str]) -> int:
    """
    Summed flattened width of the backbone's input keys.

    :param obs_spec: Observation composite spec of the environment.
    :param in_keys: Observation keys the backbone consumes; an entry may name
        a leaf field or a composite group (see :func:`_feature_width`).
    :return: Total input width fed to :class:`~src.models.backbone.MLPBackbone`.
    """
    return sum(_feature_width(obs_spec[key]) for key in in_keys)


def build_actor_critic(
        cfg: DictConfig,
        obs_spec: Composite,
        action_spec: Categorical,
) -> ActorCritic:
    """
    Build the standalone :class:`~src.models.actor_critic.ActorCritic`.

    The backbone and policy head are instantiated from their ``_target_``
    config groups (``cfg.model.backbone`` / ``cfg.model.head``) with the
    spec-derived dimensions injected, so which backbone and head are used is a
    pure config choice. The value head is built from ``cfg.model.value_head``.

    :param cfg: Hydra config carrying a ``model`` section (``embed_dim``,
        ``backbone``, ``head``, ``value_head``).
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec (a ``Categorical``); its size
        is ``max_options + 1``.
    :return: Assembled actor-critic with shared trunk and two heads.
    :raises ValueError: If the chosen head needs per-option tokens the chosen
        backbone does not emit.
    """
    embed_dim = int(cfg.model.embed_dim)
    n_actions = int(action_spec.space.n)

    in_keys = _normalize_keys(cfg.model.backbone.get("in_keys", DEFAULT_IN_KEYS))
    input_dim = _input_dim(obs_spec, in_keys)
    backbone: Backbone = instantiate(
        cfg.model.backbone,
        input_dim=input_dim,
        out_features=embed_dim,
    )
    # Hydra re-wraps list kwargs as ListConfig; set the normalized (nested-key
    # tuple) form directly so tensordict key lookups resolve.
    backbone.in_keys = in_keys
    policy_head = instantiate(cfg.model.head, in_features=embed_dim, n_actions=n_actions)
    value_head = ValueHead(in_features=embed_dim, num_cells=list(cfg.model.value_head.num_cells))

    if getattr(policy_head, "requires_option_repr", False) and not backbone.produces_option_repr:
        raise ValueError(
            f"Head {type(policy_head).__name__} needs per-option tokens, but backbone "
            f"{type(backbone).__name__} does not emit them. Pair the pointer head with a "
            f"Deep Sets / Set Transformer backbone, or use LinearPolicyHead."
        )
    return ActorCritic(backbone=backbone, policy_head=policy_head, value_head=value_head)


def build_ppo_operator(
        actor_critic: ActorCritic,
        action_spec: Categorical,
) -> ActorValueOperator:
    """
    Wrap an :class:`ActorCritic` in a shared-trunk torchrl operator.

    Uses :class:`~torchrl.modules.ActorValueOperator` so the backbone runs once
    per step and both the policy and value operators reuse its ``hidden``
    output. The policy is a masked categorical over the action slots; the same
    module instances are shared with ``actor_critic``, so training the operator
    and snapshotting ``actor_critic`` stay in sync.

    :param actor_critic: Actor-critic whose backbone/heads are wrapped.
    :param action_spec: Environment action spec, forwarded to the actor.
    :return: Operator exposing ``get_policy_operator`` / ``get_value_operator``.
    """
    backbone = actor_critic.backbone
    common_out = [HIDDEN_KEY, OPTION_REPR_KEY] if backbone.produces_option_repr else [HIDDEN_KEY]
    head_in = (
        [HIDDEN_KEY, OPTION_REPR_KEY]
        if getattr(actor_critic.policy_head, "requires_option_repr", False)
        else [HIDDEN_KEY]
    )

    common = TensorDictModule(backbone, in_keys=backbone.in_keys, out_keys=common_out)
    policy = ProbabilisticActor(
        TensorDictModule(actor_critic.policy_head, in_keys=head_in, out_keys=[LOGITS_KEY]),
        in_keys={"logits": LOGITS_KEY, "mask": ACTION_MASK_KEY},
        out_keys=[ACTION_KEY],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
        spec=action_spec,
    )
    value = ValueOperator(actor_critic.value_head, in_keys=[HIDDEN_KEY], out_keys=[VALUE_KEY])
    return ActorValueOperator(common, policy, value)


def build_ppo_actor_critic(
        cfg: DictConfig,
        obs_spec: Composite,
        action_spec: Categorical,
) -> ActorValueOperator:
    """
    Build the PPO shared-trunk actor-critic operator from config and specs.

    Drop-in compatible with the :class:`~src.policies.random_masked_policy.
    RandomMaskedPolicy` collection contract: it reads ``action_mask`` and
    writes an int64 ``action``, so the existing collector pipeline carries over
    unchanged.

    :param cfg: Hydra config with a ``model`` section.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :return: The assembled :class:`~torchrl.modules.ActorValueOperator`.
    """
    return build_ppo_operator(build_actor_critic(cfg, obs_spec, action_spec), action_spec)

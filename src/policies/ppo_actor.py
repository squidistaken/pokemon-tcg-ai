import math

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, ListConfig
from tensordict.nn import TensorDictModule
from torchrl.data import Categorical, Composite, TensorSpec
from torchrl.modules import ActorValueOperator, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical

from src.models.actor_critic import ActorCritic
from src.models.backbone import Backbone
from src.models.heads import ValueHead
from src.models.structured_obs_adapter import StructuredObsAdapter

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
    :return: Total input width fed to :class:`~src.models.mlp.MLPBackbone`.
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

    When any in-key names a composite group (the structured encoder's
    ``options``/``pokemon``/zone tables), a
    :class:`~src.models.structured_obs_adapter.StructuredObsAdapter` is built
    from ``cfg.model.adapter`` (defaults when absent) and handed to the
    backbone, which then consumes the adapter's embedded/normalized feature
    vector instead of naively flattened raw fields.

    :param cfg: Hydra config carrying a ``model`` section (``embed_dim``,
        ``backbone``, ``head``, ``value_head``, optionally ``adapter``).
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
    # Whether the adapter emits per-option tokens is derived from the chosen
    # head rather than configured separately: the two must agree, and a config
    # where they disagree is only ever a mistake.
    head_class = get_class(cfg.model.head["_target_"])
    needs_option_repr = bool(getattr(head_class, "requires_option_repr", False))
    # A trunk configured with option_tokens builds the table itself (projecting
    # it to the trunk's width and adding a stop-slot segment embedding), so the
    # adapter must not also emit one -- the backbone rejects both at once.
    backbone_builds_tokens = bool(cfg.model.backbone.get("option_tokens", False))

    backbone_kwargs: dict = {"out_features": embed_dim}
    adapter: StructuredObsAdapter | None = None
    if any(isinstance(obs_spec[key], Composite) for key in in_keys):
        adapter_kwargs = dict(cfg.model.get("adapter", None) or {})
        adapter_kwargs["emit_option_tokens"] = needs_option_repr and not backbone_builds_tokens
        adapter = StructuredObsAdapter(obs_spec=obs_spec, in_keys=in_keys, **adapter_kwargs)
        backbone_kwargs["adapter"] = adapter
        backbone_kwargs["input_dim"] = adapter.out_features
    else:
        backbone_kwargs["input_dim"] = _input_dim(obs_spec, in_keys)
    backbone: Backbone = instantiate(cfg.model.backbone, **backbone_kwargs)
    # Hydra re-wraps list kwargs as ListConfig; set the normalized (nested-key
    # tuple) form directly so tensordict key lookups resolve.
    backbone.in_keys = in_keys

    head_kwargs: dict = {"in_features": embed_dim, "n_actions": n_actions}
    if needs_option_repr:
        if adapter is None:
            raise ValueError(
                f"Head {head_class.__name__} needs per-option tokens, which require the "
                f"structured observation groups; got backbone in_keys {in_keys}."
            )
        if not backbone.produces_option_repr:
            raise ValueError(
                f"Head {head_class.__name__} needs per-option tokens, but backbone "
                f"{type(backbone).__name__} does not emit them."
            )
        # From the backbone, not the adapter: a trunk may project the adapter's
        # per-entity encodings to its own width or pass them through untouched,
        # and only it knows which. Asking the adapter would size the head to the
        # wrong width on the projecting path.
        head_kwargs["option_dim"] = backbone.option_repr_dim
    policy_head = instantiate(cfg.model.head, **head_kwargs)
    value_head = ValueHead(
        in_features=embed_dim,
        num_cells=list(cfg.model.value_head.num_cells),
        activation=cfg.model.value_head.get("activation", "tanh"),
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

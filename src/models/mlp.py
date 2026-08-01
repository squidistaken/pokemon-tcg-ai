import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP
from src.models.backbone import Backbone, activation_class

class MLPBackbone(Backbone):
    """
    Flat multi-layer-perceptron trunk — the literature's proven PPO baseline.

    Turns every input into a per-sample feature vector, concatenates them
    along the last dimension, and runs the result through a fully-connected
    stack. This is the honest control every richer (per-option-token)
    backbone must beat. It emits no per-option tokens
    (:attr:`produces_option_repr` is False) and pairs with
    :class:`~src.models.heads.LinearPolicyHead`.

    With a :class:`~src.models.structured_obs_adapter.StructuredObsAdapter`
    attached (the standard pairing, wired by
    :func:`~src.policies.ppo_actor.build_actor_critic`), the adapter performs
    the model-side half of the observation contract — embedding card/attack
    IDs, normalizing scalars, pooling unordered zones — before the MLP.
    Without one, each input is naively flattened and cast to float: plain
    tensors past their last dimension, nested
    :class:`~tensordict.TensorDictBase` groups leaf by leaf past their own
    ``batch_size``.
    """

    produces_option_repr = False

    def __init__(
            self,
            input_dim: int,
            out_features: int,
            num_cells: list[int],
            activation: str = "tanh",
            in_keys: list[str] | None = None,
            adapter: nn.Module | None = None,
    ) -> None:
        """
        :param input_dim: Width of the concatenated feature vector fed to the
            MLP; with an adapter this must equal its ``out_features``.
        :param out_features: Width of the produced ``state_repr``.
        :param num_cells: Hidden layer widths of the MLP.
        :param activation: Hidden activation name (see :func:`activation_class`).
        :param in_keys: Observation keys to consume; defaults to
            ``["observation"]`` (a single pre-built feature vector).
        :param adapter: Optional :class:`~src.models.structured_obs_adapter.
            StructuredObsAdapter` handling the structured groups; None falls
            back to naive flattening.
        :raises ValueError: If the adapter's output width disagrees with
            ``input_dim``.
        """
        super().__init__(in_keys=in_keys or ["observation"], out_features=out_features)
        if adapter is not None and adapter.out_features != input_dim:
            raise ValueError(
                f"Adapter produces {adapter.out_features} features but the MLP expects "
                f"input_dim={input_dim}."
            )
        self.input_dim = input_dim
        self.adapter = adapter
        self.mlp = MLP(
            in_features=input_dim,
            out_features=out_features,
            num_cells=list(num_cells),
            activation_class=activation_class(activation),
        )

    def forward(self, *inputs: torch.Tensor | TensorDictBase) -> torch.Tensor:
        """
        Vectorize, concatenate and encode the inputs into ``state_repr``.

        :param inputs: One entry per :attr:`in_keys`: either a per-sample
            feature vector shaped ``(*batch, features)``, or a nested
            tensordict group handed to the adapter (or naively flattened
            leaf by leaf when no adapter is attached).
        :return: ``state_repr`` of shape ``(*batch, out_features)``.
        """
        if len(inputs) != len(self.in_keys):
            raise ValueError(
                f"MLPBackbone expected {len(self.in_keys)} inputs for keys "
                f"{self.in_keys}, got {len(inputs)}."
            )
        if self.adapter is not None:
            return self.mlp(self.adapter(*inputs))
        flat_parts: list[torch.Tensor] = []
        for value in inputs:
            if isinstance(value, TensorDictBase):
                batch_ndim = len(value.batch_size)
                for leaf_key in value.keys(include_nested=True, leaves_only=True):
                    leaf = value.get(leaf_key)
                    flat_parts.append(leaf.reshape(*leaf.shape[:batch_ndim], -1).to(torch.float32))
            else:
                flat_parts.append(value.reshape(*value.shape[:-1], -1).to(torch.float32))
        features = flat_parts[0] if len(flat_parts) == 1 else torch.cat(flat_parts, dim=-1)
        return self.mlp(features)
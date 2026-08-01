import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase


class RandomMaskedPolicy(TensorDictModuleBase):
    """
    Uniform random policy over the legal actions given an action mask.

    Stand-in for the eventual PPO actor: it consumes the same "action_mask"
    key the future MaskedCategorical actor will use, so the collector
    pipeline built around it carries over unchanged.
    """

    def __init__(self, mask_key: str = "action_mask", action_key: str = "action") -> None:
        """
        :param mask_key: Tensordict key holding the boolean action mask.
        :param action_key: Tensordict key to write the sampled action to.
        """
        super().__init__()
        self.in_keys = [mask_key]
        self.out_keys = [action_key]
        self._mask_key = mask_key
        self._action_key = action_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Sample one action uniformly among the mask's True entries.

        :param tensordict: Input tensordict containing the action mask.
        :return: The same tensordict with the sampled action written in.
        """
        mask = tensordict.get(self._mask_key)
        flat_mask = mask.reshape(-1, mask.shape[-1])
        actions = torch.multinomial(flat_mask.to(torch.float32), num_samples=1).squeeze(-1)
        tensordict.set(self._action_key, actions.reshape(mask.shape[:-1]))
        return tensordict

import pytest
import torch
from tensordict import TensorDict

from src.training.loss._helpers import _sum_loss_keys


def test_sums_only_loss_prefixed_keys() -> None:
    """Only ``loss_``-prefixed entries are summed; diagnostics are ignored."""
    loss_vals = TensorDict(
        {
            "loss_objective": torch.tensor(1.0),
            "loss_critic": torch.tensor(2.0),
            "loss_entropy": torch.tensor(0.5),
            "ESS": torch.tensor(100.0),
            "kl_approx": torch.tensor(0.01),
            "entropy": torch.tensor(3.0),
        },
        batch_size=[],
    )
    total = _sum_loss_keys(loss_vals)
    assert torch.isclose(total, torch.tensor(3.5))


def test_raises_when_no_loss_keys() -> None:
    """A tensordict with no ``loss_`` entry is a misconfiguration -> KeyError."""
    loss_vals = TensorDict({"ESS": torch.tensor(1.0)}, batch_size=[])
    with pytest.raises(KeyError):
        _sum_loss_keys(loss_vals)

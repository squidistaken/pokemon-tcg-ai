import torch
from torchrl.modules.distributions import MaskedCategorical

from src.models.masked_rpo_categorical import MaskedRPOCategorical


def test_disabled_matches_masked_categorical() -> None:
    """With ``rpo_enabled`` False the distribution is identical to the plain one."""
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[:, 0] = False

    MaskedRPOCategorical.rpo_enabled = False
    rpo = MaskedRPOCategorical(logits=logits.clone(), mask=mask)
    base = MaskedCategorical(logits=logits.clone(), mask=mask)
    assert torch.allclose(rpo.logits, base.logits)


def test_enabled_perturbs_logits() -> None:
    """With ``rpo_enabled`` True the legal-action logits are perturbed."""
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)

    MaskedRPOCategorical.rpo_alpha = 0.5
    try:
        MaskedRPOCategorical.rpo_enabled = True
        rpo = MaskedRPOCategorical(logits=logits.clone(), mask=mask)
        base = MaskedCategorical(logits=logits.clone(), mask=mask)
        assert not torch.allclose(rpo.logits, base.logits)
    finally:
        MaskedRPOCategorical.rpo_enabled = False


def test_masked_actions_stay_zero_probability_even_when_enabled() -> None:
    """Perturbation runs before masking, so illegal actions keep ~0 probability."""
    torch.manual_seed(0)
    logits = torch.randn(8, 6)
    mask = torch.ones(8, 6, dtype=torch.bool)
    mask[:, 2] = False
    mask[:, 4] = False

    MaskedRPOCategorical.rpo_alpha = 5.0  # large noise: still must not leak
    try:
        MaskedRPOCategorical.rpo_enabled = True
        dist = MaskedRPOCategorical(logits=logits, mask=mask)
        probs = dist.probs
        assert torch.all(probs[:, 2] == 0.0)
        assert torch.all(probs[:, 4] == 0.0)
        assert torch.allclose(probs.sum(-1), torch.ones(8), atol=1e-5)
    finally:
        MaskedRPOCategorical.rpo_enabled = False
        MaskedRPOCategorical.rpo_alpha = 0.5

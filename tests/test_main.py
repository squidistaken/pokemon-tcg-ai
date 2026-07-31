import dataclasses

import pytest
from omegaconf import OmegaConf

import main
from src.env.battle_handle import BattleHandle
from src.env.deck import load_deck
from src.policies.greedy_policy_opponent import save_actor_critic
from src.policies.inference import build_inference_specs
from src.policies.ppo_actor import build_actor_critic
from tests.conftest import DECK_PATH, MAX_OPTIONS

DECK = load_deck(DECK_PATH)


@pytest.fixture(autouse=True)
def _reset_cached_agent():
    """
    Clear ``main``'s lazily-built agent singleton around every test.

    Without this, whichever test runs first would permanently cache an
    agent built from its own tmp_path checkpoint, and every later test
    would silently reuse it instead of loading its own.
    """
    main._agent = None  # noqa: SLF001
    yield
    main._agent = None  # noqa: SLF001


@pytest.fixture
def real_observation() -> dict:
    """
    A real first-selection observation from a fresh engine battle, in the
    same raw-dict shape Kaggle's harness passes to ``agent()``.

    :return: Observation dict with a non-``None`` ``select``.
    """
    handle = BattleHandle()
    try:
        observation = handle.start(DECK, DECK)
    finally:
        handle.finish()
    assert observation.select is not None
    return dataclasses.asdict(observation)


def _write_checkpoint(output_dir, structured_model_cfg) -> None:
    """
    Write a checkpoint + model config pair matching ``structured_model_cfg``
    into ``output_dir``, mirroring ``scripts/export_submission_checkpoint.py``.
    """
    obs_spec, _, action_spec = build_inference_specs(MAX_OPTIONS)
    actor_critic = build_actor_critic(structured_model_cfg, obs_spec, action_spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_actor_critic(actor_critic, output_dir / "model.pt")
    model_config = OmegaConf.create(
        {
            "model": OmegaConf.to_container(structured_model_cfg.model, resolve=True),
            "max_options": MAX_OPTIONS,
            "encoder": "structured",
        }
    )
    OmegaConf.save(model_config, output_dir / "model_config.yaml")


def test_agent_selects_legal_options_from_checkpoint(tmp_path, monkeypatch, structured_model_cfg, real_observation) -> None:
    """
    With a checkpoint on disk, ``agent()`` returns a legal selection for a
    real engine observation, going through the actual checkpoint-loading path.
    """
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path / "checkpoint", structured_model_cfg)

    picks = main.agent(real_observation)

    option_count = len(real_observation["select"]["option"])
    min_count = real_observation["select"]["minCount"]
    max_count = real_observation["select"]["maxCount"]
    assert min_count <= len(picks) <= max_count
    assert len(set(picks)) == len(picks)
    assert all(0 <= pick < option_count for pick in picks)


def test_agent_reuses_cached_instance(tmp_path, monkeypatch, structured_model_cfg, real_observation) -> None:
    """
    The agent is built once and cached, not rebuilt on every call.
    """
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path / "checkpoint", structured_model_cfg)

    main.agent(real_observation)
    agent = main._agent  # noqa: SLF001
    assert agent is not None
    main.agent(real_observation)
    assert main._agent is agent  # noqa: SLF001


def test_agent_raises_when_checkpoint_missing(tmp_path, monkeypatch, real_observation) -> None:
    """
    A missing checkpoint/config is a packaging mistake, not a state to
    silently degrade from: ``agent()`` must raise rather than fall back.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        main.agent(real_observation)

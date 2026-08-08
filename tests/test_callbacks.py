import inspect
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any, override

import pytest
import torch
from tensordict import TensorDict

from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training.callbacks import CallbackList, TrainingCallback, WeightsAndBiases
from src.training.env_factory import make_env_factories
from src.training.trainer import Trainer
from tests.conftest import structured_env_cfg

CORE_METRICS = {"frames", "episodes", "win_rate", "draw_rate", "fps"}


class RecordingCallback(TrainingCallback):
    """
    Callback that records every hook it receives, for assertions.

    Mappings are copied on arrival.
    """

    def __init__(self, name: str = "rec", sink: list[tuple] | None = None) -> None:
        """
        :param name: Label prefixed to this callback's events, to tell members
            of a :class:`CallbackList` apart.
        :param sink: Shared event list; None gives this callback its own. Pass a
            shared list to assert on ordering across several callbacks.
        """
        self.name = name
        self.events: list[tuple] = sink if sink is not None else []

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Record the run start.

        :param run_config: Run metadata handed over by the trainer.
        """
        self.events.append((self.name, "start", dict(run_config)))

    def on_rollout_start(self, step: int) -> None:
        """
        Record a rollout start.

        :param step: Total frames collected before this rollout.
        """
        self.events.append((self.name, "rollout_start", step))

    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Record a rollout.

        :param step: Total frames collected so far.
        :param metrics: Metrics at ``step``.
        """
        self.events.append((self.name, "rollout_end", step, dict(metrics)))

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Record an evaluation round.

        :param step: Frames collected at evaluation time.
        :param metrics: Evaluation metrics.
        """
        self.events.append((self.name, "eval", step, dict(metrics)))

    def on_train_error(self, error: BaseException) -> None:
        """
        Record a run failure.

        :param error: The exception that ended the run.
        """
        self.events.append((self.name, "error", str(error)))

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Record the run end.

        :param summary: Aggregate run statistics.
        """
        self.events.append((self.name, "end", dict(summary)))


class ExplodingCallback(TrainingCallback):
    """
    Callback whose every hook raises, standing in for a broken backend.
    """

    @override
    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        :param run_config: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("backend down: on_train_start")

    @override
    def on_rollout_start(self, step: int) -> None:
        """
        :param step: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("backend down: on_rollout_start")

    @override
    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        :param step: Ignored.
        :param metrics: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("backend down: on_rollout_end")

    @override
    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        :param step: Ignored.
        :param metrics: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("backend down: on_eval_end")

    @override
    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        :param summary: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("backend down: on_train_end")


class FailingTrainer(Trainer):
    """
    Trainer whose update raises, to exercise the failure path of ``train``.
    """

    @override
    def _update(self, data: TensorDict) -> dict[str, float] | None:
        """
        :param data: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("update exploded")


class FakeRun:
    """
    Stand-in for a ``wandb`` run, capturing what the callback sends it.
    """

    def __init__(self) -> None:
        self.id = "fake-run-id"
        self.name = "fake-run-name"
        self.url = "https://wandb.test/fake-run-id"
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, float], int | None]] = []
        self.log_lines: list[str] = []
        self.artifacts: list[tuple[str, str | None, str | None, list[str] | None]] = []
        self.finished = False

    def log(self, data: dict[str, float], step: int | None = None) -> None:
        """
        Capture one logged point.

        :param data: Metric mapping sent by the callback.
        :param step: X-axis value sent by the callback.
        """
        self.logged.append((data, step))

    def finish(self) -> None:
        """
        Mark the run as closed.
        """
        self.finished = True

    def write_logs(self, text: str) -> None:
        """Capture text written directly to W&B's Logs tab."""
        self.log_lines.append(text)

    def log_artifact(
        self,
        path: str,
        name: str | None = None,
        type: str | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Capture one model artifact logged by the callback."""
        self.artifacts.append((path, name, type, aliases))


class BrokenWandbModule(ModuleType):
    """
    Stand-in ``wandb`` module whose ``init`` always raises, for a W&B outage.
    """

    def __init__(self) -> None:
        super().__init__("wandb")

    @staticmethod
    def Settings(**kwargs: Any) -> Any:
        """Return a minimal settings object accepted by the callback."""
        return type("Settings", (), kwargs)()

    @staticmethod
    def init(**_kwargs: Any) -> Any:
        """
        :param _kwargs: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("wandb is down")


class FakeWandbModule(ModuleType):
    """
    Stand-in ``wandb`` module, injected into ``sys.modules``.

    Intercepting it there works because the real import is lazy, inside
    ``on_train_start`` — so no network, credentials or run directory.
    """

    def __init__(self, run: FakeRun) -> None:
        """
        :param run: Run object to hand back from :meth:`init`.
        """
        super().__init__("wandb")
        self.run = run
        self.init_kwargs: dict[str, Any] | None = None

    @staticmethod
    def Settings(**kwargs: Any) -> Any:
        """Return a minimal settings object whose fields tests can inspect."""
        return type("Settings", (), kwargs)()

    def init(self, **kwargs: Any) -> FakeRun:
        """
        Record the init arguments and return the fake run.

        :param kwargs: Whatever the callback passed to ``wandb.init``.
        :return: The fake run.
        """
        self.init_kwargs = kwargs
        return self.run


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeWandbModule, FakeRun]:
    """
    Install a fake ``wandb`` module for the duration of a test.

    :param monkeypatch: pytest monkeypatch fixture.
    :return: The fake module and the run it hands out.
    """
    run = FakeRun()
    module = FakeWandbModule(run)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module, run


def make_trainer(callbacks: list[TrainingCallback], **kwargs: Any) -> Trainer:
    """
    Build a small collection-only trainer over a SerialEnv.

    :param callbacks: Callbacks to attach.
    :param kwargs: Overrides forwarded to :class:`Trainer`.
    :return: A configured trainer.
    """
    return Trainer(
        env_factories=make_env_factories(structured_env_cfg()),
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=128,
        use_parallel_env=False,
        callbacks=callbacks,
        **kwargs,
    )


def test_base_callback_is_abstract() -> None:
    """
    The interface is abstract: every hook must be implemented, so a callback
    that omits one cannot be instantiated.
    """
    assert inspect.isabstract(TrainingCallback)


def test_callback_list_fans_out_to_every_member_in_order() -> None:
    """
    Each hook reaches all members, in the order they were registered.
    """
    events: list[tuple] = []
    callbacks = CallbackList(
        [RecordingCallback("first", events), RecordingCallback("second", events)]
    )

    callbacks.on_train_start({"seed": 1})
    callbacks.on_rollout_start(0)
    callbacks.on_rollout_end(10, {"win_rate": 0.5})
    callbacks.on_eval_end(10, {"win_rate": 0.7})
    callbacks.on_train_end({"frames": 10})

    assert [(name, hook) for name, hook, *_ in events] == [
        ("first", "start"),
        ("second", "start"),
        ("first", "rollout_start"),
        ("second", "rollout_start"),
        ("first", "rollout_end"),
        ("second", "rollout_end"),
        ("first", "eval"),
        ("second", "eval"),
        ("first", "end"),
        ("second", "end"),
    ]
    assert len(callbacks) == 2


def test_callback_list_propagates_a_failing_member() -> None:
    """
    A selected backend is required: its failure propagates immediately and
    callbacks after it are not invoked.
    """
    healthy = RecordingCallback("healthy")
    callbacks = CallbackList([ExplodingCallback(), healthy])

    with pytest.raises(RuntimeError, match="backend down: on_train_start"):
        callbacks.on_train_start({"seed": 2})

    assert healthy.events == []


def test_trainer_notifies_callbacks_across_the_run() -> None:
    """
    Start, one batch per collector iteration, then end, x-axed by frames.
    """
    recorder = RecordingCallback()
    trainer = make_trainer(
        [recorder], run_config={"seed": 0, "agent": {"name": "dummy"}}
    )

    stats = trainer.train()

    hooks = [hook for _, hook, *_ in recorder.events]
    assert hooks[0] == "start"
    assert hooks[-1] == "end"
    assert hooks.count("rollout_end") == 2

    assert recorder.events[0][2] == {"seed": 0, "agent": {"name": "dummy"}}

    # Each rollout is bracketed: it starts at the frames collected so far and
    # ends once its own frames are added.
    rollout_starts = [event for event in recorder.events if event[1] == "rollout_start"]
    assert [step for _, _, step in rollout_starts] == [0, 64]

    rollouts = [event for event in recorder.events if event[1] == "rollout_end"]
    assert [step for _, _, step, _ in rollouts] == [64, 128]
    for _, _, step, metrics in rollouts:
        assert set(metrics) >= CORE_METRICS
        assert metrics["frames"] == step
        assert 0.0 <= metrics["win_rate"] <= 1.0

    # The end summary is exactly what train() reports, so a backend's final
    # numbers can never disagree with the caller's.
    assert recorder.events[-1][2] == stats


def test_trainer_notifies_train_end_when_the_run_fails() -> None:
    """
    Backends still get teardown when collection raises, so a crashed run is
    closed out rather than left hanging.
    """
    recorder = RecordingCallback()
    trainer = FailingTrainer(
        env_factories=make_env_factories(structured_env_cfg()),
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=128,
        use_parallel_env=False,
        callbacks=[recorder],
    )

    with pytest.raises(RuntimeError, match="update exploded"):
        trainer.train()

    hooks = [hook for _, hook, *_ in recorder.events]
    assert hooks[0] == "start"
    assert hooks[-1] == "end"
    # The failure is reported before teardown, so a backend can mark the run
    # crashed rather than closing it as cleanly as a finished one.
    assert hooks[-2] == "error"
    assert "update exploded" in recorder.events[-2][2]


def test_trainer_without_callbacks_still_trains() -> None:
    """
    Callbacks are optional; the default path attaches none.
    """
    stats = make_trainer([]).train()
    assert stats["frames"] == 128


def test_wandb_callback_records_config_and_namespaces_metrics(
    fake_wandb: tuple[FakeWandbModule, FakeRun],
) -> None:
    """
    Construction args reach ``wandb.init``, metrics are namespaced, and the run
    closes with the aggregates in its summary.
    """
    module, run = fake_wandb
    callback = WeightsAndBiases(
        project="pokemon-tcg-ai",
        entity="team",
        tags=["baseline"],
        mode="offline",
        dir="/scratch/runs/one",
    )

    callback.on_train_start({"seed": 7})
    assert module.init_kwargs is not None
    assert module.init_kwargs["project"] == "pokemon-tcg-ai"
    assert module.init_kwargs["entity"] == "team"
    assert module.init_kwargs["tags"] == ["baseline"]
    assert module.init_kwargs["mode"] == "offline"
    assert module.init_kwargs["force"] is False
    assert module.init_kwargs["dir"] == "/scratch/runs/one"
    assert module.init_kwargs["config"] == {"seed": 7}
    assert module.init_kwargs["settings"].console == "redirect"
    assert module.init_kwargs["settings"].capture_loggers == {"root": "INFO"}

    callback.on_rollout_end(64, {"win_rate": 0.5, "loss_objective": -0.2})
    assert run.logged[-1] == ({"train/win_rate": 0.5, "train/loss_objective": -0.2}, 64)

    callback.on_eval_end(64, {"win_rate": 0.9})
    assert run.logged[-1] == ({"eval/win_rate": 0.9}, 64)

    callback.on_train_end({"frames": 128, "win_rate": 0.5})
    assert run.summary == {"summary/frames": 128, "summary/win_rate": 0.5}
    assert run.finished


def test_wandb_callback_requires_online_logging(
    fake_wandb: tuple[FakeWandbModule, FakeRun],
) -> None:
    """
    Online mode tells W&B that authentication is required instead of allowing
    its automatic offline fallback.
    """
    module, _ = fake_wandb

    WeightsAndBiases(project="pokemon-tcg-ai", mode="online").on_train_start({})

    assert module.init_kwargs is not None
    assert module.init_kwargs["mode"] == "online"
    assert module.init_kwargs["force"] is True


def test_wandb_callback_logs_checkpoint_as_model_artifact(
    fake_wandb: tuple[FakeWandbModule, FakeRun],
    tmp_path,
) -> None:
    """Checkpoint keys become W&B aliases on a run-specific model artifact."""
    _, run = fake_wandb
    checkpoint = tmp_path / "snapshot.pt"
    checkpoint.write_bytes(b"weights")
    callback = WeightsAndBiases(project="pokemon-tcg-ai", mode="offline")
    callback.on_train_start({})

    callback.log_checkpoint(checkpoint, "a" * 64, 123)

    assert run.artifacts == [
        (
            str(checkpoint),
            "checkpoint-fake-run-id",
            "model",
            ["latest", "sha-aaaaaaaaaaaa", "frames-123"],
        )
    ]


def test_wandb_callback_honours_explicit_offline_mode(
    fake_wandb: tuple[FakeWandbModule, FakeRun],
) -> None:
    """
    Offline recording remains available when the user deliberately requests it.
    """
    module, _ = fake_wandb

    WeightsAndBiases(project="pokemon-tcg-ai", mode="offline").on_train_start({})

    assert module.init_kwargs is not None
    assert module.init_kwargs["mode"] == "offline"
    assert module.init_kwargs["force"] is False


def test_wandb_callback_hooks_are_inert_without_a_run() -> None:
    """
    Metric hooks no-op without a run so cleanup is safe after startup fails.
    """
    callback = WeightsAndBiases(project="pokemon-tcg-ai")
    callback.on_rollout_start(1)
    callback.on_rollout_end(1, {"win_rate": 1.0})
    callback.on_eval_end(1, {"win_rate": 1.0})
    callback.on_train_end({"frames": 1})


def test_wandb_callback_rejects_an_invalid_mode() -> None:
    """
    A mode typo fails at construction, before any environment is built.
    """
    with pytest.raises(ValueError, match="Invalid W&B mode"):
        WeightsAndBiases(project="pokemon-tcg-ai", mode="onlien")


def test_wandb_init_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A requested W&B backend is required, so an initialization outage fails.
    """
    monkeypatch.setitem(sys.modules, "wandb", BrokenWandbModule())

    callbacks = CallbackList([WeightsAndBiases(project="pokemon-tcg-ai")])
    with pytest.raises(RuntimeError, match="wandb is down"):
        callbacks.on_train_start({"seed": 0})


def test_wandb_logging_failure_propagates(
    fake_wandb: tuple[FakeWandbModule, FakeRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A W&B failure after startup also fails the selected callback contract.
    """
    _, run = fake_wandb
    callback = WeightsAndBiases(project="pokemon-tcg-ai", mode="offline")
    callback.on_train_start({})

    def fail_log(_data: dict[str, float], step: int | None = None) -> None:
        raise RuntimeError(f"wandb logging failed at step {step}")

    monkeypatch.setattr(run, "log", fail_log)
    with pytest.raises(RuntimeError, match="wandb logging failed at step 64"):
        callback.on_rollout_end(64, {"win_rate": 0.5})


def test_wandb_start_failure_creates_no_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A backend that fails at startup costs no environment workers at all.

    Collectors are built inside the run loop, after ``on_train_start``, so a
    strict backend rejecting the run short-circuits before anything is forked
    -- there is no collector left to leak rather than one that gets cleaned up.
    """
    created: list[Any] = []

    class FakeCollector:
        def __init__(self, **_kwargs: Any) -> None:
            self.shutdown_called = False
            created.append(self)

        def __iter__(self) -> Any:
            return iter(())

        def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr("src.training.trainer.Collector", FakeCollector)
    monkeypatch.setitem(sys.modules, "wandb", BrokenWandbModule())

    trainer = make_trainer([WeightsAndBiases(project="pokemon-tcg-ai")])
    with pytest.raises(RuntimeError, match="wandb is down"):
        trainer.train()

    assert created == []


def test_wandb_marks_a_failed_run_crashed(
    fake_wandb, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A run that dies mid-collection is finished with a non-zero exit code.

    Otherwise W&B shows it as ``finished``, making a crash indistinguishable
    from a short successful run.
    """
    _, run = fake_wandb
    exit_codes: list[int | None] = []
    monkeypatch.setattr(
        run, "finish", lambda exit_code=None: exit_codes.append(exit_code)
    )

    callback = WeightsAndBiases(project="pokemon-tcg-ai", mode="offline")
    callback.on_train_start({})
    callback.on_train_error(RuntimeError("Cannot proceed, worker 20 dead."))
    callback.on_train_end({"frames": 475136.0})

    assert exit_codes == [1]
    assert "worker 20 dead" in str(run.summary["summary/error"])
    assert run.log_lines == ["RuntimeError: Cannot proceed, worker 20 dead."]


def test_wandb_training_failure_publishes_traceback(
    fake_wandb: tuple[FakeWandbModule, FakeRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A simulated OOM during training is sent to W&B before the run is closed.

    The fake run intercepts ``write_logs`` at the SDK boundary, so this covers
    the complete trainer/callback failure path without contacting W&B or
    allocating enough memory to cause a real OOM.
    """

    class FakeCollector:
        """Yield one synthetic rollout without creating environments or workers."""

        def __iter__(self):
            done = torch.zeros(64, 1, dtype=torch.bool)
            done[-1] = True
            reward = torch.zeros(64, 1)
            yield TensorDict(
                {
                    "next": TensorDict(
                        {"done": done, "terminated": done.clone(), "reward": reward},
                        batch_size=[64],
                    )
                },
                batch_size=[64],
            )

        def shutdown(self) -> None:
            pass

    class OutOfMemoryTrainer(FailingTrainer):
        @override
        def _make_collector(self, remaining_frames: int) -> FakeCollector:
            return FakeCollector()

        @override
        def _update(self, data: TensorDict) -> dict[str, float] | None:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    _, run = fake_wandb
    exit_codes: list[int | None] = []
    monkeypatch.setattr(
        run, "finish", lambda exit_code=None: exit_codes.append(exit_code)
    )
    trainer = OutOfMemoryTrainer(
        env_factories=[],
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=128,
        use_parallel_env=False,
        callbacks=[WeightsAndBiases(project="pokemon-tcg-ai", mode="offline")],
    )

    with pytest.raises(
        torch.OutOfMemoryError, match="CUDA out of memory \\(simulated\\)"
    ):
        trainer.train()

    assert exit_codes == [1]
    assert len(run.log_lines) == 1
    outgoing_message = run.log_lines[0]
    assert "Traceback (most recent call last):" in outgoing_message
    assert "in _update" in outgoing_message
    assert "OutOfMemoryError: CUDA out of memory (simulated)" in outgoing_message


def test_wandb_marks_a_clean_run_finished(
    fake_wandb, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A run that completes normally is still finished with no exit code.
    """
    _, run = fake_wandb
    exit_codes: list[int | None] = []
    monkeypatch.setattr(
        run, "finish", lambda exit_code=None: exit_codes.append(exit_code)
    )

    callback = WeightsAndBiases(project="pokemon-tcg-ai", mode="offline")
    callback.on_train_start({})
    callback.on_train_end({"frames": 5000000.0})

    assert exit_codes == [None]

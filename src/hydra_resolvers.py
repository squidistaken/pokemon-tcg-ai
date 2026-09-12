import os

from omegaconf import OmegaConf

_RUN_UID = "run_uid"


def run_uid() -> str:
    """
    Identifier separating this run from others that start at the same instant.

    Hydra stamps its run directory with a wall-clock timestamp, which is not
    unique in practice: Slurm releases the arms of a sweep together, and four
    arms landing in the same second share one output directory and therefore
    one ``checkpoints/``. Runs then overwrite each other's identically-named
    snapshots and load each other's policies as self-play opponents. The job id
    separates scheduled runs (the array task id too, so array elements stay
    distinct); the pid covers runs started outside Slurm.

    :return: The Slurm job identifier, or ``p<pid>`` when unscheduled.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return f"p{os.getpid()}"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{job_id}_{task_id}" if task_id else job_id


def register_resolvers() -> None:
    """
    Register this project's OmegaConf resolvers, once per process.

    Every ``@hydra.main`` entry point calls this at import time: the run
    directory in ``conf/config.yaml`` interpolates ``${run_uid:}``, and
    resolution happens wherever the config is composed.
    """
    if not OmegaConf.has_resolver(_RUN_UID):
        # Cached, so every interpolation in a run reads the same value.
        OmegaConf.register_new_resolver(_RUN_UID, run_uid, use_cache=True)

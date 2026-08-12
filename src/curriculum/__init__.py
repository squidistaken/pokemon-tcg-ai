"""
Prioritized Level Replay over deck-archetype matchups.

The package root deliberately re-exports only the modules that depend on
nothing outside it. :class:`~src.curriculum.callback.CurriculumStateCallback`
is imported from ``src.curriculum.callback`` directly, because it subclasses a
training callback and re-exporting it here would make ``src.env`` import
``src.training`` through this package.
"""

from .archetype_index import ArchetypeIndex
from .curriculum import Curriculum, build_curriculum
from .deck_sampler import CurriculumDeckSampler
from .handles import CurriculumHandles
from .level_buffer import LevelBuffer

__all__ = [
    "ArchetypeIndex",
    "Curriculum",
    "CurriculumDeckSampler",
    "CurriculumHandles",
    "LevelBuffer",
    "build_curriculum",
]

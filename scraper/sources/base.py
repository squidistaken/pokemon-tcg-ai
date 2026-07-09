"""Common interface for deck sources."""

from __future__ import annotations

import abc
from collections.abc import Iterable

from ..models import RawDeck


class DeckSource(abc.ABC):
    """A source yields :class:`RawDeck` objects, however it obtains them."""

    name: str = "base"

    @abc.abstractmethod
    def iter_decks(self, **kwargs) -> Iterable[RawDeck]:
        """Yield RawDecks. Implementations accept source-specific kwargs
        (e.g. ``limit``, ``fmt``, ``pages``, ``input``)."""
        raise NotImplementedError

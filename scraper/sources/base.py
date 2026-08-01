from __future__ import annotations

import abc
from collections.abc import Iterable

from ..models import RawDeck


class DeckSource(abc.ABC):
    """Common interface for deck sources: something that yields ``RawDeck``s."""

    name: str = "base"

    @abc.abstractmethod
    def iter_decks(self, **kwargs) -> Iterable[RawDeck]:
        """Yield RawDecks. Implementations accept source-specific kwargs.

        :param kwargs: Source-specific options (see each implementation).
        :return: An iterable of :class:`~scraper.models.RawDeck`.
        """
        raise NotImplementedError

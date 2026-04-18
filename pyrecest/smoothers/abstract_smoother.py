"""Abstract base class for all smoothers."""

from abc import ABC, abstractmethod


class AbstractSmoother(ABC):
    """Abstract base class for all smoothers."""

    @abstractmethod
    def smooth(self, *args, **kwargs):
        """Smooth a sequence of states produced by a forward pass."""

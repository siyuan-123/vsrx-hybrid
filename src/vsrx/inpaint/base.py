from __future__ import annotations

from abc import ABC, abstractmethod

from vsrx.domain.contracts import InpaintRequest, InpaintResult


class BaseInpainter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def inpaint(self, request: InpaintRequest) -> InpaintResult: ...

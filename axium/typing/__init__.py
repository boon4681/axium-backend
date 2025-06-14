from __future__ import annotations
from typing import Literal, TypedDict, Optional
from abc import ABC, abstractmethod
from enum import Enum


class StrEnum(str, Enum):
    """Base class for string enums. Python's StrEnum is not available until 3.11."""

    def __str__(self) -> str:
        return self.value


class TYPE(StrEnum):
    STRING = ""


class InputType(TypedDict):
    type: str


class AxiumNode(ABC):
    __CATEGORY__: str
    __FUNCTION__: str
    __DESCRIPTION__: str

    @classmethod
    @abstractmethod
    def INPUT_TYPES(s) -> list[tuple[str, TYPE, InputType]]:
        return []

    @classmethod
    @abstractmethod
    def OUTPUT_TYPES(s) -> list[tuple[str, TYPE]]:
        return []

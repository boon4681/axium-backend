from __future__ import annotations
from typing_extensions import NotRequired
from typing import Literal, TypedDict, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum


class StrEnum(str, Enum):
    """Base class for string enums. Python's StrEnum is not available until 3.11."""

    def __str__(self) -> str:
        return self.value


class TYPE(StrEnum):
    STRING = "AXIUM.STRING"
    INT = "AXIUM.INT"
    FLOAT = "AXIUM.FLOAT"


class InputType(TypedDict):
    type: str


class AxiumNode(ABC):
    __CATEGORY__: str
    __FUNCTION__: str
    __DESCRIPTION__: Union[str, None]

    @classmethod
    @abstractmethod
    def INPUT_TYPES(s) -> list[tuple[str, TYPE, InputType]]:
        return []

    @classmethod
    @abstractmethod
    def OUTPUT_TYPES(s) -> list[tuple[str, TYPE]]:
        return []


class InputTypeString(InputType):
    type = TYPE.STRING
    default: NotRequired[str]


class InputTypeInt(InputType):
    type = TYPE.INT
    default: NotRequired[int]
    min: NotRequired[int]
    max: NotRequired[int]

class InputTypeFloat(InputType):
    type = TYPE.FLOAT
    default: NotRequired[float]
    min: NotRequired[float]
    max: NotRequired[float]
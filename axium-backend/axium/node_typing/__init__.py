from __future__ import annotations
from typing_extensions import NotRequired
from typing import Literal, TypedDict, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel


class StrEnum(str, Enum):
    """Base class for string enums. Python's StrEnum is not available until 3.11."""

    def __str__(self) -> str:
        return self.value


class TYPE(StrEnum):
    STRING = "AXIUM.STRING"
    INT = "AXIUM.INT"
    FLOAT = "AXIUM.FLOAT"


class InputOptions(TypedDict):
    required: NotRequired[bool]


class AxiumNodeINPUT_TYPES(BaseModel):
    ports: list[tuple[str, TYPE, InputOptions]]
    parameters: Optional[list[tuple[str, TYPE, InputOptions]]]


class AxiumNode(ABC):
    __CATEGORY__: str
    __FUNCTION__: str
    __DESCRIPTION__: Union[str, None]

    @classmethod
    @abstractmethod
    def INPUT_TYPES(s) -> AxiumNodeINPUT_TYPES:
        return {
            "ports": []
        }

    @classmethod
    @abstractmethod
    def OUTPUT_TYPES(s) -> list[tuple[str, TYPE]]:
        return []
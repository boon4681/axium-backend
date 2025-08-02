from __future__ import annotations
from typing_extensions import TypedDict, NotRequired, Literal, Optional, Union, Any
from abc import ABC, abstractmethod

BASIC_TYPE_LIST = Literal[
    "axium.file",
    "axium.select",
    "axium.int",
    "axium.float",
    "axium.boolean",
    "axium.string",
    "axium.image",
    
]

TYPE = Union[str, BASIC_TYPE_LIST]


class AxiumNodeInputPortMeta(TypedDict):
    alias: NotRequired[str]
    """
    Used as a display name
    """

    required: NotRequired[bool]
    """
    Tell backend does this required or not
    """

    color: NotRequired[str]
    """
    Port displayed color
    """


class AxiumNodeOutputPortMeta(TypedDict):
    alias: NotRequired[str]
    """
    Used as a display name
    """

    color: NotRequired[str]
    """
    Port displayed color
    """


class AxiumNodeParameterMeta(TypedDict):
    label: NotRequired[str]
    """
    Simple label text
    """

    placeholder: NotRequired[str]
    """
    Placeholder text for input
    """

    inline: NotRequired[bool]
    """
    Place label and input in the same line
    """


class AxiumNode(ABC):
    id: str
    name: str
    category: Union[str, None]

    inputs: dict[str, tuple[str, AxiumNodeInputPortMeta]] = {}
    outputs: dict[str, tuple[str, AxiumNodeOutputPortMeta]] = {}
    parameters: Optional[dict[str, tuple[str, AxiumNodeParameterMeta]]] = None

    @classmethod
    @abstractmethod
    def validate_inputs(cls, inputs: dict):
        return {}

    @classmethod
    def process_inputs(cls, inputs: dict):
        return inputs

    @classmethod
    @abstractmethod
    def run(cls, parameters: dict, inputs: dict):
        """
        define a function to run
        """
        return {}

    @classmethod
    def get_node_details(cls):
        return {
            "id": cls.id,
            "name": cls.name,
            "category": cls.category,
            "inputs": [
                {
                    "name": internal_name,
                    "type": value[0],
                    "meta": value[1]
                }
                for internal_name, value in cls.inputs.items()
            ],
            "outputs": [
                {
                    "name": internal_name,
                    "type": value[0],
                    "meta": value[1]
                }
                for internal_name, value in cls.outputs.items()
            ],
            "parameters": [
                {
                    "name": internal_name,
                    "type": value[0],
                    "meta": value[1]
                }
                for internal_name, value in cls.parameters.items()
            ] if cls.parameters else None
        }

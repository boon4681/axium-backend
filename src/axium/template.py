import inspect
from axium import Axium
from axium.utils import split_camel_case


class AxiumTemplateRegisterMeta(type):
    def __init__(cls, name, bases, attrs):
        if name != 'AxiumTemplate':
            Axium.register(cls)

        super().__init__(name, bases, attrs)


class AxiumTemplate(metaclass=AxiumTemplateRegisterMeta):

    id: str = None
    name: str = None
    category: str = None

    input:    dict[str, str] = {}
    output:   dict[str, str] = {}
    parameter: dict[str, str] = {}

    input_alias:    dict[str, str] = {}
    output_alias:   dict[str, str] = {}
    parameter_alias: dict[str, str] = {}

    object = {}

    @classmethod
    def validate_input(cls, *args, **kwargs):
        return {}

    @classmethod
    def validate_parameter(cls, *args, **kwargs):
        return {}

    @classmethod
    def run(cls, *args, **kwargs):
        """
            define a function to run
        """
        return {}

    @classmethod
    def gen_object(cls):
        class_name = " ".join(split_camel_case(cls.__name__))

        if cls.name is None:
            cls.name = class_name
        cls.object = {
            "id": cls.id,
            "name": cls.name,
            "category": cls.category,
            "input": [
                {
                    "name": cls.input_alias[key] if key in cls.input_alias else key,
                    "type": value
                }
                for key, value in cls.input.items()
            ],
            "output": [
                {
                    "name": cls.output_alias[key] if key in cls.output_alias else key,
                    "type": value
                }
                for key, value in cls.output.items()
            ],
            "parameter": [
                {
                    "name": cls.parameter_alias[key] if key in cls.output_alias else key,
                    "type": value
                }
                for key, value in cls.parameter.items()
            ],
        }

    @classmethod
    def execute(cls, *args, **kwargs):
        """
            Do not override, this is for lib to run result
        """

        # Custom input check
        validate_input_res = cls.validate_input(*args, **kwargs)
        if len(validate_input_res) > 0:
            for _, (is_pass, msg) in validate_input_res.items():
                if not is_pass:
                    raise RuntimeError(msg)

        # Custom parameter check
        validate_parameter_res = cls.validate_parameter(*args, **kwargs)
        if len(validate_parameter_res) > 0:
            for _, (is_pass, msg) in validate_parameter_res.items():
                if not is_pass:
                    raise RuntimeError(msg)

        result = cls.run(*args, **kwargs)

        keys = cls.output.keys()
        values = list(result) if hasattr(result, "__iter__") else [result]

        return dict(zip(keys, values))

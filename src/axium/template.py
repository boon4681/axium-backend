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

    input:    dict[str, type] = {}
    output:   dict[str, type] = {}
    property: dict[str, type] = {}

    input_alias:    dict[str, str] = {}
    output_alias:   dict[str, str] = {}
    property_alias: dict[str, str] = {}

    object = {}

    @classmethod
    def validate_input(cls, *args, **kwargs):
        return {}

    @classmethod
    def validate_property(cls, *args, **kwargs):
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
                    "type": value.__name__
                }
                for key, value in cls.input.items()
            ],
            "output": [
                {
                    "name": cls.output_alias[key] if key in cls.output_alias else key,
                    "type": value.__name__
                }
                for key, value in cls.output.items()
            ],
            "property": [
                {
                    "name": cls.property_alias[key] if key in cls.output_alias else key,
                    "type": value.__name__
                }
                for key, value in cls.property.items()
            ],
        }

    @classmethod
    def execute(cls, *args, **kwargs):
        """
            Do not override, this is for lib to run result
        """
        
        validate_input_res = cls.validate_input(*args, **kwargs)

        if len(validate_input_res) > 0:
            for param, (is_pass, msg) in validate_input_res.items():
                if is_pass: continue
                
                print(param, msg)

            return 

        validate_property_res = cls.validate_property(*args, **kwargs)
        if validate_property_res is not None:
            return

        pass
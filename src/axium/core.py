import importlib
import logging
from pathlib import Path

from axium.utils import split_camel_case

class Axium:
    registry = []
    registry_id: int = 0

    @classmethod
    def register(cls, template_cls):

        name = " ".join(split_camel_case(template_cls.__name__))

        template_cls.name = name
        template_cls.id = cls.registry_id

        template_cls.gen_object()

        cls.registry.append(template_cls)
        cls.registry_id += 1

    @classmethod    
    def setup(cls):
        """
        Load the template node, called on start
        """

        nodes_dir = "src/templates"
        module_name = "templates"

        for path in Path(nodes_dir).rglob("*.py"):
            file_name = path.name[:-3]
            folder_name = str(path.parent)[len(nodes_dir) + 1:].replace("\\", ".")

            if file_name.startswith("__") or file_name.endswith("__"):
                continue

            full_module_name = f"{module_name}.{folder_name}.{file_name}" if len(folder_name) != 0 else f"{module_name}.{file_name}"

            try:
                importlib.import_module(full_module_name)
                print(f"{full_module_name} imported")
            except Exception as e:
                print(f"Failed to import {full_module_name}\n Cause: {e}")
    
    @classmethod
    def get_node(cls, id: int):
        if id < 0 or id >= len(cls.registry):
            return None
        
        return cls.registry[id]

    @classmethod
    def get_all_node(cls):
        return [template.object for template in cls.registry]
    
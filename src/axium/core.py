import importlib
import logging
from pathlib import Path

from axium.utils import split_camel_case

logger = logging.getLogger("Axium's Backend")
logger.setLevel(logging.INFO)

class Axium:
    registry = {}

    @classmethod
    def register(cls, template_cls):
        if template_cls.id is None:
            raise ImportError

        template_cls.gen_object()
        cls.registry[template_cls.id] = template_cls

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
                logger.info(f"{full_module_name} imported")
            except Exception as e:
                logger.info(f"Failed to import {full_module_name}\n Cause: {e}")
    
    @classmethod
    def get_node(cls, id: str):
        if id not in cls.registry:
            return None
        
        return cls.registry[id]

    @classmethod
    def get_all_node(cls):
        return [template.object for template in cls.registry.values()]
    
import time
import traceback
import importlib
import importlib.util
import os
import sys
import logging
import inspect
import typing

from . import folder
from . import typing as node_typing
logging.basicConfig(level=logging.INFO)

LOADED_MODULE_DIRS = {}

NODE_CLASS_MAPPINGS = {}


def get_module_name(path: str) -> str:
    base_path = os.path.basename(path)
    if os.path.isfile(path):
        base_path = os.path.splitext(base_path)[0]
    return base_path


def load_module(module_path: str, ignore=set(), module_parent="nodes") -> bool:
    module_name = get_module_name(module_path)
    sys_module_name = ""
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
        sys_module_name = module_name
    elif os.path.isdir(module_path):
        sys_module_name = module_path.replace(".", "_x_")

    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(
                sys_module_name, module_path)
            module_dir = os.path.split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(
                sys_module_name, os.path.join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[sys_module_name] = module
        module_spec.loader.exec_module(module)

        LOADED_MODULE_DIRS[module_name] = os.path.abspath(module_dir)

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = os.path.abspath(os.path.join(
                module_dir, getattr(module, "WEB_DIRECTORY")))
            # TODO
        if hasattr(module, "EXPORT_NODES") and getattr(module, "EXPORT_NODES") is not None:
            for name, node in module.EXPORT_NODES.items():
                if name not in ignore:
                    NODE_CLASS_MAPPINGS[name] = node
                    node.RELATIVE_PYTHON_MODULE = "{}.{}".format(
                        module_parent, get_module_name(module_path))
            return True
        else:
            logging.warning(
                f"Skip {module_path} module for nodes due to the lack of EXPORT_NODES.")
    except Exception as e:
        logging.warning(traceback.format_exc())
        logging.warning(
            f"Cannot import {module_path} module for nodes: {e}")
    return False


def load_nodes():
    node_paths = folder.get_folder_paths("nodes")
    base_node_names = set(NODE_CLASS_MAPPINGS.keys())
    for path in node_paths:
        possible_modules = os.listdir(os.path.realpath(path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")
        for possible_module in possible_modules:
            module_path = os.path.join(path, possible_module)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py":
                continue
            if module_path.endswith(".disabled"):
                continue
            time_before = time.perf_counter()
            success = load_module(
                module_path, base_node_names, module_parent="nodes")
            time_diff = time.perf_counter() - time_before
            logging.info("{:6.1f} seconds{}: {}".format(
                time_diff, "" if success else " Failed", module_path)
            )


def validate_interface(cls, base):
    errors = []

    hints = typing.get_type_hints(base)
    hints_extra = typing.get_type_hints(base, include_extras=True)
    for attr in getattr(base, '__annotations__', {}):
        t = hints_extra[attr]
        optional = (isinstance(None, hints[attr]))\
            or (hasattr(t, "__origin__") and t.__origin__ is typing.NotRequired)
        dict_optional = False
        if not optional and isinstance(cls, typing.Dict):
            if cls.get(attr) != getattr(base, attr):
                errors.append(
                    f"Method {name} signature mismatch. Expected: {base_params}, Got: {cls_params}")
            else:
                dict_optional = True
        if not hasattr(cls, attr) and not optional and not dict_optional:
            errors.append(f"Missing: {attr}")

    for name, method in inspect.getmembers(base, inspect.ismethod):
        if getattr(method, '__isabstractmethod__', False):
            if not hasattr(cls, name) or not callable(getattr(cls, name)):
                errors.append(f"Missing method: {name}")
            else:
                cls_method = getattr(cls, name)
                base_attr = getattr(base, name)
                try:
                    base_sig = inspect.signature(base_attr)
                    cls_sig = inspect.signature(cls_method)

                    base_params = list(base_sig.parameters.keys())
                    cls_params = list(cls_sig.parameters.keys())

                    if base_params != cls_params:
                        errors.append(
                            f"Method {name} signature mismatch. Expected: {base_params}, Got: {cls_params}")

                except (ValueError, TypeError) as e:
                    errors.append(
                        f"Could not compare signatures for {name}: {e}")
    return len(errors) == 0, errors


def get_node(id: str):
    node = NODE_CLASS_MAPPINGS[id]
    valid, errors = validate_interface(node, node_typing.AxiumNode)
    if valid:
        node = typing.cast(node_typing.AxiumNode, node)
        for name, type, meta in node.INPUT_TYPES():
            meta["type"] = type
            print(name, type, validate_interface(
                meta, node_typing.InputTypeFloat))
        pass
    else:
        for err in errors:
            logging.error(err)


__all__ = [
    "NODE_CLASS_MAPPINGS"
]

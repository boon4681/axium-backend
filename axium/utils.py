import os
import platform

from pydantic import TypeAdapter, ValidationError
from axium.validator import AxiumErrorJSONResponse
from .logger import logging
import inspect
import typing_extensions as typing
from core import folder
from axium import node_typing
from typeguard import check_type
from pathlib import Path
from axium.axium_typing import *


def AxiumErrorJSON(data):
    v = TypeAdapter(AxiumErrorJSONResponse)
    try:
        return v.validate_python(data)
    except ValidationError as e:
        return None


def validate_interface(cls, base):
    errors = []

    hints = typing.get_type_hints(base)
    hints_extra = typing.get_type_hints(base, include_extras=True)
    for attr in getattr(base, '__annotations__', {}):
        t = hints_extra[attr]
        optional = False
        try:
            optional = check_type(hints[attr], typing.Optional) or (isinstance(None, hints[attr]))\
                or (hasattr(t, "__origin__") and t.__origin__ is typing.NotRequired)
        except:
            optional = False
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


def list_files_tree(directory: str) -> FileTree:
    tree: FileTree = []
    root_path = Path(directory).resolve()

    if not root_path.is_dir():
        print(f"Error: Provided path '{directory}' is not a valid directory.")
        return []

    try:
        sorted_paths = sorted(
            root_path.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
        for path in sorted_paths:
            node_id = path.as_posix()

            if path.is_dir():
                folder_node: Folder = {
                    'type': 'folder',
                    'id': node_id,
                    'name': path.name,
                    'files': list_files_tree(str(path))  # The recursive call
                }
                tree.append(folder_node)
            elif path.is_file():
                file_node: File = {
                    'type': 'file',
                    'id': node_id,
                    'name': path.name
                }
                tree.append(file_node)

    except PermissionError:
        print(f"Warning: Permission denied for directory '{root_path}'")
    except FileNotFoundError:
        print(f"Warning: Directory '{root_path}' not found during scan.")

    return tree


def HOME_DIR():
    if platform.system() == "Windows":
        return Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'axium'
    else:
        raise RuntimeError("not support")

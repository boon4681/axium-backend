import os
from axium import config

folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}
if config['base_nodes']:
    base_path = os.path.abspath(config['base_nodes'])
else:
    base_path = os.path.dirname(os.path.realpath(__file__))

folder_names_and_paths["nodes"] = ([base_path], set())


def get_folder_paths(folder_name: str) -> list[str]:
    return folder_names_and_paths[folder_name][0][:]

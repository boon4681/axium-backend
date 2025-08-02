import tomllib
import os
from core.logger import logging

config: dict[str] = {}
config['base_nodes'] = os.path.join(os.getcwd(), "nodes")

path = os.path.join(os.getcwd(), "axium.toml")
with open(path, "rb") as f:
    data = tomllib.loads(f.read().decode("utf-8"))
    if 'axium' in data and data.get('axium') is not None:
        # TODO
        pass
    else:
        logging.warning(f"Failed to load axium.toml at {path}")
        logging.info(f"Fallback to default.")

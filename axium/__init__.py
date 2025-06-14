__version__ = "0.1"

from .config import config as config
from . import folder
from . import register

__all__ = [
    "config",
    folder.__name__,
    register.__name__
]

import sys


def _add_cwd_to_sys():
    """
    adding cwd to sys path to resolve module not found.
    """
    import os
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


_add_cwd_to_sys()

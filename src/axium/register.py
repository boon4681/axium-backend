from typing import Callable
import inspect

executor: dict[str, Callable] = {}

def node(func: Callable):
    """Decorator for register node"""
    
    # Parameter type check
    for _, param in inspect.signature(func).parameters.items():
        if param.annotation != int and param.annotation != str:
            raise TypeError("Node function input only accept integer or string")

    executor[func.__name__] = func

    return func
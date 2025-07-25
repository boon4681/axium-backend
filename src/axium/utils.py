import re
from typing import List

def split_camel_case(text: str) -> List[str]:
    """
    Split a CamelCase string into a list of words.
    """
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', text)
import re

class JSONObject:
    def __init__(self):
        pass

def split_camel_case(text: str):
    """
        find all the word that have upper char follow by lower chars
    """
    
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', text)
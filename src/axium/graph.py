class AxiumGraph:
    
    def dfs(cls, root: int):
        pass

    def add_edge(cls, a: int, b: int):
        pass

    
class AxiumNode:
    def __init__(
                self,
                id: int,
                template_id: int,
                next: list[int]
                ):
        self.id = id
        self.template_id = template_id
        self.next = next
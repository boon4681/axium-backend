from collections import defaultdict

class AxiumGraphExucutor:
    graph = defaultdict(list)
    node = set()

    @classmethod
    def add_edge(cls, u, v):
        cls.graph[u].append(v)
        cls.node.add(u)
        cls.node.add(v)

    @classmethod
    def clear(cls):
        cls.graph.clear()

    @classmethod
    def run(cls):
        sorted_node = cls.get_topological_sorted_node()

        for node in sorted_node:
            print(node)

        

    @classmethod
    def get_topological_sorted_node(cls):
        """
            Run Topological sort and exec them one by one
        """
        if cls.is_cyclic_graph():
            raise RuntimeError("Graph have a cycle, aborted")

        sorted_node = []
        visited = defaultdict(list)

        def topological_dfs(u):
            visited[u] = True
            for v in cls.graph[u]:
                if not visited[v]:
                    topological_dfs(v)

            sorted_node.append(u)

        for u in cls.node:
            if not visited[u]:
                topological_dfs(u)

        sorted_node = list(reversed(sorted_node))

        return sorted_node

    @classmethod
    def is_cyclic_graph(cls):        
        visited = defaultdict(list)
        rec_visited = defaultdict(list)

        def cyclic_dfs(u):
            if rec_visited[u]:
                return True
            
            if visited[u]:
                return False
            
            visited[u] = True
            rec_visited[u] = True

            if u in cls.node:
                for v in cls.graph[u]:
                    if cyclic_dfs(v):
                        return True
            
            rec_visited[u] = False

            return False
            
        for u in cls.node:
            if not visited[u] and cyclic_dfs(u):
                return True

        return False 

class AxiumNode:
    def __init__(
                self,
                id: int,
                template_id: str,
                next: list[int]
                ):
        self.id = id
        self.template_id = template_id
        self.next = next
        self.last_result = None
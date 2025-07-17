from collections import defaultdict

from axium.core import Axium
from axium.template import AxiumTemplate

class AxiumNode:
    def __init__(
                self,
                id: int,
                template_id: str,
                parameters: dict,
                property: dict,
                required: list
                ):
        self.id = id
        self.template_id = template_id
        self.parameters  = parameters
        self.property = property
        self.required = required
        self.last_result = {}

    @classmethod
    def from_dict(cls, data):
        return cls(
            id          = data["id"],
            template_id = data["template_id"],
            required    = data["required"],
            parameters  = data["parameters"],
            property    = data["property"],
        )

class AxiumEdge:
    def __init__(
                self,
                node_id:    int,
                src_output: str,
                dst_param:  str
                ):
        self.node_id = node_id
        self.src_output = src_output
        self.dst_param = dst_param

    @classmethod
    def from_dict(cls, data):
        return cls(
            node_id=data["node_id"],
            src_output=data["src_output"],
            dst_param=data["dst_param"]
        )

class AxiumGraph:
    graph = defaultdict(list)
    axium_node: dict[str, AxiumNode] = {}

    @classmethod
    def load_from_object(cls, objects):
        for object in objects:
            instance = AxiumNode.from_dict(object)
            cls.add_node(instance)
            
            for required in instance.required:
                cls.add_edge(required["id"], instance.id)

        if cls.is_cyclic_graph():
            cls.clear()
            raise RuntimeError("Graph has cycle, load object aborted")
        
    @classmethod
    def add_node(cls, node: AxiumNode):
        cls.axium_node[node.id] = node

    @classmethod
    def add_edge(cls, u: int, v: int):
        cls.graph[u].append(v)

    @classmethod
    def clear(cls):
        cls.graph.clear()
        cls.axium_node.clear()

    @classmethod
    def run(cls):
        """
            Run get topological sort of node and exec them one by one
        """

        for id in cls.get_topological_sorted_node():
            node = cls.axium_node[id]
            template: AxiumTemplate = Axium.get_node(node.template_id)

            for req in node.required:
                node_req_id         = req["id"]
                node_req_param_name = req["src_output"]
                node_param_name     = req["dst_param"]

                node_req = cls.axium_node[node_req_id]
                node.parameters[node_param_name] = node_req.last_result[node_req_param_name]

            node.last_result = template.execute(**node.parameters)

        return list(cls.axium_node.values())

    @classmethod
    def get_topological_sorted_node(cls):
        sorted_node = []
        visited = defaultdict(list)

        def topological_dfs(u):
            visited[u] = True
            for v in cls.graph[u]:
                if not visited[v]:
                    topological_dfs(v)

            sorted_node.append(u)

        for u in cls.axium_node.keys():
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

            if u in cls.axium_node.keys():
                for v in cls.graph[u]:
                    if cyclic_dfs(v):
                        return True
            
            rec_visited[u] = False

            return False
            
        for u in cls.axium_node.keys():
            if not visited[u] and cyclic_dfs(u):
                return True

        return False 
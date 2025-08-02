from collections import defaultdict
from pydantic import TypeAdapter, ValidationError
from axium.validator import *
from axium import register
from axium import node_typing
from .logger import logging

class Graph:
    graph = defaultdict(list)
    axium_node: dict[str, AxiumNodeData] = {}

    @staticmethod
    def load_from_object(objects):
        v = TypeAdapter(AxiumGraphData)
        data = None
        try:
            data = v.validate_python(objects)
        except ValidationError as e:
            logging.error(e)
            return None
        g = Graph()
        for node in data.nodes:
            g.add_node(node)
        for edge in data.edges:
            g.add_edge(edge[0].node_id, edge[1].node_id)
        if g.is_cyclic_graph():
            raise RuntimeError("Graph has cycle, load object aborted")
        return g

    def add_node(cls, node: AxiumNodeData):
        cls.axium_node[node.id] = node

    def add_edge(cls, u: str, v: str):
        cls.graph[u].append(v)

    def clear(cls):
        cls.graph.clear()
        cls.axium_node.clear()

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

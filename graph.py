import unittest
import torch
import networkx as nx

from typing import List, Set, Hashable
from torch_geometric.data import Data
from abc import ABC, abstractmethod


class DAG(ABC):
    def __init__(self):
        self.nodes = set()
    
    # def nodes(self):
    #     return self._nodes

    @abstractmethod
    def edges(self):
        pass

    @abstractmethod
    def add_node(self, node: Hashable):
        pass
    
    @abstractmethod
    def add_edge(self, node_from: Hashable, node_to: Hashable):
        pass
    
    @abstractmethod
    def remove_node(self, node):
        pass
    
    @abstractmethod
    def remove_edge(self, node_from, node_to):
        pass
    
    @abstractmethod
    def get_roots(self):
        pass
    
    @abstractmethod
    def get_leaves(self):
        pass
    
    @abstractmethod
    def get_successors(self, node):
        pass
    
    @abstractmethod
    def get_predecessors(self, node):
        pass

    @abstractmethod
    def print(self, mapping=None):
        pass

    def topological_sort(self) -> List :
        visited = set()
        order = []
        stack = list(self.get_leaves())
        while len(stack) > 0:
            n = stack.pop()
            if n in visited:
                continue
            stack = stack + list(self.get_predecessors(n))
            visited.add(n)
            order.append(n)
        return order[::-1]
        

class PytorchDag(DAG):
    # TODO(wilson): implement this
    def __init__(self):
        super().__init__()


class AdjacencyListDAG(DAG):
    def __init__(self):
        super().__init__()
        self.adj_list = {}

    def edges(self):
        return set([(u, v) for u in self.adj_list for v in self.adj_list[u]])

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)

        if node not in self.adj_list:
            self.adj_list[node] = set()

    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        self.adj_list[node1].add(node2)

    

    def remove_node(self, node):
        if node in self.adj_list:
            # Remove node from all its parents' children lists
            for parent in self.get_predecessors(node):
                self.adj_list[parent].remove(node)
            # Remove node and its children from the adjacency list
            self.adj_list.pop(node)
        self.nodes.remove(node)

    def remove_edge(self, node1, node2):
        if node1 in self.adj_list and node2 in self.adj_list[node1]:
            self.adj_list[node1].remove(node2)

    def get_roots(self) -> Set[Hashable]:
        roots = []
        for node in self.adj_list:
            if not self.get_predecessors(node):
                roots.append(node)
        return set(roots)

    def get_leaves(self) -> Set[Hashable]:
        leaves = []
        for node in self.adj_list:
            if not self.get_successors(node):
                leaves.append(node)
        return set(leaves)

    def get_successors(self, node) -> Set[Hashable]:
        if node not in self.adj_list:
            return []
        return self.adj_list[node]

    def get_predecessors(self, node) -> Set[Hashable]:
        predecessors = []
        for parent, children in self.adj_list.items():
            if node in children:
                predecessors.append(parent)
        return predecessors

    def topological_sort(self):
        return super().topological_sort()

    def print(self, mapping=None):
        #TODO: Implement
        return ""


class NxDAG(DAG):
    def __init__(self):
        self.graph = nx.DiGraph()

    def get_nodes(self):
        return list(self.graph.nodes())

    def edges(self):
        return list(self.graph.edges())

    def add_node(self, node: Hashable):
        self.graph.add_node(node)

    def add_edge(self, node_from: Hashable, node_to: Hashable):
        self.graph.add_edge(node_from, node_to)

    def remove_node(self, node):
        self.graph.remove_node(node)

    def remove_edge(self, node_from, node_to):
        self.graph.remove_edge(node_from, node_to)

    def get_roots(self):
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

    def get_leaves(self):
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]

    def get_successors(self, node):
        return self.graph.successors(node)

    def get_predecessors(self, node):
        return self.graph.predecessors(node)

    def topological_sort(self) -> List:
        return list(nx.topological_sort(self.graph))

    def print(self, mapping=None):
        # print(mapping)
        relabeled_graph = nx.relabel_nodes(self.graph, mapping)
        print(relabeled_graph)
        agraph = nx.drawing.nx_agraph.to_agraph(relabeled_graph)
        agraph.draw('./viz/current_task.png', prog='dot')
        # nx.draw(relabeled_graph, with_labels=True)
        # plt.show()
        # return nx.utils.nx_ascii(relabeled_graph)
        return nx.drawing.nx_agraph.to_agraph(relabeled_graph).string()


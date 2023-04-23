
import unittest
import pytest
from typing import List
from graph import DAG, AdjacencyListDAG
from abc import ABC, abstractmethod

class DAGTestCase(ABC):
    def is_topological_order_edge_approch(self, node_order):
        """
        From Ben Cooper

        Runtime:
            O(V * E)

        References:
            https://stackoverflow.com/questions/54174116/checking-validity-of-topological-sort
        """
        node_to_index = {n: idx for idx, n in enumerate(node_order)}
        print(node_order)
        for u, v in self.dag.edges():
            # For each edge, retrieve the index of each of its vertices in the ordering.
            ux = node_to_index[u]
            vx = node_to_index[v]
            # Compared the indices. If the origin vertex isn't earlier than
            # the destination vertex, return false.
            if ux >= vx:
                return False
        return True

    def setUp(self):
        # this method should be implemented in the subclass
        self.dag = self.create_dag()

    def tearDown(self):
        # clean up code here if needed
        pass

    @abstractmethod
    def create_dag(self):
        # this method should be implemented in the subclass
        pass

    def test_add_node(self):
        # test adding a node to the DAG
        self.dag.add_node('A')
        self.assertEqual(self.dag.nodes, {'A'})

    def test_add_edge(self):
        # test adding an edge to the DAG
        self.dag.add_edge('A', 'B')
        self.assertEqual(self.dag.get_successors('A'), {'B'})
    
    def test_edges(self):
        # test adding an edge to the DAG
        self.dag.add_edge('A', 'B')
        self.assertEqual(self.dag.edges(), {('A', 'B')})
        # test adding another edge to the DAG
        self.dag.add_edge('A', 'C')
        self.assertEqual(self.dag.edges(), {('A', 'B'), ('A', 'C')})

    def test_topological_sort(self):
        # test topological sorting of the DAG
        self.dag.add_edge('A', 'B')
        self.dag.add_edge('B', 'C')
        self.assertTrue(self.is_topological_order_edge_approch(self.dag.topological_sort()))
        self.dag.add_edge('D', 'B')
        self.assertEqual(self.dag.nodes, {'A', 'B', 'C', 'D'})
        self.assertTrue(self.is_topological_order_edge_approch(self.dag.topological_sort()))
        

class AdjListDAGTestCase(DAGTestCase, unittest.TestCase):
    def create_dag(self):
        # create an instance of the adj-list-based DAG
        dag = AdjacencyListDAG()
        return dag


class PyTorchDAGTestCase(DAGTestCase):
    def create_dag(self):
        # create an instance of the PyTorch-based DAG
        # using PyG library
        pass

if __name__ == '__main__':
    print("hello")
    unittest.main()


def get_dag_test_case(dag_class):
    class MyDAGTestCase(DAGTestCase):
        def create_dag(self):
            return dag_class()
    return MyDAGTestCase

@pytest.mark.parametrize("dag_class", [AdjacencyListDAG])
def test_dag(dag_class):
    dag_test_case = get_dag_test_case(dag_class)

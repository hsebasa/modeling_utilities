from standard_modeling.feature_selection import Node, DAG
import unittest


class TestDAG(unittest.TestCase):
    def test_topological_order(self):
        node_a = Node("A")
        node_b = Node("B")
        node_c = Node("C")
        node_d = Node("D")

        nodes = [node_a, node_b, node_c, node_d]
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]

        dag = DAG(nodes, edges)
        order = [node.name for node in dag.get_topological_order()]
        
        # Expected order should start with A, B or C must be before D
        self.assertTrue(order.index("A") < order.index("B"))
        self.assertTrue(order.index("A") < order.index("C"))
        self.assertTrue(order.index("B") < order.index("D"))
        self.assertTrue(order.index("C") < order.index("D"))
        
    def test_topological_order2(self):
        node_a = Node("A")
        node_b = Node("B")
        node_c = Node("C")
        node_d = Node("D")

        nodes = [node_a, node_b, node_c, node_d]
        edges = [("B", "D"), ("C", "D")]

        dag = DAG(nodes, edges)
        order = [node.name for node in dag.get_topological_order()]
        
        # Expected order should start with A, B or C must be before D
        self.assertTrue(order.index("A") < order.index("B"))
        self.assertTrue(order.index("A") < order.index("C"))
        self.assertTrue(order.index("B") < order.index("D"))
        self.assertTrue(order.index("C") < order.index("D"))
        
    def test_topological_order3(self):
        node_a = Node("A")
        node_b = Node("B")
        node_c = Node("C")
        node_d = Node("D")

        nodes = [node_b, node_c, node_a, node_d]
        edges = [("B", "D"), ("C", "D"), ("C", "A")]

        dag = DAG(nodes, edges)
        order = [node.name for node in dag.get_topological_order()]
        
        # Expected order should start with A, B or C must be before D
        self.assertTrue(order.index("A") > order.index("B"))
        self.assertTrue(order.index("A") > order.index("C"))
        self.assertTrue(order.index("A") < order.index("D"))
        self.assertTrue(order.index("B") < order.index("D"))
        self.assertTrue(order.index("C") < order.index("D"))
        
    def test_topological_order4(self):
        node_a = Node("A")
        node_b = Node("B")
        node_c = Node("C")
        node_d = Node("D")

        nodes = [node_b, node_c, node_a, node_d]
        edges = [("B", "D"), ("C", "D"), ("D", "A")]

        dag = DAG(nodes, edges)
        order = [node.name for node in dag.get_topological_order()]
        
        # Expected order should start with A, B or C must be before D
        self.assertTrue(order.index("A") > order.index("B"))
        self.assertTrue(order.index("A") > order.index("C"))
        self.assertTrue(order.index("A") > order.index("D"))
        self.assertTrue(order.index("B") < order.index("D"))
        self.assertTrue(order.index("C") < order.index("D"))

    def test_empty_dag(self):
        dag = DAG([], [])
        self.assertEqual(dag.get_topological_order(), [])

    def test_single_node_dag(self):
        node_a = Node("A")
        dag = DAG([node_a], [])
        self.assertEqual(dag.get_topological_order(), [node_a])

    def test_cycle_detection(self):
        node_a = Node("A")
        node_b = Node("B")
        
        with self.assertRaises(ValueError):
            DAG([node_a, node_b], [("A", "B"), ("B", "A")])


if __name__ == "__main__":
    unittest.main()

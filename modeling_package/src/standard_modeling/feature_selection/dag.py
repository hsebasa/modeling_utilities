from collections import deque, OrderedDict


class Node:
    def __init__(self, name, max_feats=None, params=None):
        """
        Initialize a Node with a name and a list of parameters.
        :param name: str, name of the node
        :param max_feats: str, maximum number of features to select from this node
        :param params: list of str, parameters associated with the node
        """
        self.name = name
        self.max_feats = max_feats
        self.params = params if params else []

    def __repr__(self):
        return f"Node(name={self.name}, max_feats={self.max_feats}, params={self.params})"


class DAG:
    def __init__(self, nodes, edges):
        """
        Initialize a Directed Acyclic Graph (DAG).
        :param nodes: list of Node objects
        :param edges: list of tuples (parent_name, child_name) defining dependencies
        """
        self.nodes = OrderedDict([(node.name, node) for node in nodes])  # Dictionary of nodes
        self.edges = {node.name: [] for node in nodes}  # Adjacency list representation
        
        # Validate and add edges
        for parent, child in edges:
            if parent not in self.nodes or child not in self.nodes:
                raise ValueError(f"Invalid edge ({parent} -> {child}): Both nodes must exist.")
            self.edges[parent].append(child)
        
        # Check for cycles
        if self.has_cycle():
            raise ValueError("The graph contains a cycle, which is not allowed in a DAG.")

    def has_cycle(self):
        """Detect cycles using depth-first search."""
        visited = set()
        stack = set()
        
        def dfs(node):
            if node in stack:
                return True  # Cycle detected
            if node in visited:
                return False
            
            visited.add(node)
            stack.add(node)
            for neighbor in self.edges[node]:
                if dfs(neighbor):
                    return True
            stack.remove(node)
            return False
        
        return any(dfs(node) for node in self.nodes)
    
    def __repr__(self):
        return f"DAG(nodes={list(self.nodes.keys())}, edges={self.edges})"
    
    def get_parents(self, node_name):
        """Return the parents of a given node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found in DAG.")
        return [parent for parent, children in self.edges.items() if node_name in children]

    def get_children(self, node_name):
        """Return the children of a given node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found in DAG.")
        return self.edges[node_name]
    
    def get_topological_order(self):
        """Return a list of nodes in topological order."""
        pos_dict = {a: i for i, a in enumerate(self.nodes.keys())}
        edges = OrderedDict([])
        for name in self.nodes:
            if name in self.edges:
                edges[name] = sorted(self.edges[name], key=pos_dict.get)
        
        in_degree = {node: 0 for node in self.nodes}
        for children in edges.values():
            for child in children:
                in_degree[child] += 1
        
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        ordered_nodes = []
        
        while queue:
            node = queue.popleft()
            ordered_nodes.append(self.nodes[node])
            for child in edges[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        assert len(ordered_nodes) == len(self.nodes) 
        return ordered_nodes
        
    def plot(self, figsize=(8, 6), draw_kw=None):
        """Graphically represent the DAG using NetworkX and Matplotlib."""
        import networkx as nx
        import matplotlib.pyplot as plt
      
        graph = nx.DiGraph()
        
        # Add edges
        for parent, children in self.edges.items():
            for child in children:
                graph.add_edge(parent, child)

        if draw_kw is None:
            draw_kw = dict()
          
        draw_kw = dict(
            with_labels=True,
            node_size=3000,
            node_color='lightblue',
            edge_color='gray',
            font_size=10,
            font_weight='bold'
        )|draw_kw
      
        plt.figure(figsize=figsize)
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')  # Use Graphviz for top-down ordering
        except ImportError:
            pos = nx.spring_layout(graph)
        nx.draw(graph, pos, **draw_kw)
        plt.title("DAG Representation")

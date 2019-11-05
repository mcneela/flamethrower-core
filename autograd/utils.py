def topological_sort(end_node):
    """
    Performs a topological start
    beginning with end_node and
    working backwards via parents
    until the start of the computation
    is reached.
    """
    stack = [end_node]
    visited = {}
    topo_sorted = []
    while stack:
        node = stack.pop()
        if node in visited:
            del topo_sorted[topo_sorted.index(node)]
        visited[node] = True
        parents = node.parents
        stack.extend(parents)
        topo_sorted.append(node)
    return topo_sorted

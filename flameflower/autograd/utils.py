def subval(x, i, v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)
    
def sum_with_none(x, y):
    if x is None:
        return y
    return x + y

def name(primitive):
    """
    Gets the __name__
    of a `Primitive`.
    """
    try:
        return primitive.__name__()
    except TypeError:
        return primitive.__name__

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

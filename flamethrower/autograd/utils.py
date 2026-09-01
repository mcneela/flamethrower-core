import flamethrower.autograd.tensor as ten
from collections import deque

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
    Return reachable nodes in backward-pass order using Kahn's algorithm.

    Graph edges point from an operation to its parents, so a node must appear
    before every parent that receives its gradient. Repeated parent edges are
    counted separately because expressions such as ``y + y`` contribute two
    gradients even though both edges lead to the same node.
    """
    if end_node is None:
        raise ValueError("Cannot topologically sort a graph without an end node.")

    # First discover each reachable node exactly once and count its incoming
    # child edges. The old traversal expanded parents again on every encounter,
    # which made a depth-n ``y = y + y`` graph take O(2**n) work.
    nodes = set()
    incoming_edges = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in nodes:
            continue

        nodes.add(node)
        incoming_edges.setdefault(node, 0)
        for parent in node.parents:
            # Do not deduplicate these edges: a repeated argument must decrement
            # the parent's count once per gradient contribution during ordering.
            incoming_edges[parent] = incoming_edges.get(parent, 0) + 1
            if parent not in nodes:
                stack.append(parent)

    # With child-to-parent edges, the output node has no incoming child edge and
    # is therefore Kahn's initial source. A parent becomes ready only after all
    # of its downstream children have already appeared in the backward order.
    ready = deque(node for node in nodes if incoming_edges[node] == 0)
    topo_sorted = []
    while ready:
        node = ready.popleft()
        topo_sorted.append(node)
        for parent in node.parents:
            incoming_edges[parent] -= 1
            if incoming_edges[parent] == 0:
                ready.append(parent)

    # Computation graphs should be acyclic. Reporting a cycle explicitly avoids
    # returning a partial order and producing incomplete or misleading gradients.
    if len(topo_sorted) != len(nodes):
        raise ValueError("The computation graph contains a cycle.")

    return topo_sorted

def finite_difference(f, x, h=1e-4):
    """
    Calculate the finite difference
    approximation to f'(x).
    """
    return (f(x + h) - f(x)) / h

def centered_difference(f, x, h=1e-4):
    """
    Calculate the centered difference
    approximation to f'(x).
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def substitution_approximation(f, x, u, v):
    """
    Calculate the substitution approximation
    to the function f'(x).
    """
    g1 = u.T @ f(v * (x + h))
    g2 = u.T @ f(v * (x - h))
    g = (g1 - g2) / h
    return g

def grad_check(f, x, h=1e-4, eps=1e-3, fn=centered_difference):
    """
    Wrapper which automates grad checking
    a function `f` at the point `x` from
    start to finish.
    """
    assert isinstance(x, ten.Tensor)
    y = f(x)
    y.backward()
    g2 = x.grad
    approx = fn(f, x, h=h)
    return abs(g2 - approx.data) < eps

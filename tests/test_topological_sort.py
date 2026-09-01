import numpy as np
import pytest

from flamethrower.autograd import Tensor
from flamethrower.autograd.utils import topological_sort


def reused_graph(depth):
    x = Tensor(np.array(1.0))
    y = x
    for _ in range(depth):
        y = y + y
    return x, y


def test_topological_sort_returns_each_reused_node_once():
    depth = 20
    _, output = reused_graph(depth)

    ordered = topological_sort(output.node)

    assert len(ordered) == depth + 1
    assert len(set(ordered)) == len(ordered)


def test_topological_sort_places_children_before_parents():
    x = Tensor(np.array(2.0))
    shared = x * x
    output = shared + shared * x

    ordered = topological_sort(output.node)
    positions = {node: index for index, node in enumerate(ordered)}

    for child in ordered:
        for parent in child.parents:
            assert positions[child] < positions[parent]


def test_backward_accumulates_duplicate_edges_after_sorting_once():
    depth = 12
    x, output = reused_graph(depth)

    output.backward()

    np.testing.assert_allclose(x.grad, 2 ** depth)


def test_topological_sort_rejects_cycles():
    class FakeNode:
        def __init__(self):
            self.parents = []

    first = FakeNode()
    second = FakeNode()
    first.parents.append(second)
    second.parents.append(first)

    with pytest.raises(ValueError, match="cycle"):
        topological_sort(first)

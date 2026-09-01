"""
Guards against the failure mode behind issues #9/#10: `tensor_library.wrap_namespace`
traces almost the entire numpy (and numpy.random) namespace automatically, but a
traced call only actually works if the underlying function has either a real
gradient in `grad_definitions` or an explicit `no_trace_primitives` entry saying
"this op is fine to treat as non-differentiable". Anything else raises
NotImplementedError the first time it's applied to a tracked Tensor - which
previously only ever showed up by accident (e.g. np.min, np.copy).

Rather than hand-maintain a list of "functions we remembered to check", this
scans every `.py` file actually shipped in the flamethrower/ package for
`tl.<name>` / `tlr.<name>` / `tl.random.<name>` references, resolves each one
back to the real numpy function it wraps, and asserts it has grad coverage.
New code that references an uncovered numpy function will fail this test
immediately instead of failing at first backward() call, potentially deep in
someone's training loop.
"""
import ast
import os

import numpy as np
import numpy.random as npr
import pytest

import flamethrower
import flamethrower.autograd.grad_defs as gd
import flamethrower.autograd.utils as utils

PACKAGE_DIR = os.path.dirname(flamethrower.__file__)

# The wrapping machinery itself references numpy but doesn't "use" it as a
# tensor_library primitive - scanning it would just find `_np.*`, which our
# alias tracking below already ignores, so nothing needs to be excluded here.


def _tracked_import_roots(tree):
    """Map each local name this file binds to 'numpy' or 'numpy.random',
    based on `import flamethrower.autograd.tensor_library[.random] as X`."""
    roots = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.name == "flamethrower.autograd.tensor_library":
                roots[alias.asname or "tensor_library"] = "numpy"
            elif alias.name == "flamethrower.autograd.tensor_library.random":
                roots[alias.asname or "random"] = "numpy.random"
    return roots


def _resolve_attribute_chain(node):
    """For an ast.Attribute node, return (root_name, [attr, attr, ...]) by
    walking down to the innermost ast.Name, or (None, None) if the base of
    the chain isn't a plain name (e.g. a call result)."""
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        attrs.reverse()
        return node.id, attrs
    return None, None


def _find_wrapped_usages(filepath):
    """Return {resolved_numpy_dotted_name: [(filepath, lineno), ...]} for every
    tl.<name>/tlr.<name>/tl.random.<name> reference in this file."""
    with open(filepath) as f:
        tree = ast.parse(f.read(), filename=filepath)

    roots = _tracked_import_roots(tree)
    if not roots:
        return {}

    usages = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        root_name, attrs = _resolve_attribute_chain(node)
        if root_name not in roots or not attrs:
            continue

        namespace = roots[root_name]
        # tl.random.<name> dereferences the nested `random` submodule that
        # tensor_library/__init__.py re-exports; everything else is a direct
        # numpy.<name> (or numpy.random.<name> for the tlr alias) reference.
        if namespace == "numpy" and attrs[0] == "random":
            if len(attrs) < 2:
                continue  # bare `tl.random` reference, not a call
            dotted = "numpy.random." + attrs[1]
        elif attrs[0] == "random" and namespace == "numpy":
            continue
        else:
            dotted = namespace + "." + attrs[0]

        usages.setdefault(dotted, []).append((filepath, node.lineno))
    return usages


def _all_wrapped_usages():
    usages = {}
    for dirpath, _dirnames, filenames in os.walk(PACKAGE_DIR):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(dirpath, filename)
            for dotted, sites in _find_wrapped_usages(filepath).items():
                usages.setdefault(dotted, []).extend(sites)
    return usages


def _resolve_function(dotted_name):
    """'numpy.sum' -> np.sum, 'numpy.random.uniform' -> npr.uniform."""
    parts = dotted_name.split(".")
    assert parts[0] == "numpy"
    module = np if len(parts) == 2 else npr
    attr = parts[-1]
    return getattr(module, attr)


def _covered(fn):
    name = utils.name(fn)
    return name in gd.grad_definitions or name in gd.no_trace_primitives


@pytest.fixture(scope="module")
def wrapped_usages():
    usages = _all_wrapped_usages()
    assert usages, "Scanner found no tl./tlr. usages - the AST walk is broken."
    return usages


def test_every_wrapped_numpy_function_used_in_the_codebase_has_grad_coverage(wrapped_usages):
    missing = []
    for dotted_name, sites in sorted(wrapped_usages.items()):
        try:
            fn = _resolve_function(dotted_name)
        except AttributeError:
            missing.append(
                "{} does not exist (referenced at {})".format(
                    dotted_name, ", ".join("{}:{}".format(f, l) for f, l in sites)
                )
            )
            continue
        if not _covered(fn):
            missing.append(
                "{} (numpy name: {!r}, referenced at {}) has neither a "
                "grad_definitions nor a no_trace_primitives entry".format(
                    dotted_name, utils.name(fn),
                    ", ".join("{}:{}".format(f, l) for f, l in sites),
                )
            )

    assert not missing, "Uncovered tensor_library primitives:\n" + "\n".join(
        "  - " + m for m in missing
    )


def test_scanner_detects_all_three_reference_shapes(tmp_path):
    # Regression check for the scanner logic itself, independent of anything
    # currently missing coverage: each reference shape used in the real
    # codebase (plain tl.<name>, tlr.<name>, and tl.random.<name>) must be
    # found, not silently skipped.
    probe_file = tmp_path / "probe.py"
    probe_file.write_text(
        "import flamethrower.autograd.tensor_library as tl\n"
        "import flamethrower.autograd.tensor_library.random as tlr\n"
        "def f(x):\n"
        "    a = tl.gcd(x, x)\n"          # plain tl.<name>
        "    b = tlr.dirichlet(x)\n"      # tlr.<name>
        "    c = tl.random.laplace(x)\n"  # tl.random.<name>
        "    return a, b, c\n"
    )

    usages = _find_wrapped_usages(str(probe_file))

    assert set(usages) == {"numpy.gcd", "numpy.random.dirichlet", "numpy.random.laplace"}
    for dotted, sites in usages.items():
        assert sites == [(str(probe_file), 4 if dotted == "numpy.gcd" else
                           5 if dotted == "numpy.random.dirichlet" else 6)]

from __future__ import absolute_import

def find_top_boxed_args(args):
    top_trace = -1
    top_vars = []
    top_node_type = None
    for argnum, arg in enumerate(args):
        if hasattr(arg, '_trace'):
            print("Arg is {}".format(arg))
            trace = arg._trace
            print("Trace is {}".format(trace))
            if trace > top_trace:
                top_vars = [(argnum, arg)]
                top_trace = trace
                top_node_type = type(arg._node)
            elif trace == top_trace:
                top_vars.append((argnum, arg))
    return top_vars, top_trace, top_node_type

def vals(args, var_args):
    args = list(args)
    for i, v in var_args:
        args[i] = v._value
    return tuple(args)

def primitive(f):
    def f_wrapped(*args, **kwargs):
        print("Args are: {}".format(args))
        # var_args, trace, node_type = find_top_boxed_args(args)
        var_args, trace, node_type = args, arg[0]._trace, type(args[0]._node)
        print("Var args are: {}".format(var_args))
        if var_args:
            argvals = vals(args, var_args)
            parents = [var[1]._node for var in var_args]
            argnums = [var[0] for var in var_args]
            print("Argvals: {}, Parents: {}, Argnums: {}".format(argvals, parents, argnums))
            print("KWaRGS: {}".format(kwargs))
            print("fn: {}".format(f.__name__))
            ans = f_wrapped(*argvals, **kwargs)
            print("Ans: {}")
            node = GradNode(ans, f_wrapped, argvals, kwargs, argnums, parents)
            return Tensor(ans, trace, node)
        else:
            return f(*args, **kwargs)
    try:
        f_wrapped.__name__ = f.__name__
    except:
        pass
    f_wrapped.fn = f
    f_wrapped._is_primitive = True
    return f_wrapped

def notrace_primitive(f_raw):
    def f_wrapped(*args, **kwargs):
        argvals = map(getval, args)
        return f_raw(*argvals, **kwargs)
    return f_wrapped

def wrap_intdtype(cls):
    class IntdtypeSubclass(cls):
        __new__ = notrace_primitive(cls.__new__)
    return IntdtypeSubclass

notrace_functions = [
    _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type
]
def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int, _np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    for name, obj in old.items():
        if obj in notrace_functions:
            new[name] = primitive(obj)
        elif callable(obj) and type(obj) is not type:
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = primitive(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(_np.__dict__, globals())


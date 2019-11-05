from call import Primitive
import numpy as _np

notrace_functions = [
    _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type
]

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int, _np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    for name, obj in old.items():
        if obj in notrace_functions:
            new[name] = Primitive(obj)
        elif callable(obj) and type(obj) is not type:
            new[name] = Primitive(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(_np.__dict__, globals())

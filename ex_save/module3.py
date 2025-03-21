import importlib.util
import sys
import types

a = 5

def g(f):
    module_name = "temp_module"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    temp_module = importlib.util.module_from_spec(spec)
    d = {}
    s = {'f': f}
    # temp_module.f = f
    exec('x = f()', temp_module.__dict__)
    return d, temp_module.__dict__
    
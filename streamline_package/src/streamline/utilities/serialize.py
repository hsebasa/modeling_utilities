import dis


def get_global_dependencies(func):
    """
    Returns a set of global variable names accessed by the function.
    """
    dependencies = set()
    for instr in dis.get_instructions(func):
        if instr.opname == 'LOAD_GLOBAL':
            dependencies.add(instr.argval)
    return dependencies


def mainify(obj):
    """
    https://stackoverflow.com/questions/52402783/pickle-class-definition-in-module-with-dill
    """
    import __main__
    import inspect
    import ast

    s = inspect.getsource(obj)
    m = ast.parse(s)
    co = compile(m, '<string>', 'exec')
    exec(co, __main__.__dict__)

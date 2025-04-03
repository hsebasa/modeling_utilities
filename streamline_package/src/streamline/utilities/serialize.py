import dis


def get_global_dependencies(func):
    """
    Get global variable names accessed by a function.
    
    This function analyzes the bytecode of a function to identify all
    global variables that are accessed by the function.
    
    Parameters
    ----------
    func : callable
        The function to analyze
        
    Returns
    -------
    set
        A set of global variable names accessed by the function
    """
    dependencies = set()
    for instr in dis.get_instructions(func):
        if instr.opname == 'LOAD_GLOBAL':
            dependencies.add(instr.argval)
    return dependencies


def mainify(obj):
    """
    Make an object available in the __main__ module.
    
    This function takes an object (typically a class), gets its source code,
    and executes it in the __main__ module's namespace. This is useful for
    making classes picklable when they are defined in modules that might not
    be importable during unpickling.
    
    Parameters
    ----------
    obj : object
        The object to make available in __main__
        
    Notes
    -----
    This technique is based on the solution from:
    https://stackoverflow.com/questions/52402783/pickle-class-definition-in-module-with-dill
    """
    import __main__
    import inspect
    import ast

    s = inspect.getsource(obj)
    m = ast.parse(s)
    co = compile(m, '<string>', 'exec')
    exec(co, __main__.__dict__)

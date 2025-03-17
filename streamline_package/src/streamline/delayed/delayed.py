from copy import copy
from typing import *


class _Library:
    def __init__(self, globals: Dict=None):
        if globals is None:
            globals = dict()
        else:
            globals = copy(globals)
        for c in ['min', 'max', 'type', 'str', 'float', 'int', 'round']:
            if c not in globals:
                globals[c] = __builtins__[c]
        
        self._globals = globals
        self._delayed = dict()

    @property
    def globals(self):
        return self._globals
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            assert name in self.globals
            return Delayed(prefix=name)
    
    def delayed(self, name: str):
        # Decorator for functions
        def fun(f: Callable):
            assert callable(f)
            self._globals[name] = f
            def g(*a, **kw):
                return Delayed(prefix=f'{name}(*{a}, **{kw})')
            return g
        return fun


class Delayed:
    def __init__(self, prefix=None):
        self._prefix = prefix
    
    def __repr__(self):
        return self._prefix
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            prefix = self._prefix
            if prefix is None:
                prefix = name
            else:
                prefix = prefix+'.'+name

            return Delayed(prefix=prefix)

    def __call__(self, *args, **kwargs):
        return Delayed(prefix=f'{repr(self)}(*{args}, **{kwargs})')

    def __add__(self, other):
        return Delayed(prefix=f'{repr(self)}+{repr(other)}')

    def __lt__(self, other):
        return Delayed(prefix=f'{repr(self)}<{repr(other)}')

    def __le__(self, other):
        return Delayed(prefix=f'{repr(self)}<={repr(other)}')

    def __gt__(self, other):
        return Delayed(prefix=f'{repr(self)}>{repr(other)}')

    def __ge__(self, other):
        return Delayed(prefix=f'{repr(self)}>={repr(other)}')

    def __or__(self, other):
        return Delayed(prefix=f'({repr(self)} | {repr(other)})')

    def __and__(self, other):
        return Delayed(prefix=f'({repr(self)} & {repr(other)})')

    def __eq__(self, other):
        return Delayed(prefix=f'({repr(self)} == {repr(other)})')

    def contains(self, other):
        return Delayed(prefix=f'({repr(other)} in {repr(self)})')

    def isin(self, other):
        return Delayed(prefix=f'({repr(self)} in {repr(other)})')

    def __not__(self):
        return Delayed(prefix=f'~({repr(self)})')


def eval_delay(sel: Delayed, env: Dict=None):
    if env is None:
        env = dict()

    v = eval(repr(sel), env)
    return v


step = Delayed('step')
delay_lib = _Library()

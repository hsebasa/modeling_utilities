from streamline import __version__
from typing import Optional, Callable, Dict, Set, List, Union, Tuple
import streamline as sl
import inspect


class Var:
    def __init__(self, name: str):
        assert type(name) is str
        self._name = name
    @property
    def name(self):
        return self._name
    def __repr__(self):
        return f'Var({repr(self._name)})'
    def __str__(self):
        return self._name
    def __eq__(self, other):
        if type(other) is Var:
            return self._name == other._name
        return False
    def __hash__(self):
        return hash(self._name)


class _Step:
    __version__ = __version__
    __stepname__ = 'Step'
    def __init__(
            self,
            args: Optional[List[str]]=None,
            kw: Optional[Dict]=None,
            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        if args is None:
            args = list()
        if type(args) is str or type(args) is Var:
            # if a single string or Var is passed, convert to list
            args = [args]
        # assert all elements are strings or Var
        assert all([type(a) is str or type(a) is Var for a in args]), 'All args must be strings or Var'
        assert type(args) is list
        self._args = args
        
        if kw is None:
            kw = dict()
        assert type(kw) is dict
        self._kw = kw

        if arg_cat is None:
            arg_cat = ''
        assert type(arg_cat) is str
        self._arg_cat = arg_cat
        
        if tags is None:
            tags = set()
        assert type(tags) is set
        self._tags = tags
    
    def __repr__(self):
        if self.tags is not None:
            return f'{self.__stepname__}(arg_cat={repr(self.arg_cat)}, tags={self.tags})'
        else:
            return f'{self.__stepname__}(arg_cat={repr(self.arg_cat)})'
    
    def __sourcecode__(self, create_call: Optional[bool]=False):
        assert not create_call, 'Not implemented'
        return inspect.getsource(self._fun)
    
    @property
    def arg_cat(self):
        return self._arg_cat
        
    @property
    def kw(self):
        return self._kw
        
    @property
    def fun(self):
        return self._fun
        
    @property
    def tags(self):
        return self._tags
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Subclasses must implement __call__ method')


class Function(_Step):
    __stepname__ = 'Function'
    def __init__(
            self,

            fun: Callable,
            args: Optional[List[str]]=None,
            kw: Optional[Dict]=None,

            out_var: Optional[Union[str, Tuple[str]]]=None,

            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        assert callable(fun)
        self._fun = fun
        if tags is None:
            tags = set()
        super().__init__(args=args, kw=kw, arg_cat=arg_cat, tags={'function'}|tags)
        assert out_var is None or type(out_var) is str or ((type(out_var) is tuple or type(out_var) is list) and all((type(a) is str for a in out_var)))

        self._out_var = out_var

    def __call__(self, env, kw: Dict):
        def cvar(v):
            if isinstance(v, Var):
                return env[v.name]
            return v
        args = [cvar(a) for a in self._args]
        kwargs = self._kw | kw
        kwargs = {a: cvar(b) for a, b in kwargs.items()}
        out = self._fun(*args, **kwargs)

        out_var = self._out_var
        if out_var is None:
            out_var = '_'
            
        if type(out_var) is str:
            env[out_var] = out
        else:
            assert len(out) == len(out_var)
            for v, o in zip(out_var, out):
                env[v] = o
        
        return out


class Delete(_Step):
    __stepname__ = 'Delete'
    def __init__(
            self,

            args: Optional[Union[str, List[str]]]=None,

            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        if type(args) is str:
            args = [args]
        if tags is None:
            tags = set()
        super().__init__(args=args, arg_cat=arg_cat, tags={'delete'}|tags)
        assert all([type(a) is str for a in args]), 'All args must be strings'

    def __call__(self, env, kw: Dict):
        for a in self._args:
            if a in env:
                del env[a]
        return None
    
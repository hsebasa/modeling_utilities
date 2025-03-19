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
    __steptype__ = 'Step'
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
        assert all([type(a) is str or isinstance(a, Var) for a in args]), 'All args must be strings or Var'
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
            return f'{self.__steptype__}(arg_cat={repr(self.arg_cat)}, tags={self.tags})'
        else:
            return f'{self.__steptype__}(arg_cat={repr(self.arg_cat)})'
    
    def __sourcecode__(self, create_call: Optional[bool]=False):
        assert not create_call, 'Not implemented'
        return inspect.getsource(self._fun)
    
    def get_dependencies(self):
        """
        Get the dependencies of the step. This is a list of variable names
        that are used in the step's args and kw.
        :return: A list of variable names.
        :rtype: list
        """
        dependencies = set()
        for a in self._args:
            if isinstance(a, Var):
                dependencies.add(a.name)
        for k, v in self._kw.items():
            if isinstance(v, Var):
                dependencies.add(v.name)
        return list(dependencies)

    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename variables in the step's args and kw based on the provided mapping.
        This is useful for updating variable names in a pipeline step when the
        variable names in the environment have changed.

        :param variables: A dictionary mapping old variable names to new variable names.
        :type variables: dict

        :param arg_cat: The new argument category for the step.
        :type arg_cat: str

        :return: self, for method chaining.
        :rtype: _Step
        :raises ValueError: If any of the args or kw are not strings or Var instances.
        :raises AssertionError: If the args or kw are not of the expected types.
        """
        if variables is not None:
            assert type(variables) is dict or callable, 'variables must be a dict or callable'
            if type(variables) is dict:
                fun = lambda x: variables.get(x, x)
            else:
                fun = variables
            self._args = [Var(fun(a.name)) if isinstance(a, Var) else a for a in self._args]
            self._kw = {
                k: Var(fun(v.name)) if isinstance(v, Var) else v
                for k, v in self._kw.items()
            }
        if arg_cat is not None:
            assert type(arg_cat) is dict or callable or str, 'arg_cat must be a dict or callable'
            if type(arg_cat) is dict:
                arg_cat = lambda x: arg_cat.get(x, x)
            elif type(arg_cat) is str:
                arg_cat = lambda x: arg_cat
            self._arg_cat = arg_cat(self._arg_cat)
        return self
    
    @property
    def arg_cat(self):
        return self._arg_cat
        
    @property
    def args(self):
        return self._args
        
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
    __steptype__ = 'Function'
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
        super().__init__(args=args, kw=kw, arg_cat=arg_cat, tags=sl.tags.STEP_FUNCTION|tags)
        assert out_var is None or type(out_var) is str or ((type(out_var) is tuple or type(out_var) is list) and all((type(a) is str for a in out_var)))

        self._out_var = out_var

    @property
    def out_var(self):
        return self._out_var

    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename parameters in the function call.
        """
        super().rename(variables=variables, arg_cat=arg_cat)
        if variables is not None:
            assert type(variables) is dict or callable, 'variables must be a dict or callable'
            if type(variables) is dict:
                fun = lambda x: variables.get(x, x)
            else:
                fun = variables
            if isinstance(self._out_var, str):
                self._out_var = fun(self._out_var)
            elif isinstance(self._out_var, (list, tuple)):
                self._out_var = tuple(fun(v) for v in self._out_var)
            elif self._out_var is None:
                pass
            else:
                raise ValueError('out_var must be str, list or tuple')
        return self

    def __call__(self, env, kw: Dict):
        def cvar(v):
            if isinstance(v, Var):
                assert v.name in env, f'Variable {v.name} not found in environment'
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
    __steptype__ = 'Delete'
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
        super().__init__(args=args, arg_cat=arg_cat, tags=sl.tags.STEP_DELETE|tags)
        assert all([type(a) is str or isinstance(a, Var) for a in args]), 'All args must be strings'

    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename parameters in the function call.
        """
        super().rename(variables=variables, arg_cat=arg_cat)
        if variables is not None:
            assert type(variables) is dict or callable, 'variables must be a dict or callable'
            if type(variables) is dict:
                fun = lambda x: variables.get(x, x)
            else:
                fun = variables
            self._args = [fun(k.name) if isinstance(k, Var) else fun(k) for k in self._args]
        return self
    
    def __call__(self, env, kw: Dict):
        for a in self._args:
            if isinstance(a, Var):
                a = a.name
            if a in env:
                del env[a]
        return None
    

class VariablesDict(_Step):
    __steptype__ = 'Function'
    def __init__(
            self,
            
            kw: Optional[Dict]=None,

            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        if tags is None:
            tags = set()
        super().__init__(kw=kw, arg_cat=arg_cat, tags=sl.tags.STEP_VARIABLES_DICT|tags)

    def __call__(self, env, kw: Dict):
        def cvar(v):
            if isinstance(v, Var):
                return env[v.name]
            return v
        kwargs = self._kw | kw
        kwargs = {a: cvar(b) for a, b in kwargs.items()}

        for a, b in kwargs.items():
            env[a] = cvar(b)
        return None

    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename parameters in the function call.
        """
        super().rename(variables=variables, arg_cat=arg_cat)
        if variables is not None:
            assert type(variables) is dict or callable, 'variables must be a dict or callable'
            if type(variables) is dict:
                fun = lambda x: variables.get(x, x)
            else:
                fun = variables
            self._kw = {fun(k): v for k, v in self._kw.items()}
        return self

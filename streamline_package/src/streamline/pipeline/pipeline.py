from streamline import __version__
from streamline.functions import Function
import streamline as sl
from .step import Step

from typing import List, Dict, Optional, Tuple, Union, Callable, Set
from importlib import import_module
from copy import deepcopy
import inspect


class StepNotFound(Exception):
    pass


class _Vertical:
    def __init__(self, filt: List):
        self._filt = filt

    @property
    def filt(self):
        return self._filt

    def __repr__(self):
        return repr(self._filt)

    def __len__(self):
        return len(self._filt)
        
    def __iter__(self):
        return iter(self._filt)

        
class _Pipeline:
    __version__ = __version__
    def __init__(
            self,
            a: Optional[List[Step]]=None,
        ):
        if a is None:
            a = list()
        else:
            assert isinstance(a, list) or hasattr(a, 'to_list')
            if hasattr(a, 'to_list'):
                a = a.to_list()
                kwargs = a.kwargs | kwargs
            assert all((isinstance(step, Step) for step in a))
        self.__steps = a

    def print(self, show_args: Optional[bool]=False):
        res = 'Pipeline(steps=['
        if show_args:
            res = res + ', '.join([f'\n  {step}' for step in self.__steps])
        else:
            res = res + ', '.join([f'\n  "Step {i}"' for i, _ in enumerate(self.__steps)])
                        
        res = res + "\n], "
        res += ')'
        return res

    def __repr__(self):
        return self.print(show_args=True)

    def items(self):
        return iter(self.__steps)
        
    def to_list(self):
        return list(self.items())

    def _append(self, step: Step):
        assert isinstance(step, Step)
        self.__steps.append(step)
    
    def apply(self, a, globals_=None):
        if isinstance(a, sl.Delayed):
            if globals_ is None:
                globals_ = dict()
            globals_ = sl.delay_lib.globals|globals_
            return _Vertical([
                sl.eval_delay(
                    sel=a,
                    env=globals_|{'step': step, 'step_index': i}
                )
                for i, step in enumerate(self.__steps)
            ])
        else:
            assert callable(a)
            return _Vertical([a(step) for step in self.__steps])
        
    def _getitem(self, index):
        if type(index) is _Vertical:
            assert len(index) == len(self.__steps)
            res = []
            for i, a in enumerate(index):
                if a:
                    res.append(self.__steps[i])
            return res
        elif type(index) is list:
            return [self.__steps[i] for i in index]
        else:
            assert type(index) is int
            if index < 0 or index >= len(self.__steps):
                raise StepNotFound(f"Step index {index} out of range")
            return self.__steps[index]
    
    def _delitem(self, index):
        if type(index) is _Vertical:
            assert len(index) == len(self.__steps)
            res = []
            new_ord = []
            for i, a in enumerate(index):
                if a:
                    del self.__steps[i]
        elif type(index) is list:
            for n in index:
                del self.__steps[n]
        else:
            assert type(index) is int
            if index < 0 or index >= len(self.__steps):
                raise StepNotFound(f"Step index {index} out of range")
            del self.__steps[index]
        
    def _setitem(self, index: int, step: Step):
        assert isinstance(step, Step)
        if type(index) is _Vertical:
            assert len(index) == len(self.__steps)
            res = []
            new_ord = []
            for i, a in enumerate(index):
                if a:
                    self.__steps[i] = step
        elif type(index) is list:
            for n in index:
                self.__steps[n] = step
        else:
            raise NotImplementedError
    
    def _insert(self, index: int, step: Step):
        assert type(index) is int
        self.__steps.insert(index, step)

    def _add_step(
            self,
            step: Step,
            index: Optional[int]=None,
        ):
        if index is None:
            self._append(step)
        else:
            self._insert(index, step)
        return self


class _Loc:
    def __init__(self, pipe):
        self.__pipe = pipe
    
    def _stdize_index(self, obj, throw_error: Optional[bool]=True):
        assert type(obj) is int or type(obj) is list or isinstance(obj, sl.Delayed) or isinstance(obj, slice) or isinstance(obj, _Vertical), type(obj)
        if isinstance(obj, sl.Delayed):
            return self.__pipe.apply(obj)
        else:
            return obj
    
    def __getitem__(self, index):
        index = self._stdize_index(index)
        steps = self.__pipe._getitem(index)
        if isinstance(steps, Step):
            steps = [steps]
        return Pipeline(a=steps)

    def __delitem__(self, index):
        index = self._stdize_index(index)
        self.__pipe._delitem(index)

    def __setitem__(self, index: int, step: Step):
        index = self._stdize_index(index)
        self.__pipe._setitem(index, step)


class Pipeline(_Pipeline):
    def __init__(
            self,
            a: Optional[List[Tuple]]=None,
        ):
        super().__init__(a=a)
        self._loc = _Loc(self)

    def __copy__(self):
        return Pipeline(
            a=copy(self.__steps)
        )

    def __deepcopy__(self):
        return Pipeline(
            a=deepcopy(self.__steps)
        )

    def copy(self, deep: Optional[bool]=False):
        if deep:
            deepcopy(self)
        else:
            copy(self)
        
    @property
    def loc(self):
        return self._loc

    def __add__(self, other):
        assert isinstance(other, Pipeline)
        return concat([self, other])
        
    def add_step(
            self,
            a: Union[Step, Callable],
            index: Optional[int]=None,
            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
            kw: Optional[Dict]=None,
        ):
        if isinstance(a, Step):
            assert arg_cat is None and tags is None
            step = a
        else:
            step = Step(fun=a, arg_cat=arg_cat, tags=tags, kw=kw)
        self._add_step(index=index, step=step)
        return self
    
    def add_import_lib(
            self,
            a: Union[Dict, str],
            alias: Optional[str]=None,
            index: Optional[int]=None,
            tags: Optional[Set[str]]=None,
        ):
        if tags is None:
            tags = set()
            
        if type(a) is str:
            if alias is None:
                alias = name
            a = {a: alias}
        else:
            assert type(a) is dict
            assert alias is None

        names = []
        aliases = []
        for c in a.items():
            names.append(c[0])
            aliases.append(c[1])
        self.add_step(
            a=Function(lambda : [import_module(n) for n in names], out_var=aliases),
            index=index,
            tags={'import_lib'}|tags,
        )
        return self

    def run(self, env=None, kw: Optional[Dict]=None):
        if env is None or type(env) is dict:
            env = sl.RunEnv(env=env)
        if kw is None:
            kw = dict()
        
        for step in self.__steps:
            arg_cat = step.arg_cat
            if arg_cat is None:
                kw_filt = dict()
            else:
                kw_filt = {
                    a[len(arg_cat)+1:]: b
                    for a, b in kw.items()
                    if a.startswith(arg_cat+'_')
                }
            kw_new = step.kw|kw_filt
            step(env=env, kw=kw_new)
            env._add_step(step=step, kwargs=kw_filt)
        return env

    def run_parallel(self, env_l: List=None, kw_l: Optional[List[Dict]]=None):
        import ray
        if env_l is None:
            m = 0
            env_l = [sl.RunEnv()]
        elif type(env_l) is sl.RunEnv:
            m = 0
            env_l = [env_l]
        else:
            assert type(env_l) is list
            m = len(env_l)
        
        if kw_l is None:
            n = 0
            kw_l = list()
        elif type(kw_l) is dict:
            n = 0
            kw_l = [kw_l]
        else:
            assert type(kw_l) is list
            n = len(kw_l)

        self_run = ray.remote(lambda *args: self.run(*args))
        exec_l = []
        for env in env_l:
            for kw in kw_l:
                res = self_run.remote(env, kw)
                exec_l.append(res)
        res_l = ray.get(exec_l)
        
        if m == 0 or n == 0:
            return res_l
        else:
            return [res_l[i:i+n] for i in range(0, m*n, n)]


def run_parallel(runs: List[Tuple]):
    import ray
    exec_l = []
    for pipe, env, kw in runs:
        pipe_run = ray.remote(lambda *args: pipe.run(*args))
        res = pipe_run.remote(env, kw)
        exec_l.append(res)
    res_l = ray.get(exec_l)
    return res_l
    

def concat(list_pipes: List):
    steps = []
    kwargs = dict()
    for pipe in list_pipes:
        steps.extend(pipe._Pipeline__steps)
    return Pipeline(a=steps)

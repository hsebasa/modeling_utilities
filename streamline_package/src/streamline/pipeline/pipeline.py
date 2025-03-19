# Copyright (c) 2025, streamline
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.
# -*- coding: utf-8 -*-

from streamline import __version__
import streamline as sl
from .step import _Step, Function, Delete, VariablesDict

from typing import List, Dict, Optional, Tuple, Union, Callable, Set
from importlib import import_module
from copy import copy, deepcopy


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
            a: Optional[List[_Step]]=None,
        ):
        if a is None:
            a = list()
        else:
            assert isinstance(a, list) or hasattr(a, 'to_list')
            if hasattr(a, 'to_list'):
                a = a.to_list()
            assert all((isinstance(step, _Step) for step in a))
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

    def __len__(self):
        return len(self.__steps)
    
    def __iter__(self):
        return iter(self.__steps)
    
    def get_env_info(self):
        """
        Get all dependencies for all steps in the pipeline.
        """
        res = list()
        defs = set()
        accum_deps = set()
        for step in self.__steps:
            new_defs = set()
            rm_defs = set()
            if type(step) is Function:
                if step.out_var is not None:
                    if isinstance(step.out_var, str):
                        new_defs = set([step.out_var])
                    else:
                        new_defs = set(step.out_var)
            elif type(step) is Delete:
                if step.args is not None:
                    rm_defs = set([a for a in step.args])
            elif type(step) is VariablesDict:
                if step.kw is not None:
                    new_defs = set([a for a, _ in step.kw.items()])
                    
            new_deps = set(step.get_dependencies())

            accum_deps = accum_deps | new_deps
            res.append({
                'required_vars': new_deps,
                'unresolved_vars': new_deps - defs,
                'unresolved_vars_accum': accum_deps,
                'env_vars': copy(defs),
                'removed_vars': rm_defs,
                'added_vars': new_defs,
                'steptype': step.__steptype__,
                'arg_cat': step.arg_cat,
                'tags': step.tags,
            })
            defs.update(new_defs)
            defs -= rm_defs
        return res

    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename parameters in the function call for all steps in the pipeline.
        """
        for step in self.__steps:
            step.rename(variables=variables, arg_cat=arg_cat)
        return self

    def _append(self, step: _Step):
        assert isinstance(step, _Step)
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
        elif isinstance(index, slice):
            return self.__steps[index]
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
        elif isinstance(index, slice):
            del self.__steps[index]
        else:
            assert type(index) is int
            if index < 0 or index >= len(self.__steps):
                raise StepNotFound(f"Step index {index} out of range")
            del self.__steps[index]
        
    def _setitem(self, index: int, step: _Step):
        assert isinstance(step, _Step)
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
        elif isinstance(index, slice):
            self.__steps[index] = step
        else:
            raise NotImplementedError
    
    def _insert(self, index: int, step: _Step):
        assert type(index) is int
        self.__steps.insert(index, step)

    def _add_step(
            self,
            step: _Step,
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
        if isinstance(steps, _Step):
            steps = [steps]
        return Pipeline(a=steps)

    def __delitem__(self, index):
        index = self._stdize_index(index)
        self.__pipe._delitem(index)

    def __setitem__(self, index: int, step: _Step):
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
            a=copy([copy(step) for step in self.__steps])
        )

    def __deepcopy__(self, memo):
        return Pipeline(
            a=deepcopy(self.__steps)
        )

    def copy(self, deep: Optional[bool]=False):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)
        
    @property
    def loc(self):
        return self._loc

    def __add__(self, other):
        assert isinstance(other, Pipeline)
        return concat([self, other])
        
    def add_step(
            self,
            step: Union[Function, Delete, VariablesDict],
            index: Optional[int]=None,
        ):
        assert isinstance(step, (Function, Delete, VariablesDict)), type(step)
        self._add_step(index=index, step=step)
        return self
        
    def add_function(
            self,
            a: Union[Function, Callable],
            index: Optional[int]=None,

            args: Optional[List[str]]=None,
            kw: Optional[Dict]=None,

            out_var: Optional[Union[str, Tuple[str]]]=None,

            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        assert isinstance(a, (Function, Callable)), type(a)
        if isinstance(a, Function):
            assert arg_cat is None and tags is None and out_var is None
            assert args is None and kw is None
            step = a
        else:
            step = Function(fun=a, args=args, kw=kw, out_var=out_var, arg_cat=arg_cat, tags=tags)
        self._add_step(index=index, step=step)
        return self
        
    def add_delete(
            self,
            a: Union[Delete, str, List[str]],
            index: Optional[int]=None,
            
            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        assert isinstance(a, (Delete, str, list)), type(a)
        if isinstance(a, Delete):
            assert arg_cat is None and tags is None
            step = a
        else:
            step = Delete(args=a, arg_cat=arg_cat, tags=tags)
        self._add_step(index=index, step=step)
        return self
        
    def add_variables_dict(
            self,
            a: Union[VariablesDict, Dict],
            index: Optional[int]=None,

            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        assert isinstance(a, (VariablesDict, Dict)), type(a)
        if isinstance(a, VariablesDict):
            assert arg_cat is None and tags is None
            step = a
        else:
            step = VariablesDict(a, arg_cat=arg_cat, tags=tags)
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
                alias = a
            a = {a: alias}
        else:
            assert type(a) is dict
            assert alias is None

        names = []
        aliases = []
        for c in a.items():
            names.append(c[0])
            aliases.append(c[1])
        self.add_function(
            a=Function(
                lambda : [import_module(n) for n in names], out_var=aliases,
                tags=sl.tags.STEP_IMPORT_LIB|tags,
            ),
            index=index,
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

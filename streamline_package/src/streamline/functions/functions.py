from typing import Callable, Optional, List, Union, Tuple, Dict
import streamline as sl


class Function:
    def __init__(
            self,
            fun: Callable,
            in_args: Optional[List[str]]=None,
            in_kw: Optional[Dict]=None,
            in_vars: Optional[List[str]]=None,
            in_vars_kw: Optional[Dict]=None,
            out_var: Optional[Union[str, Tuple[str]]]=None,
        ):
        assert out_var is None or type(out_var) is str or ((type(out_var) is tuple or type(out_var) is list) and all((type(a) is str for a in out_var)))
        self._fun = fun
        
        if in_args is None:
            in_args = []
        assert type(in_args) is list
        self._in_args = in_args

        if in_kw is None:
            in_kw = dict()
        self._in_kw = in_kw
        
        if in_vars is None:
            in_vars = []
        self._in_vars = in_vars

        if in_vars_kw is None:
            in_vars_kw = dict()
        self._in_vars_kw = in_vars_kw
        self._out_var = out_var

    def __sourcecode__(self, create_call: Optional[bool]=False):
        assert not create_call, 'Not implemented'
        return inspect.getsource(self._fun)

    def __call__(self, env, kw: Dict):
        args = [env[a] for a in self._in_vars]
        kwargs = {a: env[b] for a, b in self._in_vars_kw.items()}
        kwargs = self._in_kw | kwargs | kw
        out = self._fun(*self._in_args, *args, **kwargs)

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

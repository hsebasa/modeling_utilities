from streamline import __version__
from typing import Optional, Callable, Dict, Set


class Step:
    __version__ = __version__
    def __init__(
            self,
            fun: Callable,
            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
            kw: Optional[Dict]=None,
        ):
        assert type(arg_cat) is str or arg_cat is None
        assert callable(fun)
        
        if arg_cat is None:
            arg_cat = ''
        self._arg_cat = arg_cat
        
        if tags is None:
            tags = set()
        self._tags = tags
        
        if kw is None:
            kw = dict()
        self._kw = kw
        
        self._fun = fun
    
    def __repr__(self):
        if self.tags is not None:
            return f'Step(arg_cat={repr(self.arg_cat)}, tags={self.tags}, fun={self.fun})'
        else:
            return f'Step(arg_cat={repr(self.arg_cat)}, fun={self.fun})'
    
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
        return self._fun(*args, **kwargs)

from copy import copy
from typing import *


class _Library:
    """
    A library of globals and delayed functions.
    
    This class provides access to global functions and variables in a delayed
    execution context, allowing for the creation of expression templates that
    can be evaluated later.
    """
    def __init__(self, globals: Dict=None):
        """
        Initialize a _Library.
        
        Parameters
        ----------
        globals : Dict, default=None
            Dictionary of global variables and functions
        """
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
        """
        Get the globals dictionary.
        
        Returns
        -------
        dict
            Dictionary of global variables and functions
        """
        return self._globals
    
    def __getattr__(self, name):
        """
        Get a Delayed object for a global.
        
        This method provides attribute access to globals, returning a Delayed
        object that can be used to create expression templates.
        
        Parameters
        ----------
        name : str
            The name of the global variable or function
            
        Returns
        -------
        Delayed
            A Delayed object representing the global
            
        Raises
        ------
        AssertionError
            If the name is not in the globals dictionary
        """
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            assert name in self.globals
            return Delayed(prefix=name)
    
    def delayed(self, name: str):
        """
        Decorator for creating delayed functions.
        
        Parameters
        ----------
        name : str
            The name to use for the function in the globals dictionary
            
        Returns
        -------
        callable
            A decorator function
        """
        # Decorator for functions
        def fun(f: Callable):
            assert callable(f)
            self._globals[name] = f
            def g(*a, **kw):
                return Delayed(prefix=f'{name}(*{a}, **{kw})')
            return g
        return fun


class Delayed:
    """
    A class for delayed execution of expressions.
    
    This class allows for the creation of expression templates that can be
    evaluated later. It supports most common operators and method calls,
    which are captured as string representations that can be evaluated
    with eval() at a later time.
    """
    def __init__(self, prefix=None):
        """
        Initialize a Delayed object.
        
        Parameters
        ----------
        prefix : str, default=None
            The prefix string representation of the expression
        """
        self._prefix = prefix
    
    def __repr__(self):
        """
        Get string representation of the Delayed object.
        
        Returns
        -------
        str
            String representation of the expression
        """
        return self._prefix
    
    def __getattr__(self, name):
        """
        Get an attribute of the Delayed object.
        
        This method captures attribute access as part of the delayed
        expression template.
        
        Parameters
        ----------
        name : str
            The name of the attribute
            
        Returns
        -------
        Delayed
            A new Delayed object representing the attribute access
        """
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
        """
        Call the Delayed object.
        
        This method captures function calls as part of the delayed
        expression template.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments for the call
        **kwargs : dict
            Keyword arguments for the call
            
        Returns
        -------
        Delayed
            A new Delayed object representing the function call
        """
        return Delayed(prefix=f'{repr(self)}(*{args}, **{kwargs})')

    def __add__(self, other):
        """
        Add the Delayed object to another object.
        
        Parameters
        ----------
        other : object
            The object to add
            
        Returns
        -------
        Delayed
            A new Delayed object representing the addition operation
        """
        return Delayed(prefix=f'({repr(self)}+{repr(other)})')

    def __lt__(self, other):
        """
        Compare the Delayed object with another object (less than).
        
        Parameters
        ----------
        other : object
            The object to compare with
            
        Returns
        -------
        Delayed
            A new Delayed object representing the comparison operation
        """
        return Delayed(prefix=f'({repr(self)}<{repr(other)})')

    def __le__(self, other):
        """
        Compare the Delayed object with another object (less than or equal).
        
        Parameters
        ----------
        other : object
            The object to compare with
            
        Returns
        -------
        Delayed
            A new Delayed object representing the comparison operation
        """
        return Delayed(prefix=f'({repr(self)}<={repr(other)})')

    def __gt__(self, other):
        """
        Compare the Delayed object with another object (greater than).
        
        Parameters
        ----------
        other : object
            The object to compare with
            
        Returns
        -------
        Delayed
            A new Delayed object representing the comparison operation
        """
        return Delayed(prefix=f'({repr(self)}>{repr(other)})')

    def __ge__(self, other):
        """
        Compare the Delayed object with another object (greater than or equal).
        
        Parameters
        ----------
        other : object
            The object to compare with
            
        Returns
        -------
        Delayed
            A new Delayed object representing the comparison operation
        """
        return Delayed(prefix=f'({repr(self)}>={repr(other)})')

    def __or__(self, other):
        """
        Perform a logical OR operation with another object.
        
        Parameters
        ----------
        other : object
            The object to OR with
            
        Returns
        -------
        Delayed
            A new Delayed object representing the OR operation
        """
        return Delayed(prefix=f'({repr(self)} | {repr(other)})')

    def __and__(self, other):
        """
        Perform a logical AND operation with another object.
        
        Parameters
        ----------
        other : object
            The object to AND with
            
        Returns
        -------
        Delayed
            A new Delayed object representing the AND operation
        """
        return Delayed(prefix=f'({repr(self)} & {repr(other)})')

    def __eq__(self, other):
        """
        Compare the Delayed object with another object for equality.
        
        Parameters
        ----------
        other : object
            The object to compare with
            
        Returns
        -------
        Delayed
            A new Delayed object representing the equality comparison
        """
        return Delayed(prefix=f'({repr(self)} == {repr(other)})')

    def contains(self, other):
        """
        Check if the Delayed object contains another object.
        
        Parameters
        ----------
        other : object
            The object to check for
            
        Returns
        -------
        Delayed
            A new Delayed object representing the contains check
        """
        return Delayed(prefix=f'({repr(other)} in {repr(self)})')

    def isin(self, other):
        """
        Check if the Delayed object is in another object.
        
        Parameters
        ----------
        other : object
            The object to check in
            
        Returns
        -------
        Delayed
            A new Delayed object representing the isin check
        """
        return Delayed(prefix=f'({repr(self)} in {repr(other)})')

    def __invert__(self):
        """
        Negate the Delayed object.
        
        Returns
        -------
        Delayed
            A new Delayed object representing the negation
        """
        return Delayed(prefix=f'~({repr(self)})')


def eval_delay(sel: Delayed, env: Dict=None):
    """
    Evaluate a Delayed object.
    
    Parameters
    ----------
    sel : Delayed
        The Delayed object to evaluate
    env : Dict, default=None
        Dictionary of variables to use for evaluation
        
    Returns
    -------
    object
        The result of evaluating the Delayed object
    """
    if env is None:
        env = dict()

    v = eval(repr(sel), env)
    return v


step = Delayed('step')
delay_lib = _Library()

from streamline import __version__
from typing import Optional, Callable, Dict, Set, List, Union, Tuple
import streamline as sl
import inspect
from importlib import import_module


class Var:
    """
    A class to represent a variable reference in a pipeline step.
    
    This class is used to reference existing variables in the environment
    when defining pipeline steps. It allows for variable substitution
    when a step is executed.
    """
    def __init__(self, name: str):
        """
        Initialize a Var object with a variable name.
        
        Parameters
        ----------
        name : str
            The name of the variable to reference
        
        Raises
        ------
        AssertionError
            If name is not a string
        """
        assert type(name) is str
        self._name = name
    @property
    def name(self):
        """
        Get the name of the variable.
        
        Returns
        -------
        str
            The name of the variable
        """
        return self._name
    def __repr__(self):
        """
        Get the string representation of the Var object.
        
        Returns
        -------
        str
            String representation of the Var object
        """
        return f'Var({repr(self._name)})'
    def __str__(self):
        """
        Get the string value of the Var object.
        
        Returns
        -------
        str
            The name of the variable
        """
        return self._name
    def __eq__(self, other):
        """
        Check if two Var objects are equal.
        
        Parameters
        ----------
        other : object
            The object to compare with
            
        Returns
        -------
        bool
            True if other is a Var with the same name, False otherwise
        """
        if type(other) is Var:
            return self._name == other._name
        return False
    def __hash__(self):
        """
        Get the hash of the Var object.
        
        Returns
        -------
        int
            The hash of the variable name
        """
        return hash(self._name)


class _Step:
    """
    Base class for all pipeline steps.
    
    This is an abstract class that provides the foundation for 
    different types of pipeline steps. It defines common properties
    and methods shared across all step implementations.
    """
    __version__ = __version__
    __steptype__ = 'Step'
    def __init__(
            self,
            args: Optional[List[str]]=None,
            kw: Optional[Dict]=None,
            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        """
        Initialize a Step object.
        
        Parameters
        ----------
        args : Optional[List[str]], default=None
            List of positional arguments or Var references
        kw : Optional[Dict], default=None
            Dictionary of keyword arguments
        arg_cat : Optional[str], default=None
            Argument category, used for grouping and filtering steps
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
            
        Raises
        ------
        AssertionError
            If arguments don't meet the expected types
        """
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
        """
        Get string representation of the Step object.
        
        Returns
        -------
        str
            String representation of the step
        """
        if self.tags is not None:
            return f'{self.__steptype__}(arg_cat={repr(self.arg_cat)}, tags={self.tags})'
        else:
            return f'{self.__steptype__}(arg_cat={repr(self.arg_cat)})'
    
    def __sourcecode__(self, create_call: Optional[bool]=False):
        """
        Get the source code of the step's function.
        
        Parameters
        ----------
        create_call : Optional[bool], default=False
            Whether to create a call string
            
        Returns
        -------
        str
            Source code of the function
            
        Raises
        ------
        AssertionError
            If create_call is True (not implemented)
        """
        assert not create_call, 'Not implemented'
        return inspect.getsource(self._fun)
    
    def get_dependencies(self):
        """
        Get the dependencies of the step.
        
        This is a list of variable names that are used in the step's args and kw.
        
        Returns
        -------
        list
            A list of variable names that this step depends on
        """
        dependencies = set()
        for a in self._args:
            if isinstance(a, Var):
                dependencies.add(a.name)
        for k, v in self._kw.items():
            if isinstance(v, Var):
                dependencies.add(v.name)
        return list(dependencies)
    
    def get_outputs(self):
        """
        Get the outputs of the step.
        
        This method should be implemented by subclasses to return
        the variable names that are created or modified by this step.
        
        Returns
        -------
        set
            A set of variable names that this step outputs
        """
        raise NotImplementedError('Subclasses must implement get_outputs method')

    def has_tag(self, tag: str) -> bool:
        """
        Check if this step has the specified tag.
        
        Parameters
        ----------
        tag : str
            The tag to check for
            
        Returns
        -------
        bool
            True if the step has the specified tag, False otherwise
        """
        return tag in self._tags

    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename variables in the step's args and kw based on the provided mapping.
        
        This is useful for updating variable names in a pipeline step when the
        variable names in the environment have changed.

        Parameters
        ----------
        variables : Optional[Dict[str, str] or callable], default=None
            A dictionary mapping old variable names to new variable names,
            or a function that takes a variable name and returns a new name

        arg_cat : Optional[str], default=None
            The new argument category for the step

        Returns
        -------
        _Step
            Self, for method chaining
            
        Raises
        ------
        AssertionError
            If the arguments are not of the expected types
        """
        if variables is not None:
            assert type(variables) is dict or callable, 'variables must be a dict or callable'
            if type(variables) is dict:
                fun = lambda x: variables.get(x, x)
            else:
                fun = variables
            self._args = [Var(fun(a.name)) if isinstance(a, Var) else a for a in self._args]
            self._kw = {
                Var(fun(k.name)) if isinstance(k, Var) else k: Var(fun(v.name)) if isinstance(v, Var) else v
                for k, v in self._kw.items()
            }
        if arg_cat is not None:
            assert type(arg_cat) is dict or callable or str, 'arg_cat must be a dict or callable'
            if type(arg_cat) is dict:
                fun = lambda x: arg_cat.get(x, x)
            elif type(arg_cat) is str:
                fun = lambda x: arg_cat
            else:
                fun = arg_cat
            self._arg_cat = fun(self._arg_cat)
        return self
    
    @property
    def arg_cat(self):
        """
        Get the argument category.
        
        Returns
        -------
        str
            The argument category
        """
        return self._arg_cat
        
    @property
    def args(self):
        """
        Get the positional arguments.
        
        Returns
        -------
        list
            List of positional arguments
        """
        return self._args
        
    @property
    def kw(self):
        """
        Get the keyword arguments.
        
        Returns
        -------
        dict
            Dictionary of keyword arguments
        """
        return self._kw
        
    @property
    def fun(self):
        """
        Get the function.
        
        Returns
        -------
        callable
            The function to be executed by this step
        """
        return self._fun
        
    @property
    def tags(self):
        """
        Get the tags associated with this step.
        
        Returns
        -------
        set
            Set of tags
        """
        return self._tags
    
    def __call__(self, *args, **kwargs):
        """
        Execute the step.
        
        This method must be implemented by subclasses.
        
        Raises
        ------
        NotImplementedError
            Always, as this is an abstract method
        """
        raise NotImplementedError('Subclasses must implement __call__ method')


class Function(_Step):
    """
    A pipeline step that executes a function.
    
    This step executes a Python function with the specified arguments and
    stores the result in the environment with the given output variable(s).
    """
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
        """
        Initialize a Function step.
        
        Parameters
        ----------
        fun : Callable
            The function to be executed
        args : Optional[List[str]], default=None
            List of positional arguments or Var references
        kw : Optional[Dict], default=None
            Dictionary of keyword arguments
        out_var : Optional[Union[str, Tuple[str]]], default=None
            Name of the variable(s) to store the function's output.
            If None, output is stored in '_'.
            If a tuple, function must return a tuple of the same length.
        arg_cat : Optional[str], default=None
            Argument category, used for grouping and filtering steps
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
            
        Raises
        ------
        AssertionError
            If arguments don't meet the expected types
        """
        assert callable(fun)
        self._fun = fun
        if tags is None:
            tags = set()
        super().__init__(args=args, kw=kw, arg_cat=arg_cat, tags=sl.tags.STEP_FUNCTION|tags)
        assert out_var is None or type(out_var) is str or ((type(out_var) is tuple or type(out_var) is list) and all((type(a) is str for a in out_var)))

        self._out_var = out_var

    @property
    def out_var(self):
        """
        Get the output variable name(s).
        
        Returns
        -------
        Union[str, Tuple[str], None]
            Name(s) of the variable(s) to store the function's output
        """
        return self._out_var
    
    def get_outputs(self):
        """
        Get the outputs of the function step.
        
        Returns
        -------
        set
            A set of variable names that this step outputs
        """
        out_var = self._out_var
        if out_var is None:
            return {'_'}
        elif isinstance(out_var, str):
            return {out_var}
        else:
            return set(out_var)

    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename variables in the function step.
        
        This renames both arguments and output variables.
        
        Parameters
        ----------
        variables : Optional[Dict[str, str] or callable], default=None
            A dictionary mapping old variable names to new variable names,
            or a function that takes a variable name and returns a new name
            
        arg_cat : Optional[str], default=None
            The new argument category for the step
            
        Returns
        -------
        Function
            Self, for method chaining
            
        Raises
        ------
        ValueError
            If out_var is of an unexpected type
        AssertionError
            If variables is neither a dict nor callable
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
        """
        Execute the function with the given environment and keyword arguments.
        
        This method resolves variable references, calls the function with
        appropriate arguments, and stores the results in the environment.
        
        Parameters
        ----------
        env : dict
            The environment dictionary containing variables
        kw : Dict
            Additional keyword arguments
            
        Returns
        -------
        any
            The output of the function
            
        Raises
        ------
        AssertionError
            If a referenced variable is not found in the environment
        """
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
    """
    A pipeline step that removes variables from the environment.
    
    This step is used to delete specified variables from the pipeline's
    runtime environment, which can be useful for cleaning up temporary
    variables or reducing memory usage.
    """
    __steptype__ = 'Delete'
    def __init__(
            self,

            args: Optional[Union[str, List[str]]]=None,

            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        """
        Initialize a Delete step.
        
        Parameters
        ----------
        args : Optional[Union[str, List[str]]], default=None
            Name(s) of variable(s) to delete from the environment
        arg_cat : Optional[str], default=None
            Argument category, used for grouping and filtering steps
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
            
        Raises
        ------
        AssertionError
            If args contains elements that are not strings or Var instances
        """
        if type(args) is str:
            args = [args]
        if tags is None:
            tags = set()
        super().__init__(args=args, arg_cat=arg_cat, tags=sl.tags.STEP_DELETE|tags)
        assert all([type(a) is str or isinstance(a, Var) for a in args]), 'All args must be strings'
    
    def get_outputs(self):
        """
        Get the outputs of the delete step.
        
        Since this step removes variables rather than creating them,
        it doesn't have any outputs in the typical sense.
        
        Returns
        -------
        set
            An empty set since this step doesn't produce any outputs
        """
        return set()

    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename variables in the Delete step.
        
        Updates the variables to be deleted based on the provided mapping.
        
        Parameters
        ----------
        variables : Optional[Dict[str, str] or callable], default=None
            A dictionary mapping old variable names to new variable names,
            or a function that takes a variable name and returns a new name
            
        arg_cat : Optional[str], default=None
            The new argument category for the step
            
        Returns
        -------
        Delete
            Self, for method chaining
            
        Raises
        ------
        AssertionError
            If variables is neither a dict nor callable
        """
        super().rename(variables=None, arg_cat=arg_cat)
        if variables is not None:
            assert type(variables) is dict or callable, 'variables must be a dict or callable'
            if type(variables) is dict:
                fun = lambda x: variables.get(x, x)
            else:
                fun = variables
            self._args = [Var(fun(k.name)) if isinstance(k, Var) else fun(k) for k in self._args]
        return self

    def __call__(self, env, kw: Dict):
        """
        Execute the delete operation on the given environment.
        
        Removes the specified variables from the environment.
        
        Parameters
        ----------
        env : dict
            The environment dictionary containing variables
        kw : Dict
            Additional keyword arguments (not used in this step)
            
        Returns
        -------
        None
            This step does not produce any output
        """
        for a in self._args:
            if type(a) is str:
                if a in env:
                    del env[a]
            elif isinstance(a, Var):
                if a.name in env:
                    del env[a.name]
        return None


class VariablesDict(_Step):
    """
    A pipeline step that adds variables to the environment.
    
    This step is used to add predefined variables or values to the pipeline's
    runtime environment. It can be used to set constants or transform existing
    variables into new ones.
    """
    __steptype__ = 'Function'
    def __init__(
            self,
            
            kw: Optional[Dict]=None,

            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        """
        Initialize a VariablesDict step.
        
        Parameters
        ----------
        kw : Optional[Dict], default=None
            Dictionary of variable names and their values to add to the environment
        arg_cat : Optional[str], default=None
            Argument category, used for grouping and filtering steps
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
        """
        if tags is None:
            tags = set()
        super().__init__(kw=kw, arg_cat=arg_cat, tags=sl.tags.STEP_VARIABLES_DICT|tags)

    def __call__(self, env, kw: Dict):
        """
        Execute the variable addition operation on the given environment.
        
        Adds the specified variables to the environment.
        
        Parameters
        ----------
        env : dict
            The environment dictionary containing variables
        kw : Dict
            Additional keyword arguments to merge with the step's keywords
            
        Returns
        -------
        None
            This step does not produce any output
        """
        def cvar(v):
            if isinstance(v, Var):
                return env[v.name]
            return v
        kwargs = self._kw | kw
        kwargs = {a: cvar(b) for a, b in kwargs.items()}

        for a, b in kwargs.items():
            env[a] = cvar(b)
        return None

    def get_outputs(self):
        """
        Get the outputs of the variables dictionary step.
        
        Returns
        -------
        set
            A set of variable names that will be added to the environment
        """
        return set(self._kw.keys())
    
    def rename(self, variables: Optional[Dict[str, str]]=None, arg_cat: Optional[str]=None):
        """
        Rename variables in the VariablesDict step.
        
        Updates both the variable names and their values if they are Var references.
        
        Parameters
        ----------
        variables : Optional[Dict[str, str] or callable], default=None
            A dictionary mapping old variable names to new variable names,
            or a function that takes a variable name and returns a new name
            
        arg_cat : Optional[str], default=None
            The new argument category for the step
            
        Returns
        -------
        VariablesDict
            Self, for method chaining
            
        Raises
        ------
        AssertionError
            If variables is neither a dict nor callable
        """
        super().rename(variables=None, arg_cat=arg_cat)
        if variables is not None:
            assert type(variables) is dict or callable, 'variables must be a dict or callable'
            if type(variables) is dict:
                fun = lambda x: variables.get(x, x)
            else:
                fun = variables
            self._kw = {fun(k): v for k, v in self._kw.items()}
        return self


class ImportLib(_Step):
    """
    A pipeline step that imports a Python module.
    
    This step imports a Python module and makes it available in the
    environment with a specified variable name. It is useful for
    importing libraries that will be used by subsequent steps.
    """
    __steptype__ = 'ImportLib'
    def __init__(
            self,
            module_name: str,
            var_name: Optional[str]=None,
            arg_cat: Optional[str]=None,
            tags: Optional[Set[str]]=None,
        ):
        """
        Initialize an ImportLib step.
        
        Parameters
        ----------
        module_name : str
            The name of the module to import
        var_name : Optional[str], default=None
            The variable name to use for the imported module.
            If None, uses the module name.
        arg_cat : Optional[str], default=None
            Argument category, used for grouping and filtering steps
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
            
        Raises
        ------
        AssertionError
            If module_name is not a string
        """
        assert isinstance(module_name, str), 'module_name must be a string'
        var_name = var_name or module_name

        if tags is None:
            tags = set()
        super().__init__(
            kw={Var(var_name): module_name},
            arg_cat=arg_cat, tags=sl.tags.STEP_IMPORT_LIB|tags
        )
    
    @property   
    def _module_name(self):
        """
        Get the module name.
        
        Returns
        -------
        str
            The name of the module to import
        """
        return list(self._kw.values())[0]
    
    @property   
    def _var_name(self):
        """
        Get the variable name.
        
        Returns
        -------
        str
            The name of the variable to store the module
        """
        return list(self._kw.keys())[0].name
    
    @property
    def module_name(self):
        """
        Get the module name.
        
        Returns
        -------
        str
            The name of the module to import
        """
        return self._module_name
    
    @property
    def var_name(self):
        """
        Get the variable name.
        
        Returns
        -------
        str
            The variable name for the imported module
        """
        return self._var_name
    
    def __repr__(self):
        """
        Get string representation of the ImportLib step.
        
        Returns
        -------
        str
            String representation showing module_name and var_name
        """
        return f"{self.__steptype__}(module_name={repr(self._module_name)}, var_name={repr(self._var_name)}, arg_cat={repr(self.arg_cat)}, tags={self.tags})"
    
    def __call__(self, env, kw: Dict):
        """
        Execute the import operation on the given environment.
        
        Imports the specified module and adds it to the environment.
        
        Parameters
        ----------
        env : dict
            The environment dictionary containing variables
        kw : Dict
            Additional keyword arguments (not used in this step)
            
        Returns
        -------
        module
            The imported module
            
        Raises
        ------
        ModuleNotFoundError
            If the specified module cannot be imported
        """
        module = import_module(self._module_name)
        env[self._var_name] = module
        return module
    
    
    def get_outputs(self):
        """
        Get the outputs of the step.
        
        Returns
        -------
        set
            A set containing the variable name of the imported module
        """
        return {self._var_name}
    
    def get_dependencies(self):
        """
        Get the dependencies of the step.
        
        ImportLib steps have no dependencies.
        
        Returns
        -------
        set
            An empty set
        """
        return set()

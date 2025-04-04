# Copyright (c) 2025, streamline
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.
# -*- coding: utf-8 -*-

from streamline import __version__
import streamline as sl
from .step import _Step, Function, Delete, VariablesDict, ImportLib

from typing import List, Dict, Optional, Tuple, Union, Callable, Set
from importlib import import_module
from copy import copy, deepcopy


class StepNotFound(Exception):
    """
    Exception raised when a step is not found in a pipeline.
    
    This exception is raised when attempting to access, modify, or remove
    a step that does not exist in the pipeline.
    """
    pass


class _Vertical:
    """
    Class for representing boolean masks for pipeline steps.
    
    This class is used internally to represent a mask of boolean values
    corresponding to steps in a pipeline. It is used for filtering and
    selecting steps based on conditions.
    """
    def __init__(self, filt: List):
        """
        Initialize a _Vertical with a list of boolean values.
        
        Parameters
        ----------
        filt : List
            List of boolean values representing a mask
        """
        self._filt = filt

    @property
    def filt(self):
        """
        Get the filter mask.
        
        Returns
        -------
        list
            The list of boolean values
        """
        return self._filt

    def __repr__(self):
        """
        Get string representation of the _Vertical.
        
        Returns
        -------
        str
            String representation of the filter mask
        """
        return repr(self._filt)

    def __len__(self):
        """
        Get the length of the filter mask.
        
        Returns
        -------
        int
            The number of elements in the filter mask
        """
        return len(self._filt)
        
    def __iter__(self):
        """
        Get an iterator over the filter mask.
        
        Returns
        -------
        iterator
            Iterator over the boolean values
        """
        return iter(self._filt)

        
class _Pipeline:
    """
    Base class for Pipeline implementations.
    
    This class provides the core functionality for managing a sequence of
    steps and executing them. It is not intended to be used directly;
    instead, use the Pipeline class, which inherits from this class.
    """
    __version__ = __version__
    def __init__(
            self,
            a: Optional[List[_Step]]=None,
        ):
        """
        Initialize a _Pipeline.
        
        Parameters
        ----------
        a : Optional[List[_Step]], default=None
            List of steps to add to the pipeline
            
        Raises
        ------
        AssertionError
            If any element in a is not a _Step instance
        """
        if a is None:
            a = list()
        else:
            assert isinstance(a, list) or hasattr(a, 'to_list')
            if hasattr(a, 'to_list'):
                a = a.to_list()
            assert all((isinstance(step, _Step) for step in a))
        self.__steps = a

    def __copy__(self):
        """
        Create a shallow copy of the pipeline.
        
        Returns
        -------
        _Pipeline
            A new pipeline with shallow copies of the steps
        """
        return self.__class__(
            a=copy([copy(step) for step in self.__steps])
        )

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the pipeline.
        
        Parameters
        ----------
        memo : dict
            Dictionary for memoization
            
        Returns
        -------
        _Pipeline
            A new pipeline with deep copies of the steps
        """
        return self.__class__(
            a=deepcopy(self.__steps)
        )

    def copy(self, deep: Optional[bool]=False):
        """
        Create a copy of the pipeline.
        
        Parameters
        ----------
        deep : Optional[bool], default=False
            Whether to create a deep copy
            
        Returns
        -------
        _Pipeline
            A copy of the pipeline
        """
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def print(self, show_args: Optional[bool]=False):
        """
        Get a string representation of the pipeline.
        
        Parameters
        ----------
        show_args : Optional[bool], default=False
            Whether to show the arguments of each step
            
        Returns
        -------
        str
            String representation of the pipeline
        """
        res = 'Pipeline(steps=['
        if show_args:
            res = res + ', '.join([f'\n  {step}' for step in self.__steps])
        else:
            res = res + ', '.join([f'\n  "Step {i}"' for i, _ in enumerate(self.__steps)])
                        
        res = res + "\n], "
        res += ')'
        return res

    def __repr__(self):
        """
        Get string representation of the pipeline.
        
        Returns
        -------
        str
            String representation of the pipeline
        """
        return self.print(show_args=True)

    def items(self):
        """
        Get an iterator over the steps in the pipeline.
        
        Returns
        -------
        iterator
            Iterator over the steps
        """
        return iter(self.__steps)
        
    def to_list(self):
        """
        Get a list of the steps in the pipeline.
        
        Returns
        -------
        list
            List of the steps
        """
        return list(self.items())

    def __len__(self):
        """
        Get the number of steps in the pipeline.
        
        Returns
        -------
        int
            The number of steps
        """
        return len(self.__steps)
    
    def __iter__(self):
        """
        Get an iterator over the steps in the pipeline.
        
        Returns
        -------
        iterator
            Iterator over the steps
        """
        return iter(self.__steps)
    
    def save(self, path: str):
        """
        Save the pipeline to a file.
        
        Parameters
        ----------
        path : str
            Path to save the pipeline to
            
        Returns
        -------
        _Pipeline
            Self, for method chaining
        """
        sl.utilities.save_obj(self, path=path)
        return self
        
    @classmethod
    def load(cls, path: str):
        """
        Load a pipeline from a file.
        
        Parameters
        ----------
        path : str
            Path to load the pipeline from
            
        Returns
        -------
        _Pipeline
            The loaded pipeline
            
        Notes
        -----
        This is a class method, so it can be called on the class directly,
        e.g., `_Pipeline.load('path/to/file')`.
        """
        return sl.utilities.load_obj(path=path)
        
    def get_outputs(self):
        """
        Get all outputs from all steps in the pipeline.
        
        Returns
        -------
        set
            Set of variable names that are outputs of the pipeline steps
        """
        # Get all outputs from all steps
        all_outputs = set()
        for step in self.__steps:
            step_outputs = step.get_outputs()
            all_outputs.update(step_outputs)
            
        return all_outputs
        
    def get_steps_by_input(self, input_name):
        """
        Get all steps that have a specific input dependency.
        
        Parameters
        ----------
        input_name : str
            The name of the input variable to search for
            
        Returns
        -------
        list
            List of steps that depend on the specified input variable
        """
        steps = []
        for step in self.__steps:
            step_deps = step.get_dependencies()
            if isinstance(step_deps, list):
                step_deps = set(step_deps)
                
            if input_name in step_deps:
                steps.append(step)
                
        return steps
        
    def get_steps_by_output(self, output_name):
        """
        Get all steps that produce a specific output.
        
        Parameters
        ----------
        output_name : str
            The name of the output variable to search for
            
        Returns
        -------
        list
            List of steps that produce the specified output variable
        """
        steps = []
        for step in self.__steps:
            step_outputs = step.get_outputs()
            if output_name in step_outputs:
                steps.append(step)
                
        return steps

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

            accum_deps = accum_deps | (new_deps - defs)
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
    
    def extend(self, steps: List[_Step]):
        """
        Extend the pipeline with a list of steps.
        """
        for step in steps:
            self._add_step(step)
        return self

    def _append(self, step: _Step):
        assert isinstance(step, _Step)
        self.__steps.append(step)
    
    def map(self, a, globals_=None):
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
    
    def apply(self, func: Callable):
        return Pipeline(a=[func(step) for step in self.__steps])

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

    def get_dependencies(self):
        """
        Get all input dependencies for the pipeline.
        
        This method uses get_env_info to track all variables through the pipeline
        and identifies which external variables are required as inputs.
        
        Returns
        -------
        set
            Set of variable names that this pipeline depends on from outside
        """
        # Get environment information for all steps
        env_info = self.get_env_info()
        
        # If there are no steps, return an empty set
        if not env_info:
            return set()
            
        # Get dependencies from the last step's information
        # The 'unresolved_vars_accum' field contains all variables that haven't been
        # resolved (defined) within the pipeline
        last_step_info = env_info[-1]
        external_deps = last_step_info['unresolved_vars_accum']
        
        return external_deps

    def get_steps_by_tag(self, tag):
        """
        Get all steps that have a specific tag.
        
        Parameters
        ----------
        tag : str
            The tag to search for
            
        Returns
        -------
        list
            List of steps that have the specified tag
        """
        steps = []
        for step in self.__steps:
            if hasattr(step, 'has_tag') and step.has_tag(tag):
                steps.append(step)
                
        return steps

    def get_subgraph(self, inputs=None, outputs=None):
        """
        Extract a subgraph of the pipeline based on specified inputs and outputs.
        
        This method identifies all steps that form a path from the specified 
        input variables to the specified output variables, creating a new pipeline
        containing only those steps.
        
        Parameters
        ----------
        inputs : list or set or str, default=None
            The input variable(s) where the subgraph starts.
            If None, all pipeline inputs are used.
        outputs : list or set or str, default=None
            The output variable(s) where the subgraph ends.
            If None, all pipeline outputs are used.
            
        Returns
        -------
        Pipeline
            A new pipeline containing only the steps that form the subgraph
            
        Notes
        -----
        The algorithm works by:
        1. Identifying all steps that produce the specified outputs
        2. Working backwards to find all steps needed to produce the inputs for those steps
        3. Continuing until reaching the specified inputs or the start of the pipeline
        """
        # Convert single strings to lists for consistent handling
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
            
        # If no inputs are specified, use all external dependencies
        if inputs is None:
            inputs = self.get_dependencies()
            
        # If no outputs are specified, use all outputs
        if outputs is None:
            outputs = self.get_outputs()
            
        # Convert to sets for efficient membership testing
        if not isinstance(inputs, set):
            inputs = set(inputs)
        if not isinstance(outputs, set):
            outputs = set(outputs)
            
        # Get environment info to track variable dependencies
        env_info = self.get_env_info()
        if not env_info:
            return self.__class__()  # Return empty pipeline if no steps
            
        # Track which variables have been defined in the subgraph
        defined_vars = set()
        
        # Track which steps will be included in the subgraph
        included_steps = set()
        
        # First pass: identify all steps that produce output variables
        output_steps = set()
        for i, step_info in enumerate(env_info):
            step_outputs = step_info['added_vars']
            if not outputs.isdisjoint(step_outputs):
                output_steps.add(i)
                
        # If no output steps found, return empty pipeline
        if not output_steps:
            return self.__class__()
            
        # Work backwards from output steps to find all required steps
        pending_steps = output_steps.copy()
        while pending_steps:
            step_idx = max(pending_steps)  # Process steps in reverse order
            pending_steps.remove(step_idx)
            
            # Add this step to included steps
            included_steps.add(step_idx)
            
            # Add outputs of this step to defined variables
            step_info = env_info[step_idx]
            defined_vars.update(step_info['added_vars'])
            
            # Check if this step depends on variables that need to be produced
            required_vars = step_info['required_vars'] - inputs
            if required_vars:
                # Find steps that produce these variables
                for i, prev_step_info in enumerate(env_info[:step_idx]):
                    if i in included_steps:
                        continue  # Skip steps already included
                        
                    # If this step produces any of the required variables, add it to pending
                    if not required_vars.isdisjoint(prev_step_info['added_vars']):
                        pending_steps.add(i)
        
        # Create new pipeline with included steps
        included_indices = sorted(included_steps)
        subgraph_steps = [self.__steps[i] for i in included_indices]
        
        return self.__class__(a=subgraph_steps)

    def validate(self, env=None):
        """
        Validate that the pipeline can be executed with the given environment.
        
        This method checks if all required input variables for the pipeline
        will be available when executed with the given environment. It identifies
        potential missing dependencies that would cause execution errors.
        
        Parameters
        ----------
        env : dict or RunEnv, default=None
            The environment to validate against. If None, an empty environment is used.
            
        Returns
        -------
        bool
            True if the pipeline is valid, False otherwise
            
        Raises
        ------
        AssertionError
            If there are missing dependencies, with details about what's missing
        """
        # Convert environment to a set of available variables
        if env is None:
            available_vars = set()
        else:
            available_vars = set(env.keys())
            
        # Get environment info to track variable dependencies
        env_info = self.get_env_info()
        
        # If no steps in pipeline, it's valid
        if not env_info:
            return True
            
        # Track variables that become available during pipeline execution
        defined_vars = available_vars.copy()
        
        # Check each step for missing dependencies
        for step_idx, step_info in enumerate(env_info):
            # Get variables required by this step
            required_vars = step_info['required_vars']
            
            # Check if all required variables are available
            missing_vars = required_vars - defined_vars
            
            if missing_vars:
                step = self.__steps[step_idx]
                raise AssertionError(
                    f"Missing dependencies at step {step_idx}: {missing_vars}\n"
                    f"Step: {step}\n"
                    f"Available variables: {defined_vars}"
                )
                
            # Add variables defined by this step for next iterations
            defined_vars.update(step_info['added_vars'])
            
            # Remove variables deleted by this step
            defined_vars -= step_info['removed_vars']
        
        return True


class _Loc:
    def __init__(self, pipe):
        self.__pipe = pipe
    
    def _stdize_index(self, obj, throw_error: Optional[bool]=True):
        assert type(obj) is int or type(obj) is list or isinstance(obj, sl.Delayed) or isinstance(obj, slice) or isinstance(obj, _Vertical), type(obj)
        if isinstance(obj, sl.Delayed):
            return self.__pipe.map(obj)
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
    """
    Main pipeline class for executing a sequence of steps.
    
    A Pipeline is a container for a sequence of steps that can be executed in order.
    It provides methods for adding, removing, and modifying steps, as well as
    running the pipeline on a given environment.
    """
    def __init__(
            self,
            a: Optional[List[Tuple]]=None,
        ):
        """
        Initialize a Pipeline.
        
        Parameters
        ----------
        a : Optional[List[Tuple]], default=None
            List of steps to add to the pipeline
        """
        super().__init__(a=a)
        self._loc = _Loc(self)
        
    @property
    def loc(self):
        """
        Get a _Loc object for accessing steps by boolean indexing.
        
        Returns
        -------
        _Loc
            Accessor for the pipeline's steps
        """
        return self._loc

    def __add__(self, other):
        """
        Concatenate two pipelines.
        
        Parameters
        ----------
        other : Pipeline
            The pipeline to concatenate with this one
            
        Returns
        -------
        Pipeline
            A new pipeline containing all steps from both pipelines
        """
        assert isinstance(other, Pipeline)
        return concat([self, other])
        
    def add_step(
            self,
            step: Union[Function, Delete, VariablesDict, ImportLib],
            index: Optional[int]=None,
        ):
        """
        Add a step to the pipeline.
        
        Parameters
        ----------
        step : Union[Function, Delete, VariablesDict]
            The step to add
        index : Optional[int], default=None
            The index at which to insert the step.
            If None, the step is appended to the end.
            
        Returns
        -------
        Pipeline
            Self, for method chaining
        """
        assert isinstance(step, (Function, Delete, VariablesDict, ImportLib)), type(step)
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
        """
        Add a function step to the pipeline.
        
        Parameters
        ----------
        a : Union[Function, Callable]
            The function or Function step to add
        index : Optional[int], default=None
            The index at which to insert the step
        args : Optional[List[str]], default=None
            List of positional arguments or Var references
        kw : Optional[Dict], default=None
            Dictionary of keyword arguments
        out_var : Optional[Union[str, Tuple[str]]], default=None
            Name of the variable(s) to store the function's output
        arg_cat : Optional[str], default=None
            Argument category, used for grouping and filtering steps
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
            
        Returns
        -------
        Pipeline
            Self, for method chaining
        """
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
        """
        Add a delete step to the pipeline.
        
        Parameters
        ----------
        a : Union[Delete, str, List[str]]
            The variables to delete or a Delete step
        index : Optional[int], default=None
            The index at which to insert the step
        arg_cat : Optional[str], default=None
            Argument category, used for grouping and filtering steps
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
            
        Returns
        -------
        Pipeline
            Self, for method chaining
        """
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
        """
        Add a variables dictionary step to the pipeline.
        
        Parameters
        ----------
        a : Union[VariablesDict, Dict]
            The variables to add or a VariablesDict step
        index : Optional[int], default=None
            The index at which to insert the step
        arg_cat : Optional[str], default=None
            Argument category, used for grouping and filtering steps
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
            
        Returns
        -------
        Pipeline
            Self, for method chaining
        """
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
        """
        Add an import library step to the pipeline.
        
        Parameters
        ----------
        a : Union[Dict, str]
            The library to import. If a string, imports a single library.
            If a dictionary, keys are library names and values are aliases.
        alias : Optional[str], default=None
            The alias for the imported library (only used if a is a string)
        index : Optional[int], default=None
            The index at which to insert the step
        tags : Optional[Set[str]], default=None
            Set of tags associated with this step
            
        Returns
        -------
        Pipeline
            Self, for method chaining
        """
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

    def _run(self, env: Optional[Dict]=None, kw: Optional[Dict]=None):
        """
        Internal method to run the pipeline on a given environment.
        
        Parameters
        ----------
        env : Optional[Dict], default=None
            The environment to use for execution
        kw : Optional[Dict], default=None
            Additional keyword arguments
            
        Returns
        -------
        Dict
            The environment after executing all steps
            
        Notes
        -----
        This method modifies the environment in place.
        """
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
            out = step(env=env, kw=kw_new)
            env._add_step(step=step, kwargs=kw_filt)
            yield out, env
        return env

    def run(self, env: Optional[Dict]=None, kw: Optional[Dict]=None):
        """
        Run the pipeline on a given environment.
        
        Parameters
        ----------
        env : Optional[Dict], default=None
            The environment to use for execution
        kw : Optional[Dict], default=None
            Additional keyword arguments
            
        Returns
        -------
        Dict
            The environment after executing all steps
        """
        if env is None or type(env) is dict:
            env = sl.RunEnv(env=env)
        iterator = self._run(env=env, kw=kw)
        for _, env in iterator:
            pass
        return env

    def run_parallel(self, env_l: List=None, kw_l: Optional[List[Dict]]=None):
        """
        Run the pipeline on multiple environments in parallel.
        
        Parameters
        ----------
        env_l : List, default=None
            List of environments to use for execution
        kw_l : Optional[List[Dict]], default=None
            List of additional keyword arguments, one for each environment
            
        Returns
        -------
        List
            List of environments after executing all steps
        """
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


class DebugPipeline:
    # TODO: finish class
    """
    DebugPipeline is a subclass of Pipeline that is used for debugging purposes.
    It allows you to run the pipeline and inspect the environment at each step.
    """
    def __init__(self, a: Optional[List[Tuple]]=None, env: Optional[List[Tuple]]=None, start: Optional[int]=0):
        """
        Initialize the DebugPipeline with an optional list of steps and an environment.
        """
        if env is None:
            env = sl.RunEnv()
        self._pipeline = Pipeline(a=a)
        self._env = env
        self._run_step = start
    
    def reset_run(self, start: Optional[int]=0):
        """
        Reset the current step to 0.
        """
        self._run_step = start

    def reset_env(self):
        """
        Reset the environment to an empty dictionary.
        """
        self._env = sl.RunEnv()
        return self._env
    
    def reset(self):
        """
        Reset the pipeline and environment to their initial states.
        """
        self.reset_step()
        self.reset_env()
        return self._env
    
    def run(self, kw: Optional[Dict]=None):
        """
        Run the pipeline from the current step
        """
        self.reset()
        self._pipeline.run(env=self._env, kw=kw)
        self._run_step = len(self._pipeline)
        return self._env
        
    def resume(self, kw: Optional[Dict]=None):
        """
        Run the pipeline with the given keyword arguments.
        """
        l = len(self._pipeline)
        self._pipeline[self._run_step:].run(env=self._env, kw=kw)
        self._run_step = l
        return self._env
    
    def sim_resume(self, a=None, kw: Optional[Dict]=None):
        """
        Simulate the run of the pipeline without modifying the environment.
        """
        if a is None:
            pipe = self._pipeline[self._run_step:]
        else:
            pipe = Pipeline(a=a)
        env = self._env.copy()
        pipe.run(env=env, kw=kw)
        return env


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
    """
    Concatenate multiple pipelines into a single pipeline.
    
    Parameters
    ----------
    list_pipes : List
        List of pipelines to concatenate
        
    Returns
    -------
    Pipeline
        A new pipeline containing all steps from all input pipelines
    """
    assert len(list_pipes) > 0, 'list_pipes must have at least one element'
    steps = sum([pipe._Pipeline__steps for pipe in list_pipes], [])
    return Pipeline(a=steps)


def load_pipeline(path: str):
    """
    Load a pipeline from a file.
    
    Parameters
    ----------
    path : str
        Path to load the pipeline from
        
    Returns
    -------
    Pipeline
        The loaded pipeline
    """
    return Pipeline.load(path)

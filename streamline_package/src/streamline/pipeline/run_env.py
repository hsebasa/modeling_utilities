from streamline.utilities import save_json, load_json, save_obj, load_obj, save_pandas_df, load_pandas_df
from streamline import __version__
import streamline as sl
from .step import _Step


from typing import Optional, Dict, List, Any
import pandas as pd
import os


class DEFAULT:
    """
    A sentinel class used to indicate that default values should be used.
    
    This is used as a marker to distinguish between explicitly passing None
    and not providing a value at all.
    """
    pass


class SAVE_PANDAS_DF(dict):
    """
    Configuration class for saving pandas DataFrames.
    
    This class specifies options for how pandas DataFrames should be saved
    when saving a RunEnv.
    """
    def __init__(self, format: Optional[str]='parquet', **kwargs):
        """
        Initialize a SAVE_PANDAS_DF configuration.
        
        Parameters
        ----------
        format : Optional[str], default='parquet'
            The format to save DataFrames in ('parquet' or 'csv')
        **kwargs : dict
            Additional keyword arguments to pass to the save function
            
        Raises
        ------
        AssertionError
            If format is not one of 'parquet' or 'csv'
        """
        assert format in {'parquet', 'csv'}
        super().__init__(pandas_df=dict(format=format, kwargs=kwargs))


class RunEnv:
    """
    Runtime environment for pipeline execution.
    
    RunEnv is a dictionary-like object that stores variables during pipeline
    execution. It can be saved to and loaded from disk, and can be used to
    generate a pipeline that recreates its contents.
    """
    __version__ = __version__
    def __init__(self, env: Dict=None):
        """
        Initialize a RunEnv.
        
        Parameters
        ----------
        env : Dict, default=None
            Initial dictionary of variables
        """
        if env is None:
            env = dict()
        self._env = env

    def __repr__(self):
        """
        Get string representation of the RunEnv.
        
        Returns
        -------
        str
            String representation of the underlying dictionary
        """
        return repr(self._env)

    def __str__(self):
        """
        Get string representation of the RunEnv.
        
        Returns
        -------
        str
            String representation of the underlying dictionary
        """
        return str(self._env)

    def __contains__(self, key):
        """
        Check if a key is in the RunEnv.
        
        Parameters
        ----------
        key : str
            The key to check
            
        Returns
        -------
        bool
            True if the key is in the RunEnv, False otherwise
        """
        return key in self._env
        
    def __getitem__(self, key):
        """
        Get a value from the RunEnv.
        
        Parameters
        ----------
        key : str
            The key to get
            
        Returns
        -------
        Any
            The value associated with the key
            
        Raises
        ------
        KeyError
            If the key is not in the RunEnv
        """
        return self._env[key]

    def __delitem__(self, key):
        """
        Delete a key from the RunEnv.
        
        Parameters
        ----------
        key : str
            The key to delete
            
        Raises
        ------
        KeyError
            If the key is not in the RunEnv
        """
        del self._env[key]
    
    def get(self, key, default=None):
        """
        Get a value from the RunEnv.
        
        Parameters
        ----------
        key : str
            The key to get
        default : Any, default=None
            The value to return if the key is not in the RunEnv
        """
        return self._env.get(key, default)
    
    def update(self, other: Dict):
        """
        Update the RunEnv with another dictionary.
        
        Parameters
        ----------
        other : Dict
            Dictionary of key-value pairs to update the RunEnv with
        """
        self._env.update(other)
    
    def __setitem__(self, key, obj):
        """
        Set a value in the RunEnv.
        
        Parameters
        ----------
        key : str
            The key to set
        obj : Any
            The value to set
        """
        self._env[key] = obj

    def __copy__(self):
        """
        Create a shallow copy of the RunEnv.
        
        Returns
        -------
        RunEnv
            A new RunEnv with a shallow copy of the underlying dictionary
        """
        return RunEnv(env=self._env.copy())
    
    def __deepcopy__(self, memo):
        """
        Create a deep copy of the RunEnv.
        
        Parameters
        ----------
        memo : dict
            Dictionary for memoization
            
        Returns
        -------
        RunEnv
            A new RunEnv with a deep copy of the underlying dictionary
        """
        from copy import deepcopy
        return RunEnv(env=deepcopy(self._env, memo=memo))
    
    def copy(self, deep: bool=False):
        """
        Create a copy of the RunEnv.
        
        Parameters
        ----------
        deep : bool, default=False
            Whether to create a deep copy
            
        Returns
        -------
        RunEnv
            A copy of the RunEnv
        """
        if deep:
            return self.__deepcopy__(memo={})
        else:
            return self.__copy__()

    def keys(self):
        """
        Get the keys in the RunEnv.
        
        Returns
        -------
        dict_keys
            The keys in the underlying dictionary
        """
        return self._env.keys()

    def values(self):
        """
        Get the values in the RunEnv.
        
        Returns
        -------
        dict_values
            The values in the underlying dictionary
        """
        return self._env.values()

    def items(self):
        """
        Get the items in the RunEnv.
        
        Returns
        -------
        dict_items
            The items in the underlying dictionary
        """
        return self._env.items()

    def __iter__(self):
        """
        Get an iterator over the keys in the RunEnv.
        
        Returns
        -------
        iterator
            Iterator over the keys in the underlying dictionary
        """
        return self._env.__iter__()

    def _add_step(self, step: _Step, kwargs: dict()):
        """
        Add a step to the RunEnv's preamble.
        
        Parameters
        ----------
        step : _Step
            The step to add
        kwargs : dict
            Keyword arguments for the step
        """
        if '__preamble__' not in self._env:
            self._env['__preamble__'] = dict()
        
        if '__steps__' not in self._env['__preamble__']:
            self._env['__preamble__']['__steps__'] = list()
        
        self._env['__preamble__']['__steps__'].append({'step': step, 'kwargs': kwargs})
        
    def gen_pipeline(self):
        """
        Generate a pipeline that recreates this RunEnv.
        
        Returns
        -------
        Pipeline
            A pipeline that, when run, will recreate the contents of this RunEnv
        """
        return sl.Pipeline([c['step'] for c in self._env['__preamble__']['__steps__']])

    def add_var(
            self,
            var_name: str,
            var: Any,
        ):
        """
        Add a variable to the RunEnv.
        
        Parameters
        ----------
        var_name : str
            The name of the variable
        var : Any
            The value of the variable
        """
        self[var_name] = var
        
    def gen_run_kwargs(self):
        """
        Generate keyword arguments for running the pipeline.
        
        Returns
        -------
        dict
            Dictionary of keyword arguments
        """
        kw = dict()
        for c in self._env['__preamble__']['__steps__']:
            for d, e in c['kwargs'].items():
                kw[c['step'].arg_cat+'_'+d] = e
        return kw
    
    def save(
            self,
            path_folder: str,
            save_vars: Optional[List]=None,
            save_options: Optional[Dict]=DEFAULT(),
        ):
        """
        Save the RunEnv to disk.
        
        This method saves the RunEnv to a directory, handling special cases
        for certain types of objects like pandas DataFrames.
        
        Parameters
        ----------
        path_folder : str
            The path to the directory to save to
        save_vars : Optional[List], default=None
            List of variable names to save. If None, all variables are saved.
        save_options : Optional[Dict], default=DEFAULT()
            Options for saving. If DEFAULT(), default options are used.
            
        Notes
        -----
        The RunEnv is saved as a directory containing:
        - env.dill: The serialized environment
        - info.json: Metadata about the saved environment
        - dedicated/: Directory for specially-handled objects
        """
        if isinstance(save_options, DEFAULT):
            save_options = SAVE_PANDAS_DF()
        if save_vars is None:
            save_vars = list(self._env.keys())
            
        path_folder_dedicated = os.path.join(path_folder, 'dedicated')
        path_env = os.path.join(path_folder, 'env.dill')
        path_info = os.path.join(path_folder, 'info.json')

        if not os.path.isdir(path_folder):
            os.makedirs(path_folder)
        if not os.path.isdir(path_folder_dedicated):
            os.makedirs(path_folder_dedicated)

        dict_info = dict(
            version=self.__version__,
            dedicated=dict(),
            saved_vars=list(),
            save_options=save_options
        )
        env_filt = dict()
        for name in save_vars:
            if name not in self._env:
                continue
            var = self._env[name]
            if type(var) is pd.DataFrame and 'pandas_df' in save_options:
                options = save_options['pandas_df']
                kwargs = dict()
                if 'kwargs' in options:
                    kwargs = options['kwargs'].copy()
                if 'format' in options:
                    kwargs['format'] = options['format']
                path = save_pandas_df(var, os.path.join(path_folder_dedicated, name), **kwargs)
                dict_info['dedicated'][name] = {'type': 'pandas_df', 'path': os.path.relpath(path, path_folder)}
            else:
                env_filt[name] = self._env[name]
            dict_info['saved_vars'].append(name)

        save_obj(d=env_filt, path=path_env)
        dict_info['path_env'] = os.path.relpath(path_env, path_folder)
        
        save_json(dict_info, path_info)


def load_runenv(path_folder: str):
    """
    Load a RunEnv from disk.
    
    Parameters
    ----------
    path_folder : str
        The path to the directory to load from
        
    Returns
    -------
    RunEnv
        The loaded RunEnv
        
    Raises
    ------
    NotImplementedError
        If a variable type is not supported
    """
    path_env = os.path.join(path_folder, 'env.dill')
    path_info = os.path.join(path_folder, 'info.json')
    
    dict_info = load_json(path_info)
    env_filt = load_obj(path_env)
    for var in dict_info['saved_vars']:
        if var in dict_info['dedicated']:
            var_type = dict_info['dedicated'][var]['type']
            path = os.path.join(path_folder, dict_info['dedicated'][var]['path'])
            assert var_type in dict_info['save_options']
            if var_type == 'pandas_df':
                format = dict_info['save_options'][var_type]['format']
                df = load_pandas_df(path=path, format=format)
                env_filt[var] = df
            else:
                raise NotImplementedError
    return RunEnv(env_filt)
                
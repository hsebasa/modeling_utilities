from streamline.utilities import save_json, load_json, save_obj, load_obj, save_pandas_df, load_pandas_df
from streamline import __version__
import streamline as sl
from .step import _Step


from typing import Optional, Dict, List, Any
import pandas as pd
import json
import os


class DEFAULT:
    pass


class SAVE_PANDAS_DF(dict):
    def __init__(self, format: Optional[str]='parquet', **kwargs):
        assert format in {'parquet', 'csv'}
        super().__init__(pandas_df=dict(format=format, kwargs=kwargs))


class RunEnv:
    __version__ = __version__
    def __init__(self, env: Dict=None):
        if env is None:
            env = dict()
        self._env = env

    def __repr__(self):
        return repr(self._env)

    def __str__(self):
        return str(self._env)

    def __contains__(self, key):
        return key in self._env
        
    def __getitem__(self, key):
        return self._env[key]

    def __delitem__(self, key):
        del self._env[key]

    def __setitem__(self, key, obj):
        self._env[key] = obj

    def keys(self):
        return self._env.keys()

    def values(self):
        return self._env.values()

    def items(self):
        return self._env.items()

    def __iter__(self):
        return self._env.__iter__()

    def _add_step(self, step: _Step, kwargs: dict()):
        if '__preamble__' not in self._env:
            self._env['__preamble__'] = dict()
        
        if '__steps__' not in self._env['__preamble__']:
            self._env['__preamble__']['__steps__'] = list()
        
        self._env['__preamble__']['__steps__'].append({'step': step, 'kwargs': kwargs})
        
    def gen_pipeline(self):
        return sl.Pipeline([c['step'] for c in self._env['__preamble__']['__steps__']])

    def add_var(
            self,
            var_name: str,
            var: Any,
        ):
        self[var_name] = var
        
    def gen_run_kwargs(self):
        kw = dict()
        for c in self._env['__preamble__']['__steps__']:
            for d, e in c['kwargs'].items():
                kw[c['step'].arg_cat+'.'+d] = e
        return kw
    
    def save(
            self,
            path_folder: str,
            save_vars: Optional[List]=None,
            save_options: Optional[Dict]=DEFAULT(),
        ):
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
    path_folder_dedicated = os.path.join(path_folder, 'dedicated')
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
                
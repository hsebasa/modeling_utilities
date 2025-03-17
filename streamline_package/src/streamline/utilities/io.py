from typing import Dict, Any, Optional
import pandas as pd
import json
import dill
import os


def save_json(d: Dict, path: str):
    if not path.endswith('json'):
        path = path+'.json' 
    with open(path, 'w') as f:
        f.write(json.dumps(d))
    return path


def load_json(path: str):
    with open(path, 'r') as f:
        c = json.loads(f.read())
    return c


def save_obj(d: Any, path: str):
    if not path.endswith('dill'):
        path = path+'.dill' 
    with open(path, 'wb') as f:
        f.write(dill.dumps(d))
    return path


def load_obj(path: str):
    with open(path, 'rb') as f:
        c = dill.loads(f.read())
    return c


def save_pandas_df(df: pd.DataFrame, path: str, format: Optional[str]='parquet', **kwargs):
    assert format in {'parquet', 'csv'}
    kwargs = {'index': False}|kwargs
    if not path.endswith(format):
        path = path+'.'+format
    if format == 'parquet':
        df.to_parquet(path, **kwargs)
    elif format == 'csv':
        df.to_csv(path, **kwargs)
    else:
        df.to_excel(path, **kwargs)
    return path


def load_pandas_df(path: str, format: Optional[str]='parquet'):
    assert format in {'parquet', 'csv'}
    if not path.endswith(format):
        path = path+'.'+format
    if format == 'parquet':
        df = pd.read_parquet(path)
    elif format == 'csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return df
    
from typing import Dict, Any, Optional
import pandas as pd
import json
import dill
import os


def save_json(d: Dict, path: str):
    """
    Save a dictionary to a JSON file.
    
    Parameters
    ----------
    d : Dict
        The dictionary to save
    path : str
        The path to save the JSON file to
        
    Returns
    -------
    str
        The full path to the saved file
    """
    if not path.endswith('json'):
        path = path+'.json' 
    with open(path, 'w') as f:
        f.write(json.dumps(d))
    return path


def load_json(path: str):
    """
    Load a dictionary from a JSON file.
    
    Parameters
    ----------
    path : str
        The path to the JSON file
        
    Returns
    -------
    Dict
        The loaded dictionary
    """
    with open(path, 'r') as f:
        c = json.loads(f.read())
    return c


def save_obj(d: Any, path: str):
    """
    Save an object to a dill file.
    
    Parameters
    ----------
    d : Any
        The object to save
    path : str
        The path to save the dill file to
        
    Returns
    -------
    str
        The full path to the saved file
    """
    if not path.endswith('dill'):
        path = path+'.dill' 
    with open(path, 'wb') as f:
        f.write(dill.dumps(d))
    return path


def load_obj(path: str):
    """
    Load an object from a dill file.
    
    Parameters
    ----------
    path : str
        The path to the dill file
        
    Returns
    -------
    Any
        The loaded object
    """
    with open(path, 'rb') as f:
        c = dill.loads(f.read())
    return c


def save_pandas_df(df: pd.DataFrame, path: str, format: Optional[str]='parquet', **kwargs):
    """
    Save a pandas DataFrame to a file.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save
    path : str
        The path to save the file to
    format : Optional[str], default='parquet'
        The format to save the DataFrame in ('parquet' or 'csv')
    **kwargs : dict
        Additional keyword arguments to pass to the save function
        
    Returns
    -------
    str
        The full path to the saved file
        
    Raises
    ------
    AssertionError
        If format is not one of 'parquet' or 'csv'
    """
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
    """
    Load a pandas DataFrame from a file.
    
    Parameters
    ----------
    path : str
        The path to the file
    format : Optional[str], default='parquet'
        The format of the file ('parquet' or 'csv')
        
    Returns
    -------
    pd.DataFrame
        The loaded DataFrame
        
    Raises
    ------
    AssertionError
        If format is not one of 'parquet' or 'csv'
    """
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
    
# Streamline Package API Reference

This document provides a detailed reference of the Streamline package API.

## Table of Contents

- [Pipeline](#pipeline)
- [Steps](#steps)
- [Delayed Evaluation](#delayed-evaluation)
- [Run Environment](#run-environment)
- [Models](#models)
- [Utilities](#utilities)

## Pipeline

### Pipeline Class

The main class for creating and managing computational pipelines.

```python
from streamline.pipeline import Pipeline
```

#### Constructor

```python
Pipeline(steps=None)
```

**Parameters:**
- `steps` (list, optional): A list of Step objects to initialize the pipeline with.

#### Methods

##### `add_step`

```python
add_step(step, index=None, tags=None)
```

**Parameters:**
- `step` (Step): The step to add to the pipeline.
- `index` (int, optional): Position to insert the step. If None, appends to the end.
- `tags` (list, optional): Tags to associate with the step.

**Returns:** The Pipeline instance for method chaining.

##### `add_function`

```python
add_function(func, args=None, out_var=None, index=None, tags=None)
```

**Parameters:**
- `func` (callable): The function to add as a step.
- `args` (str, list, optional): Input variable name(s) for the function.
- `out_var` (str, optional): The name of the output variable.
- `index` (int, optional): Position to insert the step.
- `tags` (list, optional): Tags to associate with the step.

**Returns:** The Pipeline instance for method chaining.

##### `add_variables_dict`

```python
add_variables_dict(var_dict, index=None, tags=None)
```

**Parameters:**
- `var_dict` (dict): Dictionary of variables to add.
- `index` (int, optional): Position to insert the step.
- `tags` (list, optional): Tags to associate with the step.

**Returns:** The Pipeline instance for method chaining.

##### `add_model`

```python
add_model(model, index=None, args=None, kw=None, out_var=None, fit_args=None, 
          fit_kw=None, fit_out_var=None, predict_run=True, predict_args=None, 
          predict_kw=None, predict_out_var=None, arg_cat=None, tags=None)
```

**Parameters:**
- `model` (BaseModel): The model to add to the pipeline.
- `index` (int, optional): Position to insert the step.
- `args`/`kw` (list/dict, optional): Arguments for model operations.
- `out_var` (str, optional): Output variable name.
- `fit_args`/`fit_kw` (list/dict, optional): Arguments for model fitting.
- `fit_out_var` (str, optional): Output variable name for fit operation.
- `predict_run` (bool): Whether to run prediction after fitting.
- `predict_args`/`predict_kw` (list/dict, optional): Arguments for prediction.
- `predict_out_var` (str, optional): Output variable name for prediction.
- `arg_cat` (bool, optional): Whether to concatenate arguments.
- `tags` (list, optional): Tags to associate with the step.

**Returns:** The Pipeline instance for method chaining.

##### `run`

```python
run(variables=None, env=None, return_env=False)
```

**Parameters:**
- `variables` (dict, optional): Variables to use when running the pipeline.
- `env` (RunEnv, optional): Environment to use when running the pipeline.
- `return_env` (bool): Whether to return the environment instead of just variables.

**Returns:** Dictionary of variables or RunEnv instance.

##### `save`

```python
save(path)
```

**Parameters:**
- `path` (str): Path to save the pipeline to.

##### `get_subgraph`

```python
get_subgraph(inputs, outputs)
```

**Parameters:**
- `inputs` (list): Input variable names for the subgraph.
- `outputs` (list): Output variable names for the subgraph.

**Returns:** A new Pipeline containing only the steps needed to compute the outputs from the inputs.

##### `__getitem__`

```python
__getitem__(key)
```

**Parameters:**
- `key` (int, str, slice): Index, name, or slice to retrieve steps.

**Returns:** Step or list of Steps.

##### `__len__`

```python
__len__()
```

**Returns:** Number of steps in the pipeline.

### Loading a Pipeline

```python
from streamline.pipeline import load_pipeline

pipeline = load_pipeline(path)
```

**Parameters:**
- `path` (str): Path to load the pipeline from.

**Returns:** Loaded Pipeline instance.

## Steps

### Base Step Class

```python
from streamline.pipeline.step import _Step
```

Abstract base class for all step types.

### Function Step

```python
from streamline.pipeline.step import Function
```

#### Constructor

```python
Function(func, args=None, out_var=None)
```

**Parameters:**
- `func` (callable): The function to execute.
- `args` (str, list, optional): Input variable name(s).
- `out_var` (str, optional): Output variable name.

### Variable Step

```python
from streamline.pipeline.step import Var
```

#### Constructor

```python
Var(name)
```

**Parameters:**
- `name` (str): Variable name.

### Delete Step

```python
from streamline.pipeline.step import Delete
```

#### Constructor

```python
Delete(args)
```

**Parameters:**
- `args` (str, list): Names of variables to delete.

### Variables Dictionary Step

```python
from streamline.pipeline.step import VariablesDict
```

#### Constructor

```python
VariablesDict(**kwargs)
```

**Parameters:**
- `**kwargs`: Key-value pairs of variables to add.

## Delayed Evaluation

### Delayed Class

```python
from streamline.delayed import Delayed
```

#### Constructor

```python
Delayed(name, prefix='')
```

**Parameters:**
- `name` (str): Name of the variable.
- `prefix` (str, optional): Prefix for the variable reference.

#### Supported Operations

Delayed objects support most Python operators including:
- Arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Comparison operators: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Logical operators: `and`, `or`, `not`
- Container operations: `in`, indexing with `[]`, attribute access with `.`
- Function calls: `Delayed('func')(arg1, arg2)`

### Delay Library

```python
from streamline.delayed import delay_lib as d
```

Provides easy access to delayed variables and functions.

**Example:**
```python
x = d.some_variable
result = d.len(x) + 5
```

## Run Environment

### RunEnv Class

```python
from streamline.run_env import RunEnv
```

#### Constructor

```python
RunEnv(variables=None)
```

**Parameters:**
- `variables` (dict, optional): Initial variables.

#### Methods

##### `save`

```python
save(path, vars_to_save=None, options=None)
```

**Parameters:**
- `path` (str): Path to save the environment to.
- `vars_to_save` (list, optional): List of variables to save.
- `options` (dict, optional): Save options.

##### `load`

```python
@classmethod
load(path)
```

**Parameters:**
- `path` (str): Path to load the environment from.

**Returns:** Loaded RunEnv instance.

##### Dictionary-like Methods

RunEnv implements most dictionary methods, including:
- `__getitem__`, `__setitem__`, `__delitem__`
- `get`, `keys`, `values`, `items`
- `__contains__`, `__iter__`

## Models

### BaseModel Class

```python
from streamline.model import BaseModel
```

Abstract base class for models that can be used in pipelines.

#### Methods to Implement

##### `fit`

```python
fit(X, y=None, **kwargs)
```

Trains the model on the provided data.

**Parameters:**
- `X`: Training data.
- `y` (optional): Target values.
- `**kwargs`: Additional keyword arguments.

**Returns:** Self or fitted model.

##### `predict`

```python
predict(X, **kwargs)
```

Makes predictions using the model.

**Parameters:**
- `X`: Input data.
- `**kwargs`: Additional keyword arguments.

**Returns:** Predictions or dictionary of predictions.

## Utilities

### I/O Utilities

```python
from streamline.utilities.io import save_json, load_json, save_obj, load_obj, save_pandas_df, load_pandas_df
```

#### `save_json`

```python
save_json(d, path)
```

Saves a dictionary to a JSON file.

**Parameters:**
- `d` (dict): Dictionary to save.
- `path` (str): Path to save the file to.

#### `load_json`

```python
load_json(path)
```

Loads a dictionary from a JSON file.

**Parameters:**
- `path` (str): Path to load the file from.

**Returns:** Loaded dictionary.

#### `save_obj`

```python
save_obj(d, path)
```

Saves an object to a dill file.

**Parameters:**
- `d` (any): Object to save.
- `path` (str): Path to save the file to.

#### `load_obj`

```python
load_obj(path)
```

Loads an object from a dill file.

**Parameters:**
- `path` (str): Path to load the file from.

**Returns:** Loaded object.

#### `save_pandas_df`

```python
save_pandas_df(df, path, format='parquet', **kwargs)
```

Saves a pandas DataFrame to a file.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to save.
- `path` (str): Path to save the file to.
- `format` (str, optional): File format (parquet, csv, excel).
- `**kwargs`: Additional keyword arguments for the save function.

#### `load_pandas_df`

```python
load_pandas_df(path, format='parquet')
```

Loads a pandas DataFrame from a file.

**Parameters:**
- `path` (str): Path to load the file from.
- `format` (str, optional): File format (parquet, csv, excel).

**Returns:** Loaded DataFrame.

### Serialization Utilities

```python
from streamline.utilities.serialize import get_global_dependencies, mainify
```

#### `get_global_dependencies`

```python
get_global_dependencies(func)
```

Identifies global variable names accessed by a function.

**Parameters:**
- `func` (callable): Function to analyze.

**Returns:** Set of global variable names.

#### `mainify`

```python
mainify(obj)
```

Makes an object available in the `__main__` module to make it picklable.

**Parameters:**
- `obj` (any): Object to make available in `__main__`.

**Returns:** Object reference. 
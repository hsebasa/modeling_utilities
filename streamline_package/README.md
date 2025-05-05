# Streamline

Streamline is a powerful, flexible Python library for building and running computational pipelines. It provides a clean interface for defining, executing, and managing complex data workflows with dependency tracking and lazy evaluation.

## Features

- **Pipeline-based workflow** - Build modular data processing pipelines step by step
- **Variable tracking** - Automatically track dependencies between pipeline steps
- **Lazy evaluation** - Define operations that are executed only when needed
- **Serialization** - Save and load pipelines for later use
- **Extensibility** - Easy to add custom steps and integrate with other libraries
- **Subgraph extraction** - Extract portions of pipelines for targeted execution
- **Pipeline validation** - Verify that your pipeline will run correctly before execution

## Installation

```bash
pip install streamline
```

## Quick Start

Here's a simple example to get you started:

```python
from streamline.pipeline import Pipeline
from streamline.pipeline.step import Function, Var, VariablesDict

# Create a new pipeline
pipeline = Pipeline()

# Add steps to the pipeline
pipeline.add_variables_dict({'input_data': [1, 2, 3, 4, 5]})  # Define input
pipeline.add_function(lambda x: [i * 2 for i in x], 'input_data', 'doubled')  # Process
pipeline.add_function(lambda x: sum(x), 'doubled', 'total')  # Aggregate

# Run the pipeline
result = pipeline.run()

# Get results
print(result['doubled'])  # [2, 4, 6, 8, 10]
print(result['total'])    # 30
```

## Core Concepts

### Pipeline

The `Pipeline` class is the central component for building computation workflows. It manages a sequence of steps that process data in a specific order.

```python
from streamline.pipeline import Pipeline

# Create an empty pipeline
pipeline = Pipeline()

# Add steps (see examples below)
# ...

# Run the pipeline
result = pipeline.run({'input_var': value})
```

### Steps

Steps are the building blocks of pipelines. Streamline includes several types of steps:

- **Function** - Execute a Python function
- **Delete** - Remove variables from the environment
- **VariablesDict** - Add variables to the environment
- **ImportLib** - Import Python modules

### Variables and Dependencies

Streamline automatically tracks dependencies between steps, ensuring that each step has access to the variables it needs:

```python
from streamline.pipeline import Pipeline
from streamline.pipeline.step import Function, Var

pipeline = Pipeline([
    Function(lambda x: x + 1, 'input', 'intermediate'),
    Function(lambda y: y * 2, 'intermediate', 'output')
])

result = pipeline.run({'input': 5})
print(result['output'])  # (5 + 1) * 2 = 12
```

## Examples

### Basic Data Transformation

```python
from streamline.pipeline import Pipeline

# Create a pipeline for data transformation
pipeline = Pipeline()

# Add steps
pipeline.add_function(lambda x: x + 1, 'input', 'step1')
pipeline.add_function(lambda x: x * 2, 'step1', 'step2')
pipeline.add_function(lambda x: x ** 2, 'step2', 'result')

# Run with input
result = pipeline.run({'input': 5})
print(result['result'])  # ((5 + 1) * 2) ** 2 = 144
```

### Importing Libraries

```python
from streamline.pipeline import Pipeline

# Create a pipeline
pipeline = Pipeline()

# Import libraries
pipeline.add_import_lib({'numpy': 'np', 'pandas': 'pd'})

# Use imported libraries
pipeline.add_function(lambda np: np.array([1, 2, 3]), 'np', 'array')
pipeline.add_function(lambda pd, array: pd.DataFrame(array, columns=['value']), 
                     ['pd', 'array'], 'df')

# Run the pipeline
result = pipeline.run()
print(result['df'])
```

### Adding Variables

```python
from streamline.pipeline import Pipeline

# Create a pipeline
pipeline = Pipeline()

# Add variables
pipeline.add_variables_dict({
    'config': {'rate': 0.05, 'periods': 12},
    'principal': 1000
})

# Use those variables
pipeline.add_function(
    lambda principal, config: principal * (1 + config['rate']) ** config['periods'],
    ['principal', 'config'],
    'final_amount'
)

# Run the pipeline
result = pipeline.run()
print(result['final_amount'])  # Compound interest calculation
```

### Subgraph Extraction

```python
from streamline.pipeline import Pipeline

# Create a complex pipeline
pipeline = Pipeline()
pipeline.add_variables_dict({'initial': 10})
pipeline.add_function(lambda initial: initial * 2, 'initial', 'doubled')
pipeline.add_function(lambda doubled: doubled + 5, 'doubled', 'intermediate')
pipeline.add_function(lambda intermediate: intermediate ** 2, 'intermediate', 'squared')
pipeline.add_function(lambda squared: squared / 10, 'squared', 'result')

# Extract a subgraph from 'doubled' to 'squared'
subgraph = pipeline.get_subgraph(['doubled'], ['squared'])

# Run just the subgraph
result = subgraph.run({'doubled': 20})
print(result['squared'])  # (20 + 5) ** 2 = 625
```

### Pipeline Validation

```python
from streamline.pipeline import Pipeline

# Create a pipeline with a dependency
pipeline = Pipeline()
pipeline.add_function(lambda x, y: x + y, ['x', 'y'], 'result')

# Validate the pipeline with a given environment
try:
    pipeline.validate({'x': 5})  # Missing 'y'
except AssertionError as e:
    print(f"Validation failed: {e}")
    
# Provide all required inputs
pipeline.validate({'x': 5, 'y': 10})  # Validation passes
```

### Saving and Loading Pipelines

```python
from streamline.pipeline import Pipeline, load_pipeline

# Create and configure a pipeline
pipeline = Pipeline()
pipeline.add_function(lambda x: x + 1, 'input', 'output')

# Save the pipeline to a file
pipeline.save('my_pipeline.dill')

# Later, load the pipeline
loaded_pipeline = load_pipeline('my_pipeline.dill')

# Use the loaded pipeline
result = loaded_pipeline.run({'input': 5})
print(result['output'])  # 6
```

### Working with Machine Learning Models

```python
from streamline.pipeline import Pipeline
from streamline.pipeline.modeling import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

# Create a wrapper model
class SklearnModel(BaseModel):
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)

# Create a pipeline
pipeline = Pipeline()

# Add data preparation steps
pipeline.add_variables_dict({
    'X': np.array([[1], [2], [3], [4], [5]]),
    'y': np.array([2, 4, 6, 8, 10])
})

# Add model
model = SklearnModel(LinearRegression())
pipeline.add_model(
    model=model,
    fit_args=['X', 'y'],
    predict_args=['X'],
    predict_out_var='predictions'
)

# Run pipeline
result = pipeline.run()
print(result['predictions'])  # Predictions from the linear model
```

## Advanced Features

### Delayed Evaluation

```python
from streamline.delayed import delay, delay_lib as dl

# Define a delayed computation
expr = delay('x') + delay('y') * 2

# Create a function that evaluates the expression
def compute(x, y):
    return expr.eval({'x': x, 'y': y})

# Use in pipeline
pipeline = Pipeline()
pipeline.add_function(compute, ['a', 'b'], 'result')
result = pipeline.run({'a': 5, 'b': 10})
print(result['result'])  # 5 + 10 * 2 = 25
```

### Run Environment Customization

```python
from streamline.run_env import RunEnv
from streamline.pipeline import Pipeline

# Create a custom environment
env = RunEnv({'initial_value': 100})

# Add a tracking callback
def track_step(step, result, env):
    print(f"Executed step: {step}")
    print(f"Current vars: {list(env.keys())}")

env.add_callback(track_step)

# Use the environment
pipeline = Pipeline()
pipeline.add_function(lambda x: x + 1, 'initial_value', 'result')
pipeline.run(env)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

```
python -m unittest discover
```

## Graphical report
Run graphical tests by installing nose
```
pip install nose2[coverage_plugin]
pip install nose2-html-report
```

And then
```
nose2 --with-coverage --coverage-report html --coverage ./
```
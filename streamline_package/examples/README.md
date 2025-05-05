# Streamline Package Examples

This directory contains example notebooks demonstrating how to use the Streamline package for different use cases.

## Available Examples

### 1. Basic Pipeline
**File**: [`basic_pipeline.ipynb`](basic_pipeline.ipynb)

This notebook demonstrates the core functionality of the Streamline package, including:
- Creating a basic pipeline
- Adding variables and processing steps
- Running the pipeline with different inputs
- Saving and loading pipelines
- Extracting subgraphs from a pipeline

Perfect for beginners who want to understand the fundamentals of the Streamline package.

### 2. Machine Learning Pipeline
**File**: [`ml_pipeline.ipynb`](ml_pipeline.ipynb)

This notebook shows how to use Streamline for machine learning workflows:
- Creating custom model classes by extending `BaseModel`
- Building pipelines for data preprocessing, model training, and evaluation
- Using the `add_model` method to integrate models into pipelines
- Extracting model results and feature importances
- Comparing multiple models in different pipelines

Ideal for data scientists and ML engineers looking to streamline their model development and evaluation process.

### 3. Delayed Evaluation
**File**: [`delayed_evaluation.ipynb`](delayed_evaluation.ipynb)

This notebook explores the delayed evaluation feature of Streamline:
- Creating delayed expressions using `Delayed` and `delay_lib`
- Building pipelines with concise, readable code using delayed expressions
- Accessing nested attributes and performing complex operations
- Using built-in and custom functions in delayed expressions
- Working with various data types and structures in delayed mode

Great for users who want to write more expressive and maintainable pipeline code.

## Running the Examples

To run these examples, make sure you have the Streamline package installed:

```bash
pip install streamline-package
```

Then open the notebooks in Jupyter:

```bash
jupyter notebook
```

## Additional Resources

- [Streamline Package Documentation](https://github.com/yourusername/streamline_package)
- [README](../README.md): Main package documentation
- [API Reference](../docs/api_reference.md): Detailed API documentation 